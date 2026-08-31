"""
unk_audit.py

이미 크롤링된 2017~2025년(9개 연도) pbp_full/{year}/*.csv를 2017년에서 발견한
7개 패턴(HBP/내야안타/실책출루/IBB/번트안타/야수선택/기타실책변형)으로
연도별 분류하고, 설명되지 않는 잔여 UNK를 리포트한다.

추가로 "게임의 진짜 마지막 PA가 UNK인 경우"(state_transition의
_estimate_runs_from_pa_result 폴백이 적용되는 유일한 경로)를 식별하고,
그 경기의 실제 최종 스코어(schedule API)와 우리가 계산한 마지막 score_diff를
대조해 runs_scored=0 폴백이 실제로 맞았는지 검증한다.

산출: data_analysis/results/unk_audit.md

실행:
    uv run python unk_audit.py
"""

from __future__ import annotations

import asyncio
import glob
import os

import aiohttp
import pandas as pd

from data_analysis.methods.pa_aggregator import _classify_pa_result

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
STAGING_DIR = "data_analysis/results/staging"
PBP_DIR_TMPL = "data/pbp_full/{year}"
REPORT_PATH = "data_analysis/results/unk_audit.md"

PATTERNS = {
    "HBP(몸에 맞는 볼)": r"몸에 맞는 볼",
    "IBB(고의4구)": r"고의4구",
    "내야안타": r"내야안타",
    "번트안타": r"번트안타",
    "실책으로 출루": r"실책으로 출루",
    "야수선택으로 출루": r"야수선택으로 출루",
    "기타 실책 변형(플라이실책/타격방해)": r"플라이 실책|타격방해로 출루",
}

SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://sports.naver.com/",
}


def load_year_pitch_df(year: int) -> pd.DataFrame:
    csvs = glob.glob(os.path.join(PBP_DIR_TMPL.format(year=year), "*.csv"))
    dfs = [pd.read_csv(f, low_memory=False) for f in csvs]
    return pd.concat(dfs, ignore_index=True)


def audit_year(year: int) -> dict:
    df = load_year_pitch_df(year)
    df["pa_result"] = df["relay_text"].apply(_classify_pa_result)

    # PA 단위로 축약(중복 pitch 행 제거) — relay_text가 있는 마지막 행 기준
    key_cols = ["game_id", "inning", "pitcher_id", "batter_id"]
    df["_pa_seq"] = (df[key_cols].ne(df[key_cols].shift()).any(axis=1)).cumsum()
    pa_last = df.groupby("_pa_seq").agg(
        game_id=("game_id", "last"), relay_text=("relay_text", "last"), pa_result=("pa_result", "last"),
    )

    n_pa = len(pa_last)
    unk = pa_last[pa_last["pa_result"] == "UNK"]
    n_unk = len(unk)

    remaining = unk.copy()
    pattern_counts = {}
    for name, pat in PATTERNS.items():
        mask = remaining["relay_text"].str.contains(pat, na=False, regex=True)
        pattern_counts[name] = int(mask.sum())
        remaining = remaining[~mask]

    return {
        "year": year, "n_pa": n_pa, "n_unk": n_unk,
        "unk_rate": n_unk / n_pa if n_pa else 0.0,
        "pattern_counts": pattern_counts,
        "n_remaining": len(remaining),
        "remaining_samples": remaining["relay_text"].dropna().unique()[:20].tolist(),
    }


async def fetch_final_score(session: aiohttp.ClientSession, game_id: str) -> dict | None:
    date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"
    params = {
        "fields": "basic,schedule,baseball,manualRelayUrl", "upperCategoryId": "kbaseball",
        "categoryId": "kbo", "fromDate": date_str, "toDate": date_str, "size": 500,
    }
    try:
        async with session.get(SCHEDULE_URL, params=params, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except Exception:
        return None
    for g in data.get("result", {}).get("games", []):
        if g.get("gameId") == game_id:
            return {"home": g.get("homeTeamScore"), "away": g.get("awayTeamScore")}
    return None


async def audit_terminal_unk_runs(years: list[int]) -> pd.DataFrame:
    """게임의 진짜 마지막 PA가 UNK인 경우, 실제 최종 스코어와 대조해
    runs_scored=0 폴백이 맞았는지 검증한다."""
    rows = []
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for year in years:
            path = os.path.join(STAGING_DIR, f"pa_states_{year}.parquet")
            if not os.path.isfile(path):
                continue
            df = pd.read_parquet(path).sort_values("_orig_idx").reset_index(drop=True)
            last_idx = df.groupby("game_id")["_orig_idx"].idxmax()
            last_rows = df.loc[last_idx]
            terminal_unk = last_rows[last_rows["pa_result"] == "UNK"]

            for _, row in terminal_unk.iterrows():
                truth = await fetch_final_score(session, row["game_id"])
                await asyncio.sleep(1.0)
                if truth is None or truth["home"] is None:
                    rows.append({"year": year, "game_id": row["game_id"], "status": "truth_unavailable"})
                    continue

                true_final_diff = int(truth["home"]) - int(truth["away"])
                # 우리 score_diff는 H1 컨벤션(home-away). runs_scored=0 폴백이면
                # 마지막 PA 시작 시점 score_diff가 곧 최종 score_diff와 같아야 정상.
                half = row["half"]
                sda = int(row["score_diff_attacker"])  # 공격팀 관점
                our_final_home_away_diff = sda if half == "bot" else -sda

                match = our_final_home_away_diff == true_final_diff
                rows.append({
                    "year": year, "game_id": row["game_id"], "status": "checked",
                    "our_diff_at_pa_start": our_final_home_away_diff,
                    "true_final_diff": true_final_diff, "match": match,
                })

    return pd.DataFrame(rows)


def build_report(results: list[dict], terminal_check: pd.DataFrame) -> str:
    lines = ["# UNK 정규식 커버리지 감사 (2017-2025)\n"]

    lines.append("## 연도 × 패턴 매트릭스\n")
    header = ["year", "n_pa", "n_unk", "unk_rate"] + list(PATTERNS.keys()) + ["잔여(미설명)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in results:
        row = [
            str(r["year"]), f"{r['n_pa']:,}", f"{r['n_unk']:,}", f"{r['unk_rate']*100:.2f}%",
        ] + [str(r["pattern_counts"][k]) for k in PATTERNS] + [str(r["n_remaining"])]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 연도별 잔여(미설명) UNK 샘플\n")
    for r in results:
        if r["n_remaining"] == 0:
            continue
        lines.append(f"### {r['year']}년 (잔여 {r['n_remaining']}건)\n")
        lines.append("```")
        for t in r["remaining_samples"]:
            lines.append(repr(t))
        lines.append("```\n")

    lines.append("## 게임 마지막 PA가 UNK인 경우 — 실제 최종 스코어 대조\n")
    lines.append(f"검증 대상: {len(terminal_check)}건\n")
    if len(terminal_check):
        checked = terminal_check[terminal_check["status"] == "checked"]
        n_mismatch = int((~checked["match"]).sum()) if len(checked) else 0
        lines.append(f"- 검증 가능: {len(checked)}건 / 불일치(실제 득점 누락 의심): {n_mismatch}건\n")
        lines.append("| year | game_id | 우리 계산(마지막 PA 시점) | 실제 최종 스코어차 | 일치 |")
        lines.append("|---|---|---|---|---|")
        for _, row in terminal_check.iterrows():
            if row["status"] != "checked":
                lines.append(f"| {row['year']} | {row['game_id']} | - | - | 조회실패 |")
                continue
            lines.append(
                f"| {row['year']} | {row['game_id']} | {row['our_diff_at_pa_start']:+d} | "
                f"{row['true_final_diff']:+d} | {'O' if row['match'] else '**X**'} |"
            )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    results = [audit_year(y) for y in YEARS]

    for r in results:
        print(f"{r['year']}: PA={r['n_pa']:,} UNK={r['n_unk']:,} ({r['unk_rate']*100:.2f}%) 잔여={r['n_remaining']}")

    terminal_check = asyncio.run(audit_terminal_unk_runs(YEARS))

    report = build_report(results, terminal_check)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n저장 완료: {REPORT_PATH}")

    n_mismatch = int((~terminal_check.loc[terminal_check["status"] == "checked", "match"]).sum())
    print(f"\n[핵심] 마지막 PA UNK 폴백 검증: 불일치(득점 누락 의심) {n_mismatch}건")


if __name__ == "__main__":
    main()
