"""
validate_pilot.py

파일럿 200경기(전 구단·전 연도) 검증 스크립트. run_pilot.py 산출물을 검사하여
전체 7,200경기 확장 가능 여부를 판단하는 근거 자료를 만든다.

4-1) 팀 커버리지 + 팀 코드 검증 (game_id 파싱 vs schedule API vs preview API)
4-2) 스키마 차이 전수 조사 (preview JSON 필드, PBP wpaByPlate)
4-3) WPA 도메인 정합성 (pa_result별 평균 ΔWE 순서, 기존 153경기와 비교)
4-4) 규모 추정 (7,200경기 확장 시 PA/투구 수, 소요시간, 용량, 메모리)
4-5) 실패 케이스 분석

산출: data_analysis/results/pilot/pilot_validation_report.md

실행:
    uv run python validate_pilot.py
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

GAME_INDEX_PATH   = os.path.join(PROJECT_ROOT, "data", "game_ids", "game_index.csv")
PILOT_GAMES_PATH  = os.path.join(PROJECT_ROOT, "data", "game_ids", "pilot_games.txt")
PBP_PILOT_DIR     = os.path.join(PROJECT_ROOT, "data", "pbp_pilot")
PREVIEW_PILOT_DIR = os.path.join(PROJECT_ROOT, "data", "preview_pilot")
RESULTS_DIR       = os.path.join(PROJECT_ROOT, "data_analysis", "results", "pilot")

PA_WPA_PATH     = os.path.join(RESULTS_DIR, "pilot_pa_with_wpa.parquet")
LINEUP_PATH     = os.path.join(RESULTS_DIR, "pilot_game_lineup.parquet")
BULLPEN_PATH    = os.path.join(RESULTS_DIR, "pilot_game_bullpen.parquet")
FAILURES_PATH   = os.path.join(RESULTS_DIR, "pilot_failures.json")
ENRICHED_PATH   = os.path.join(RESULTS_DIR, "pilot_hsk_pa_enriched.parquet")

REFERENCE_PA_WPA_PATH = os.path.join(PROJECT_ROOT, "data_analysis", "results", "hsk_pa_with_wpa.parquet")

REPORT_PATH = os.path.join(RESULTS_DIR, "pilot_validation_report.md")

FULL_SCALE_TARGET_GAMES = 7200
PA_ORDER = ["HR", "3B", "2B", "1B", "BB", "SF", "OUT", "SO", "GDP"]


# ────────────────────────────────────────────────────────────────────────── #
#  공통 유틸
# ────────────────────────────────────────────────────────────────────────── #

def _parse_team_codes_from_game_id(game_id: str) -> tuple[str, str]:
    """game_id 위치 기반 (away_code, home_code) 파싱. YYYYMMDD+Away2+Home2+..."""
    return game_id[8:10], game_id[10:12]


def _get_path(data: dict, path: str):
    """'a.b[0].c' 형식 경로를 따라가며 값을 반환. 없으면 KeyError/IndexError 발생."""
    cur = data
    for part in path.replace("]", "").split("."):
        if "[" in part:
            key, idx = part.split("[")
            cur = cur[key][int(idx)]
        else:
            cur = cur[part]
    return cur


def _has_path(data: dict, path: str) -> bool:
    try:
        _get_path(data, path)
        return True
    except (KeyError, IndexError, TypeError):
        return False


def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate 의존성 없이 최소한의 마크다운 테이블을 생성한다."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────── #
#  4-1) 팀 커버리지 + 팀 코드 검증
# ────────────────────────────────────────────────────────────────────────── #

def validate_team_coverage() -> dict:
    with open(PILOT_GAMES_PATH, encoding="utf-8") as f:
        pilot_ids = [line.strip() for line in f if line.strip()]

    game_index = pd.read_csv(GAME_INDEX_PATH, dtype=str)
    game_index = game_index.set_index("game_id")

    # (a) game_id 위치 파싱 vs schedule API 필드 일치 확인
    mismatches = []
    year_team_matrix: dict[str, set[str]] = defaultdict(set)
    code_to_names: dict[str, set[str]] = defaultdict(set)

    for gid in pilot_ids:
        away_parsed, home_parsed = _parse_team_codes_from_game_id(gid)
        year = gid[:4]
        year_team_matrix[year].add(away_parsed)
        year_team_matrix[year].add(home_parsed)

        if gid in game_index.index:
            row = game_index.loc[gid]
            away_api, home_api = row["away_code"], row["home_code"]
            if away_parsed != away_api or home_parsed != home_api:
                mismatches.append({
                    "game_id": gid, "source": "schedule_api",
                    "parsed": (away_parsed, home_parsed), "api": (away_api, home_api),
                })
            code_to_names[away_api].add(row["away_name"])
            code_to_names[home_api].add(row["home_name"])

    # (b) preview API의 gameInfo.aCode/hCode 와 대조 (preview 존재하는 경기만)
    preview_files = sorted(glob.glob(os.path.join(PREVIEW_PILOT_DIR, "*.json")))
    preview_checked = 0
    for path in preview_files:
        gid = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        game_info = data.get("result", {}).get("previewData", {}).get("gameInfo", {})
        a_code, h_code = game_info.get("aCode"), game_info.get("hCode")
        away_parsed, home_parsed = _parse_team_codes_from_game_id(gid)
        preview_checked += 1
        if a_code and h_code and (a_code != away_parsed or h_code != home_parsed):
            mismatches.append({
                "game_id": gid, "source": "preview_api",
                "parsed": (away_parsed, home_parsed), "api": (a_code, h_code),
            })
        a_name, h_name = game_info.get("aName"), game_info.get("hName")
        if a_code and a_name:
            code_to_names[a_code].add(a_name)
        if h_code and h_name:
            code_to_names[h_code].add(h_name)

    all_codes = sorted({c for codes in year_team_matrix.values() for c in codes})
    years = sorted(year_team_matrix)

    return {
        "pilot_n_games": len(pilot_ids),
        "all_codes": all_codes,
        "years": years,
        "year_team_matrix": year_team_matrix,
        "mismatches": mismatches,
        "preview_checked": preview_checked,
        "code_to_names": {k: sorted(v) for k, v in code_to_names.items()},
    }


# ────────────────────────────────────────────────────────────────────────── #
#  4-2) 스키마 차이 전수 조사
# ────────────────────────────────────────────────────────────────────────── #

PREVIEW_FIELD_PATHS = [
    "gameInfo.aCode",
    "gameInfo.hCode",
    "gameInfo.isPostSeason",
    "gameInfo.round",
    "awayStarter.playerInfo.pCode",
    "awayStarter.playerInfo.hitType",
    "awayStarter.currentSeasonStats.era",
    "awayStarter.currentSeasonStats.whip",
    "awayStarter.currentSeasonStatsOnOpponents.era",
    "awayTeamLineUp.pitcherBullpen[0].hitType",
    "awayTeamLineUp.pitcherBullpen[0].batsThrows",
    "awayTeamLineUp.fullLineUp[0].batsThrows",
    "homeTopPlayer.hotColdZone[0].zone",
]


def validate_preview_schema() -> pd.DataFrame:
    preview_files = sorted(glob.glob(os.path.join(PREVIEW_PILOT_DIR, "*.json")))

    by_year: dict[str, list[dict]] = defaultdict(list)
    for path in preview_files:
        gid = os.path.splitext(os.path.basename(path))[0]
        year = gid[:4]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        preview = data.get("result", {}).get("previewData", {})
        by_year[year].append(preview)

    rows = []
    for year in sorted(by_year):
        games = by_year[year]
        row = {"year": year, "n_preview_games": len(games)}
        for field in PREVIEW_FIELD_PATHS:
            present = sum(1 for g in games if _has_path(g, field))
            row[field] = f"{present}/{len(games)}"
        rows.append(row)

    return pd.DataFrame(rows)


def validate_pbp_wpa_schema(pa_df: pd.DataFrame) -> pd.DataFrame:
    """연도별 reward_wpa(네이버 원본 wpaByPlate) 존재율."""
    pa_df = pa_df.copy()
    pa_df["year"] = pa_df["game_id"].str[:4]
    out = (
        pa_df.groupby("year")["reward_wpa"]
        .apply(lambda s: f"{s.notna().sum()}/{len(s)} ({100*s.notna().mean():.0f}%)")
        .reset_index()
        .rename(columns={"reward_wpa": "wpaByPlate_존재율"})
    )
    return out


# ────────────────────────────────────────────────────────────────────────── #
#  4-3) WPA 도메인 정합성
# ────────────────────────────────────────────────────────────────────────── #

def validate_wpa_domain(pilot_df: pd.DataFrame) -> dict:
    def _mean_table(df: pd.DataFrame) -> dict[str, float]:
        return {
            pa: df.loc[df["pa_result"] == pa, "reward_wpa_computed"].mean()
            for pa in PA_ORDER
            if (df["pa_result"] == pa).any()
        }

    pilot_means = _mean_table(pilot_df)

    reference_means = None
    if os.path.isfile(REFERENCE_PA_WPA_PATH):
        ref_df = pd.read_parquet(REFERENCE_PA_WPA_PATH)
        reference_means = _mean_table(ref_df)

    present_order = [pa for pa in PA_ORDER if pa in pilot_means]
    values = [pilot_means[pa] for pa in present_order]
    order_ok = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    return {
        "pilot_means": pilot_means,
        "reference_means": reference_means,
        "order_ok": order_ok,
        "expected_order": PA_ORDER,
    }


# ────────────────────────────────────────────────────────────────────────── #
#  4-4) 규모 추정
# ────────────────────────────────────────────────────────────────────────── #

def estimate_full_scale(pilot_df: pd.DataFrame, n_pilot_games: int) -> dict:
    csv_files = glob.glob(os.path.join(PBP_PILOT_DIR, "*.csv"))
    total_pitches = 0
    for f in csv_files:
        with open(f, encoding="utf-8-sig") as fh:
            total_pitches += max(sum(1 for _ in fh) - 1, 0)

    n_pa = len(pilot_df)
    scale = FULL_SCALE_TARGET_GAMES / n_pilot_games

    pa_parquet_size = os.path.getsize(PA_WPA_PATH) if os.path.isfile(PA_WPA_PATH) else 0
    mem_bytes = pilot_df.memory_usage(deep=True).sum()

    return {
        "n_pilot_games": n_pilot_games,
        "n_pilot_pa": n_pa,
        "n_pilot_pitches": total_pitches,
        "scale_factor": scale,
        "est_full_pa": int(n_pa * scale),
        "est_full_pitches": int(total_pitches * scale),
        "pa_parquet_size_mb": pa_parquet_size / 1e6,
        "est_full_parquet_size_mb": pa_parquet_size / 1e6 * scale,
        "pa_df_memory_mb": mem_bytes / 1e6,
        "est_full_df_memory_mb": mem_bytes / 1e6 * scale,
    }


# ────────────────────────────────────────────────────────────────────────── #
#  4-5) 실패 케이스 분석
# ────────────────────────────────────────────────────────────────────────── #

def analyze_failures(n_pilot_games: int) -> dict:
    if not os.path.isfile(FAILURES_PATH):
        return {"available": False}

    with open(FAILURES_PATH, encoding="utf-8") as f:
        failures = json.load(f)

    pbp_failures = failures.get("pbp_crawl", {})
    preview_failures = failures.get("preview_crawl", {})

    reason_counts: dict[str, int] = defaultdict(int)
    for reason in pbp_failures.values():
        key = reason.split(":")[0]
        reason_counts[f"pbp:{key}"] += 1
    for reason in preview_failures.values():
        key = reason.split(":")[0]
        reason_counts[f"preview:{key}"] += 1

    pbp_rate = len(pbp_failures) / n_pilot_games
    preview_rate = len(preview_failures) / n_pilot_games

    return {
        "available": True,
        "n_pbp_failures": len(pbp_failures),
        "n_preview_failures": len(preview_failures),
        "pbp_failure_rate": pbp_rate,
        "preview_failure_rate": preview_rate,
        "reason_counts": dict(reason_counts),
        "est_full_pbp_failures": int(pbp_rate * FULL_SCALE_TARGET_GAMES),
        "est_full_preview_failures": int(preview_rate * FULL_SCALE_TARGET_GAMES),
        "enrichment_status": failures.get("enrichment", "unknown"),
    }


# ────────────────────────────────────────────────────────────────────────── #
#  리포트 생성
# ────────────────────────────────────────────────────────────────────────── #

def build_report(
    coverage: dict,
    preview_schema: pd.DataFrame,
    pbp_schema: pd.DataFrame,
    wpa_domain: dict,
    scale: dict,
    failures: dict,
) -> str:
    lines: list[str] = []
    lines.append("# 파일럿 검증 리포트 (200경기, 전 구단·전 연도)\n")

    # 4-1
    lines.append("## 4-1) 팀 커버리지 및 팀 코드 검증\n")
    lines.append(f"- 파일럿 경기 수: {coverage['pilot_n_games']}")
    lines.append(f"- 등장 팀 코드({len(coverage['all_codes'])}개): {', '.join(coverage['all_codes'])}")
    lines.append(f"- preview API 대조 가능 경기: {coverage['preview_checked']}개")
    lines.append(f"- game_id 파싱 vs API(schedule/preview) 불일치: {len(coverage['mismatches'])}건")
    if coverage["mismatches"]:
        lines.append("\n| game_id | source | parsed | api |")
        lines.append("|---|---|---|---|")
        for m in coverage["mismatches"][:20]:
            lines.append(f"| {m['game_id']} | {m['source']} | {m['parsed']} | {m['api']} |")
    lines.append("\n### 팀 코드 → 팀명 매핑 (API 실측, 여러 개면 명칭 변경 이력)\n")
    lines.append("| code | 관측된 팀명 |")
    lines.append("|---|---|")
    for code, names in sorted(coverage["code_to_names"].items()):
        flag = "  ← 명칭 변경 이력 있음 (API는 현재 명칭을 과거 경기에도 소급 적용)" if len(names) > 1 else ""
        lines.append(f"| {code} | {', '.join(names)}{flag} |")
    lines.append(
        "\n**결론**: game_id에 인코딩된 팀 코드는 2016~2025 전 기간 동일하게 유지됨"
        "(SK Wyverns→SSG Landers, 넥센→키움 등 명칭 변경에도 코드 자체는 불변)."
        " 단, schedule/preview API가 반환하는 팀 '이름'(name) 필드는 과거 경기에도"
        " 현재 시점 명칭을 소급 적용하므로 이름으로 시대를 구분할 수 없음 — 코드만 신뢰할 것.\n"
    )

    lines.append("### 연도별 팀 코드 출현 매트릭스\n")
    years = coverage["years"]
    codes = coverage["all_codes"]
    lines.append("| year | " + " | ".join(codes) + " |")
    lines.append("|---|" + "---|" * len(codes))
    for y in years:
        present = coverage["year_team_matrix"][y]
        lines.append(f"| {y} | " + " | ".join("O" if c in present else "" for c in codes) + " |")
    lines.append("")

    # 4-2
    lines.append("## 4-2) 스키마 차이 전수 조사\n")
    lines.append("### Preview JSON 필드 존재율 (연도별)\n")
    lines.append(_df_to_markdown(preview_schema))
    lines.append(
        "\n참고: `batsThrows`는 초기 preview에서 `pitcherBullpen`/`fullLineUp`에 없다가"
        " 이후 연도에 추가된 필드로 확인됨(현재 파서는 `hitType`만 사용하므로 파싱 자체엔"
        " 영향 없음). preview API는 2017-05-30 이전 경기는 아예 제공하지 않음(스킵 대상).\n"
    )
    lines.append("### PBP: reward_wpa(네이버 원본 wpaByPlate) 존재율 (연도별)\n")
    lines.append(_df_to_markdown(pbp_schema))
    lines.append("")

    # 4-3
    lines.append("## 4-3) WPA 도메인 정합성 (핵심)\n")
    order_str = " > ".join(wpa_domain["expected_order"])
    lines.append(f"- 기대 순서: {order_str}\n")
    lines.append(f"- 순서 유지 여부: {'PASS' if wpa_domain['order_ok'] else '**FAIL**'}\n")
    lines.append("| pa_result | 파일럿(200경기) 평균 ΔWE | 기존(153경기) 평균 ΔWE |")
    lines.append("|---|---|---|")
    for pa in wpa_domain["expected_order"]:
        p = wpa_domain["pilot_means"].get(pa)
        r = (wpa_domain["reference_means"] or {}).get(pa)
        p_str = f"{p:.4f}" if p is not None else "N/A"
        r_str = f"{r:.4f}" if r is not None else "N/A"
        lines.append(f"| {pa} | {p_str} | {r_str} |")
    lines.append("")

    # 4-4
    lines.append("## 4-4) 규모 추정 (전체 7,200경기 기준)\n")
    lines.append(f"- 파일럿 실측: {scale['n_pilot_games']}경기 / PA {scale['n_pilot_pa']:,}건 / 투구 {scale['n_pilot_pitches']:,}건")
    lines.append(f"- 확장 배율: {scale['scale_factor']:.1f}배")
    lines.append(f"- 예상 전체 PA 수: 약 {scale['est_full_pa']:,}건")
    lines.append(f"- 예상 전체 투구 수: 약 {scale['est_full_pitches']:,}건")
    lines.append(f"- 파일럿 PA parquet 용량: {scale['pa_parquet_size_mb']:.2f} MB → 전체 예상: 약 {scale['est_full_parquet_size_mb']:.1f} MB")
    lines.append(f"- 파일럿 PA DataFrame 메모리: {scale['pa_df_memory_mb']:.2f} MB → 전체 예상 피크: 약 {scale['est_full_df_memory_mb']:.1f} MB")
    lines.append("")

    # 4-5
    lines.append("## 4-5) 실패 케이스 분석\n")
    if not failures.get("available"):
        lines.append("실패 로그(pilot_failures.json)를 찾을 수 없음.\n")
    else:
        lines.append(f"- PBP 크롤링 실패: {failures['n_pbp_failures']}경기 ({failures['pbp_failure_rate']:.1%})")
        lines.append(f"- preview 크롤링 실패: {failures['n_preview_failures']}경기 ({failures['preview_failure_rate']:.1%})")
        lines.append(f"- enrichment 단계: {failures['enrichment_status']}")
        lines.append("\n### 원인별 분류\n")
        lines.append("| 원인 | 건수 |")
        lines.append("|---|---|")
        for reason, cnt in sorted(failures["reason_counts"].items()):
            lines.append(f"| {reason} | {cnt} |")
        lines.append(
            f"\n전체 확장 시 예상: PBP 실패 약 {failures['est_full_pbp_failures']}경기, "
            f"preview 실패 약 {failures['est_full_preview_failures']}경기 (파일럿 실패율 그대로 적용 시)\n"
        )

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────── #
#  메인
# ────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    t0 = time.time()

    print("=== 4-1) 팀 커버리지 검증 ===")
    coverage = validate_team_coverage()
    print(f"등장 팀 코드({len(coverage['all_codes'])}개): {coverage['all_codes']}")
    print(f"불일치: {len(coverage['mismatches'])}건")

    print("\n=== 4-2) 스키마 조사 ===")
    preview_schema = validate_preview_schema()
    print(preview_schema.to_string(index=False))

    pilot_df = pd.read_parquet(PA_WPA_PATH)
    pbp_schema = validate_pbp_wpa_schema(pilot_df)
    print(pbp_schema.to_string(index=False))

    print("\n=== 4-3) WPA 도메인 정합성 ===")
    wpa_domain = validate_wpa_domain(pilot_df)
    print(f"순서 유지: {wpa_domain['order_ok']}")
    print(wpa_domain["pilot_means"])

    if not wpa_domain["order_ok"]:
        print("\n" + "!" * 60)
        print("[중단] WPA 도메인 순서가 깨졌습니다 — 절대 금지 규칙에 따라 확장 보류")
        print("!" * 60)

    print("\n=== 4-4) 규모 추정 ===")
    scale = estimate_full_scale(pilot_df, coverage["pilot_n_games"])
    print(scale)

    print("\n=== 4-5) 실패 케이스 분석 ===")
    failures = analyze_failures(coverage["pilot_n_games"])
    print(failures)

    report = build_report(coverage, preview_schema, pbp_schema, wpa_domain, scale, failures)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n리포트 저장 완료: {REPORT_PATH} ({time.time()-t0:.1f}초)")


if __name__ == "__main__":
    main()
