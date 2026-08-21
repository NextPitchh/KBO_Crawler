"""
data_analysis/methods/build_enriched_dataset.py

Phase 1의 최종 산출물: hsk_pa_with_wpa.parquet(11,984 PA, 원본 불변)에서
정규시즌 PA만 남긴 뒤(season_filter.py), 아래 세 소스를 PA 단위로 병합한다.

  1) pitcher_history.parquet  → 키 (pitcher_id, game_id, half)
                                 (이미 정규시즌 등판만으로 재계산된 상태)
  2) pa_bullpen_state.parquet → 키 _orig_idx (PA 1:1 직접 병합)
  3) pitcher_throws           → game_bullpen.parquet(throws) 우선,
                                 없으면 game_lineup.parquet(starter_hand),
                                 둘 다 없으면 NaN 유지(임의 채움 금지)

hsk_pa_with_wpa.parquet 자체는 절대 덮어쓰지 않는다 — 읽기 전용으로만 사용.
10등판 미만 투수의 PA는 여기서 제외하지 않는다(표본 부족은 데이터 품질
문제가 아니므로 버리지 않음 — shrinkage는 소비 측에서 처리).

출력  : data_analysis/results/hsk_pa_enriched.parquet
        data_analysis/results/enrichment_report.md

실행:
    uv run python -m data_analysis.methods.build_enriched_dataset
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .season_filter import (
    cross_validate_with_bullpen_size,
    filter_regular_season,
    is_regular_season,
)

logger = logging.getLogger(__name__)

PA_PATH              = "data_analysis/results/hsk_pa_with_wpa.parquet"
PITCHER_HISTORY_PATH = "data_analysis/results/pitcher_history.parquet"
BULLPEN_STATE_PATH   = "data_analysis/results/pa_bullpen_state.parquet"
GAME_BULLPEN_PATH    = "data_analysis/results/game_bullpen.parquet"
GAME_LINEUP_PATH     = "data_analysis/results/game_lineup.parquet"
LEAGUE_BASELINE_PATH    = "data_analysis/results/league_baseline.json"
LEAGUE_BASELINE_V1_PATH = "data_analysis/results/league_baseline_v1.json"

OUTPUT_PATH = "data_analysis/results/hsk_pa_enriched.parquet"
REPORT_PATH = "data_analysis/results/enrichment_report.md"

RATIO_TOLERANCE = 0.02  # ±2%p — bullpen_source / pitcher_throws 근사치 검증 허용오차


# ────────────────────────────────────────────────────────────────────────── #
#  Step 1: pitcher_history 병합
# ────────────────────────────────────────────────────────────────────────── #

def _merge_pitcher_history(df: pd.DataFrame, hist_path: str) -> pd.DataFrame:
    hist = pd.read_parquet(hist_path)
    key = ["pitcher_id", "game_id", "half"]

    dup = hist.duplicated(subset=key).sum()
    if dup:
        raise ValueError(f"pitcher_history.parquet의 {key} 조합에 중복 {dup}건 — 1:1 병합 불가")

    # PA 원본 컬럼과 이름이 겹치지 않는 신규 컬럼만 병합한다.
    new_cols = [c for c in hist.columns if c not in df.columns or c in key]
    merged = df.merge(hist[new_cols], on=key, how="left")

    n_unmatched = merged["prior_n_apps"].isna().sum()
    logger.info(
        "pitcher_history 병합 완료 | 신규 컬럼 %d개 / 조인 실패(등판 이력 없음) %d행",
        len(new_cols) - len(key), n_unmatched,
    )
    return merged


# ────────────────────────────────────────────────────────────────────────── #
#  Step 2: pa_bullpen_state 병합 (PA 1:1, _orig_idx 기준)
# ────────────────────────────────────────────────────────────────────────── #

def _merge_bullpen_state(df: pd.DataFrame, state_path: str) -> pd.DataFrame:
    state = pd.read_parquet(state_path)

    if state["_orig_idx"].duplicated().any():
        raise ValueError("pa_bullpen_state.parquet의 _orig_idx가 유일하지 않음 — PA 1:1 병합 불가")

    keep_cols = [
        "_orig_idx", "n_pitchers_used", "current_pitcher_pa_in_app", "is_pitcher_change",
        "bullpen_listed", "bullpen_used", "bullpen_available",
        "bullpen_available_ratio", "bullpen_source",
    ]
    merged = df.merge(state[keep_cols], on="_orig_idx", how="left")

    n_unmatched = merged["bullpen_source"].isna().sum()
    if n_unmatched:
        raise ValueError(f"pa_bullpen_state 병합 실패 {n_unmatched}행 — _orig_idx 정합성 확인 필요")

    logger.info("pa_bullpen_state 병합 완료 | %d행 전부 매칭", len(merged))
    return merged


# ────────────────────────────────────────────────────────────────────────── #
#  Step 3: pitcher_throws — 불펜명단 우선, 선발 폴백, 그 외 NaN
# ────────────────────────────────────────────────────────────────────────── #

def _build_throws_lookup(game_bullpen_path: str, game_lineup_path: str) -> tuple[dict, dict]:
    gb = pd.read_parquet(game_bullpen_path)
    gl = pd.read_parquet(game_lineup_path)

    # 빈 문자열("")은 hitType 파싱 실패를 의미 — NaN과 동일하게 취급(임의 채움 방지)
    bullpen_map = {
        (pid, gid): (hand if hand else np.nan)
        for pid, gid, hand in zip(gb["pitcher_id"], gb["game_id"], gb["throws"])
    }
    starter_map = {
        (pid, gid): (hand if hand else np.nan)
        for pid, gid, hand in zip(gl["starter_id"], gl["game_id"], gl["starter_hand"])
    }
    return bullpen_map, starter_map


def _add_pitcher_throws(df: pd.DataFrame, game_bullpen_path: str, game_lineup_path: str) -> pd.DataFrame:
    bullpen_map, starter_map = _build_throws_lookup(game_bullpen_path, game_lineup_path)

    def _lookup(pid, gid):
        if (pid, gid) in bullpen_map:
            return bullpen_map[(pid, gid)]
        if (pid, gid) in starter_map:
            return starter_map[(pid, gid)]
        return np.nan

    df["pitcher_throws"] = df.apply(lambda r: _lookup(r["pitcher_id"], r["game_id"]), axis=1)

    n_missing = df["pitcher_throws"].isna().sum()
    logger.info(
        "pitcher_throws 부착 완료 | 결측 %d행 (%.1f%%)", n_missing, 100 * n_missing / len(df)
    )
    return df


# ────────────────────────────────────────────────────────────────────────── #
#  정규시즌 필터 관련 보조 통계 (리포트용)
# ────────────────────────────────────────────────────────────────────────── #

def _expected_preview_share(
    full_original: pd.DataFrame, regular_game_ids: set, game_lineup_path: str
) -> float:
    """
    pa_bullpen_state.parquet의 자체 분류(bullpen_source)에 기대지 않고,
    game_lineup.parquet(어떤 경기가 preview 커버리지가 있는지)만으로
    독립적으로 계산한 "기대" preview 비중 — 순환 검증을 피하기 위함.
    """
    gl = pd.read_parquet(game_lineup_path)
    covered_games = set(gl["game_id"].unique())
    reg = full_original[full_original["game_id"].isin(regular_game_ids)]
    if len(reg) == 0:
        return float("nan")
    return float(reg["game_id"].isin(covered_games).mean())


def _preseason_breakdown(full_original: pd.DataFrame) -> pd.DataFrame:
    games = full_original[["game_id"]].drop_duplicates().copy()
    games["is_regular"] = games["game_id"].apply(is_regular_season)
    games["year"] = games["game_id"].str[:4].astype(int)

    preseason_ids = set(games.loc[~games["is_regular"], "game_id"])
    pa = full_original.copy()
    pa["year"] = pa["game_id"].str[:4].astype(int)
    pa["is_preseason"] = pa["game_id"].isin(preseason_ids)

    by_year = (
        pa.groupby("year")
        .agg(
            n_games_total=("game_id", "nunique"),
            n_pa_total=("game_id", "size"),
        )
    )
    preseason_by_year = (
        pa[pa["is_preseason"]]
        .groupby("year")
        .agg(
            n_games_preseason=("game_id", "nunique"),
            n_pa_preseason=("game_id", "size"),
        )
    )
    out = by_year.join(preseason_by_year, how="left").fillna(0)
    for c in ["n_games_preseason", "n_pa_preseason"]:
        out[c] = out[c].astype(int)
    out["n_games_regular"] = out["n_games_total"] - out["n_games_preseason"]
    return out.reset_index()


# ────────────────────────────────────────────────────────────────────────── #
#  검증
# ────────────────────────────────────────────────────────────────────────── #

def validate_enriched(
    full_original: pd.DataFrame,
    original: pd.DataFrame,
    enriched: pd.DataFrame,
    game_lineup_path: str = GAME_LINEUP_PATH,
) -> list[dict]:
    """7항목 검증. 각 항목 dict(name, passed, detail) 리스트 반환."""
    results: list[dict] = []

    # 1) 행 수 = 정규시즌 필터 적용 후 PA 수와 일치
    passed1 = len(enriched) == len(original)
    results.append({
        "name": "행 수 = 정규시즌 필터 후 PA 수",
        "passed": passed1,
        "detail": f"{len(full_original):,} → {len(enriched):,}행 (시범경기 {len(full_original) - len(original):,}행 제외)",
    })

    # 2) 기존 29개 컬럼 값 전수 일치 (남은 행에 대해)
    orig_cols = full_original.columns.tolist()
    try:
        pd.testing.assert_frame_equal(
            original[orig_cols].reset_index(drop=True),
            enriched[orig_cols].reset_index(drop=True),
            check_dtype=True,
        )
        passed2 = True
        detail2 = f"남은 {len(original):,}행에 대해 원본 {len(orig_cols)}개 컬럼 전수 일치"
    except AssertionError as exc:
        passed2 = False
        detail2 = f"불일치 발견: {str(exc)[:300]}"
    results.append({"name": "기존 29개 컬럼 값 전수 일치", "passed": passed2, "detail": detail2})

    # 3) prior_n_apps==0 → prior_wpa_std NaN
    first_app = enriched["prior_n_apps"] == 0
    n_violation = int((first_app & enriched["prior_wpa_std"].notna()).sum())
    results.append({
        "name": "prior_n_apps==0 행의 prior_wpa_std가 NaN",
        "passed": n_violation == 0,
        "detail": f"위반 {n_violation}행 (첫 등판 PA {int(first_app.sum())}행 중)",
    })

    # 4) n_pitchers_used 단조 비감소 (game_id, half) 내
    ordered = enriched.sort_values("_orig_idx")
    non_monotonic = (
        ordered.groupby(["game_id", "half"])["n_pitchers_used"]
        .apply(lambda s: (s.diff().dropna() < 0).any())
    )
    n_bad_groups = int(non_monotonic.sum())
    results.append({
        "name": "n_pitchers_used가 (game_id, half) 내 단조 증가",
        "passed": n_bad_groups == 0,
        "detail": f"위반 그룹 {n_bad_groups}개 / 전체 {non_monotonic.shape[0]}개",
    })

    # 5) 비율 컬럼 [0,1] 또는 NaN
    ratio_cols = ["prior_bb_rate", "prior_so_rate", "prior_hr_rate", "bullpen_available_ratio"]
    bad_cols = []
    for c in ratio_cols:
        s = enriched[c].dropna()
        if not s.between(0, 1).all():
            bad_cols.append(c)
    results.append({
        "name": "비율 컬럼이 [0,1] 또는 NaN",
        "passed": len(bad_cols) == 0,
        "detail": "모두 [0,1] 범위 내" if not bad_cols else f"범위 위반 컬럼: {bad_cols}",
    })

    # 6) bullpen_source 분포 근사치 — game_lineup.parquet 기반 독립 기대치와 비교
    regular_game_ids = set(original["game_id"].unique())
    expected_preview_share = _expected_preview_share(full_original, regular_game_ids, game_lineup_path)
    share = enriched["bullpen_source"].value_counts(normalize=True).get("preview", 0.0)
    passed6 = abs(share - expected_preview_share) <= RATIO_TOLERANCE
    results.append({
        "name": "bullpen_source 분포(preview 비중이 preview 커버리지 게임 비중과 근접)",
        "passed": passed6,
        "detail": (
            f"실측 preview={share:.1%} (독립 기대치 {expected_preview_share:.1%}, "
            f"허용오차 ±{RATIO_TOLERANCE:.0%})"
        ),
    })

    # 7) pitcher_throws 결측률 ≈ (preview 없는 게임 비중)
    expected_missing = 1 - expected_preview_share
    missing_rate = enriched["pitcher_throws"].isna().mean()
    passed7 = abs(missing_rate - expected_missing) <= RATIO_TOLERANCE
    results.append({
        "name": "pitcher_throws 결측률이 preview 미커버리지 비중과 근접",
        "passed": passed7,
        "detail": (
            f"실측 {missing_rate:.1%} (독립 기대치 {expected_missing:.1%}, "
            f"허용오차 ±{RATIO_TOLERANCE:.0%})"
        ),
    })

    return results


# ────────────────────────────────────────────────────────────────────────── #
#  리포트
# ────────────────────────────────────────────────────────────────────────── #

def _load_json(path: str) -> dict | None:
    import json
    import os

    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_report(
    full_original: pd.DataFrame,
    enriched: pd.DataFrame,
    results: list[dict],
    cross_val: pd.DataFrame,
    report_path: str,
) -> None:
    lines: list[str] = []
    lines.append("# Enrichment Report — hsk_pa_enriched.parquet\n")
    lines.append(f"- 행 수: {len(enriched):,}")
    lines.append(f"- 열 수: {len(enriched.columns):,}\n")

    lines.append("## 검증 결과\n")
    lines.append("| # | 항목 | 결과 | 상세 |")
    lines.append("|---|------|------|------|")
    for i, r in enumerate(results, 1):
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"| {i} | {r['name']} | {mark} | {r['detail']} |")
    lines.append("")

    # ── 시범경기 제외 내역 ────────────────────────────────────────────────
    lines.append("## 시범경기 제외 내역\n")
    breakdown = _preseason_breakdown(full_original)
    lines.append("| 연도 | 전체 경기 | 정규시즌 경기 | 시범경기 | 시범경기 PA |")
    lines.append("|---|---|---|---|---|")
    for _, row in breakdown.iterrows():
        lines.append(
            f"| {int(row['year'])} | {int(row['n_games_total'])} | "
            f"{int(row['n_games_regular'])} | {int(row['n_games_preseason'])} | "
            f"{int(row['n_pa_preseason'])} |"
        )
    total_preseason_games = int(breakdown["n_games_preseason"].sum())
    total_preseason_pa = int(breakdown["n_pa_preseason"].sum())
    lines.append(
        f"\n**합계**: 시범경기 {total_preseason_games}경기, "
        f"{total_preseason_pa}PA ({100*total_preseason_pa/len(full_original):.1f}%) 제외.\n"
    )
    lines.append(
        "참고: 배경 이슈에서 언급된 \"3월 경기 14건/1,040 PA/8.7%\"는 달력상 3월 경기 전체를 "
        "가리키며, 실제 개막일 기준(+불펜 20명 초과 기준과 100% 교차 일치) 시범경기는 "
        f"{total_preseason_games}건/{total_preseason_pa}PA 이다. "
        "2018-03-30/31, 2024-03-26/27/28은 3월이지만 그 해 개막일 이후 정규시즌 경기이며 "
        "불펜 명단도 11~12명으로 정상 범위여서 정규시즌으로 유지했다.\n"
    )

    # ── 개막일 vs 불펜 인원 교차 검증 ────────────────────────────────────
    lines.append("## 개막일 기준 vs 불펜 인원 기준 교차 검증\n")
    n_mismatch = int(cross_val["mismatch"].sum()) if len(cross_val) else 0
    lines.append(
        f"- 비교 가능 경기(불펜 명단 preview 있는 경기): {len(cross_val)}개\n"
        f"- 불일치: {n_mismatch}건\n"
    )
    if n_mismatch:
        lines.append("| game_id | game_date | opening_date | max_bullpen_size | by_date | by_bullpen |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in cross_val[cross_val["mismatch"]].iterrows():
            lines.append(
                f"| {row['game_id']} | {row['game_date']} | {row['opening_date']} | "
                f"{row['max_bullpen_size']} | {row['is_regular_by_date']} | {row['is_regular_by_bullpen']} |"
            )
        lines.append("")
    else:
        lines.append("불일치 없음 — 두 방법이 완전히 일치한다.\n")

    # ── league_baseline 변경 전후 비교 ───────────────────────────────────
    lines.append("## league_baseline 변경 전후 비교\n")
    v1 = _load_json(LEAGUE_BASELINE_V1_PATH)
    v2 = _load_json(LEAGUE_BASELINE_PATH)
    if v1 and v2:
        lines.append("| 지표 | v1(수정 전) | v2(수정 후) |")
        lines.append("|---|---|---|")
        keys = [
            "league_wpa_std", "league_wpa_std_sd",
            "league_bb_rate", "league_bb_rate_sd",
            "league_so_rate", "league_so_rate_sd",
            "league_hr_rate", "league_hr_rate_sd",
        ]
        for k in keys:
            v1_val = v1.get(k)
            v2_val = v2.get(k)
            v1_str = f"{v1_val:.4f}" if isinstance(v1_val, (int, float)) else str(v1_val)
            v2_str = f"{v2_val:.4f}" if isinstance(v2_val, (int, float)) else str(v2_val)
            lines.append(f"| {k} | {v1_str} | {v2_str} |")
        lines.append(f"| n_established (appearances→pitchers) | {v1.get('n_established_appearances')} | {v2.get('n_established_pitchers')} |")
        lines.append("")
        lines.append(
            "v1은 등판 단위로 sd를 계산해 bb_rate_sd가 평균의 1.65배, hr_rate_sd가 평균의 "
            "2.7배로 나왔다(표본 적은 투수의 극단값 오염). v2는 정규시즌·prior_n_apps>=10 "
            "필터 후 투수 단위(각 투수 최종 시점 1개 값)로 sd를 계산해 세 비율 지표 모두 "
            "sd < 평균으로 정상화되었다.\n"
        )
    else:
        lines.append("v1 또는 v2 baseline 파일을 찾을 수 없어 비교를 생략합니다.\n")

    # ── prior_n_apps 분포 ─────────────────────────────────────────────────
    lines.append("## prior_n_apps 분포 (표본 부족 현황)\n")
    lines.append("```")
    lines.append(enriched["prior_n_apps"].describe().to_string())
    lines.append("```\n")

    # ── preview 커버리지 ──────────────────────────────────────────────────
    lines.append("## preview 커버리지 실측치\n")
    src_counts = enriched["bullpen_source"].value_counts()
    src_share = enriched["bullpen_source"].value_counts(normalize=True)
    lines.append("| source | PA 수 | 비중 |")
    lines.append("|---|---|---|")
    for k in src_counts.index:
        lines.append(f"| {k} | {src_counts[k]:,} | {src_share[k]:.1%} |")
    lines.append("")

    # ── 결측률 ────────────────────────────────────────────────────────────
    lines.append("## 컬럼별 결측률 (상위 20개)\n")
    na_rate = enriched.isna().mean().sort_values(ascending=False)
    na_rate = na_rate[na_rate > 0].head(20)
    lines.append("| 컬럼 | 결측률 |")
    lines.append("|---|---|")
    for col, rate in na_rate.items():
        lines.append(f"| {col} | {rate:.1%} |")
    lines.append("")

    lines.append("## pitcher_throws 분포 (L/R/U)\n")
    lines.append("```")
    lines.append(enriched["pitcher_throws"].value_counts(dropna=False).to_string())
    lines.append("```\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("리포트 저장 완료: %s", report_path)


# ────────────────────────────────────────────────────────────────────────── #
#  메인
# ────────────────────────────────────────────────────────────────────────── #

def build_enriched_dataset(
    pa_path: str = PA_PATH,
    hist_path: str = PITCHER_HISTORY_PATH,
    bullpen_state_path: str = BULLPEN_STATE_PATH,
    game_bullpen_path: str = GAME_BULLPEN_PATH,
    game_lineup_path: str = GAME_LINEUP_PATH,
    output_path: str = OUTPUT_PATH,
    report_path: str = REPORT_PATH,
) -> pd.DataFrame:
    full_original = pd.read_parquet(pa_path)  # 읽기 전용 — 이 함수는 절대 이 경로에 쓰지 않는다
    logger.info("로드: %d행, %d컬럼", *full_original.shape)

    original = filter_regular_season(full_original)  # 시범경기 PA 제외

    df = original.copy()
    df = _merge_pitcher_history(df, hist_path)
    df = _merge_bullpen_state(df, bullpen_state_path)
    df = _add_pitcher_throws(df, game_bullpen_path, game_lineup_path)

    cross_val = cross_validate_with_bullpen_size()

    results = validate_enriched(full_original, original, df, game_lineup_path)
    n_fail = sum(1 for r in results if not r["passed"])
    if n_fail:
        failed_names = [r["name"] for r in results if not r["passed"]]
        raise AssertionError(f"검증 {n_fail}건 실패 — 저장 중단: {failed_names}")

    df.to_parquet(output_path, index=False)
    logger.info("저장 완료: %s (%d행, %d컬럼)", output_path, *df.shape)

    _write_report(full_original, df, results, cross_val, report_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out = build_enriched_dataset()

    print("\n" + "=" * 60)
    print(f"[hsk_pa_enriched] {len(out):,}행 x {len(out.columns)}열")
    print(f"\n[bullpen_source]\n{out['bullpen_source'].value_counts().to_string()}")
    print(f"\n[pitcher_throws 결측률] {out['pitcher_throws'].isna().mean():.1%}")
    print(f"\n리포트: {REPORT_PATH}")
    print("=" * 60)
