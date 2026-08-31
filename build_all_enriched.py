"""
build_all_enriched.py

10년 규모 최종 통합: all_pa_with_wpa + pitcher_history(prior_*) +
bullpen_state + pitcher_throws.

build_enriched_dataset.py의 병합 헬퍼(_merge_pitcher_history,
_merge_bullpen_state, _add_pitcher_throws)는 이미 경로 파라미터화되어
있으므로 그대로 재사용한다(수정 없음). 다만 그 모듈의 top-level
build_enriched_dataset()은 153경기 전용 하드코딩 경로(cross_validate_with_
bullpen_size(), league_baseline 비교 리포트)를 갖고 있어 10년 데이터에는
맞지 않으므로, 이 스크립트는 병합 로직만 재사용하고 검증/리포트는 새로
작성한다.

산출: data_analysis/results/all_pa_enriched.parquet

실행:
    uv run python build_all_enriched.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_analysis.methods.build_enriched_dataset import (
    _merge_pitcher_history, _merge_bullpen_state, _add_pitcher_throws,
)
from data_analysis.methods.pitcher_history import verify_no_leakage
from data_analysis.methods.run_full_year import evaluate_wpa_order

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PA_PATH = "data_analysis/results/all_pa_with_wpa.parquet"
HIST_PATH = "data_analysis/results/all_pitcher_history.parquet"
BULLPEN_STATE_PATH = "data_analysis/results/all_pa_bullpen_state.parquet"
GAME_BULLPEN_PATH = "data_analysis/results/all_game_bullpen.parquet"
GAME_LINEUP_PATH = "data_analysis/results/all_game_lineup.parquet"
OUTPUT_PATH = "data_analysis/results/all_pa_enriched.parquet"


def telescoping_check(df: pd.DataFrame) -> dict:
    """홈팀 관점 통일 공식으로 경기별 telescoping 오차를 계산."""
    winner_map: dict[str, float] = {}
    final_scores = pd.read_csv("data/game_ids/final_scores.csv", dtype={"game_id": str})
    for _, row in final_scores.iterrows():
        w = 1.0 if row["home_score"] > row["away_score"] else (0.0 if row["home_score"] < row["away_score"] else 0.5)
        winner_map[row["game_id"]] = w

    import glob
    extras_games: set[str] = set()
    for year in range(2016, 2026):
        for f in glob.glob(f"data/pbp_full/{year}/*.csv"):
            pass  # 아래에서 별도 로딩

    # 연장 게임 판별: raw pbp에서 inning>=10 존재 여부(이미 필터링된 df엔 없음)
    all_raw_gids: set[str] = set()
    for year in range(2016, 2026):
        files = glob.glob(f"data/pbp_full/{year}/*.csv")
        if not files:
            continue
        raw = pd.concat([pd.read_csv(f, usecols=["game_id", "inning"], low_memory=False) for f in files], ignore_index=True)
        extras_games |= set(raw.loc[raw["inning"] >= 10, "game_id"].unique())

    df = df.sort_values("_orig_idx")
    is_bot = df["half"] == "bot"
    we_before_home = np.where(is_bot, df["we_before"], 1 - df["we_before"])
    we_after_home = np.where(is_bot, df["we_after"], 1 - df["we_after"])
    delta_home = we_after_home - we_before_home
    tmp = df[["game_id"]].copy()
    tmp["delta_home"] = delta_home
    tmp["we_before_home"] = we_before_home

    errors = []
    for gid, g in tmp.groupby("game_id"):
        initial = g["we_before_home"].iloc[0]
        final_we = 0.5 if gid in extras_games else winner_map.get(gid)
        if final_we is None:
            continue
        theoretical = final_we - initial
        actual = g["delta_home"].sum()
        errors.append(actual - theoretical)

    errors = np.array(errors)
    return {
        "n_games": len(errors),
        "mean_abs_error": float(np.abs(errors).mean()),
        "n_error_ge_0_1": int((np.abs(errors) >= 0.1).sum()),
    }


def main() -> None:
    full_original = pd.read_parquet(PA_PATH)
    logger.info("로드: %d행, %d컬럼", *full_original.shape)
    orig_cols = list(full_original.columns)

    df = full_original.copy()
    df = _merge_pitcher_history(df, HIST_PATH)
    df = _merge_bullpen_state(df, BULLPEN_STATE_PATH)
    df = _add_pitcher_throws(df, GAME_BULLPEN_PATH, GAME_LINEUP_PATH)

    results: list[dict] = []

    # 1) 행 수
    results.append({"name": "행 수 == Task2 결과와 동일", "passed": len(df) == len(full_original),
                     "detail": f"{len(full_original):,} -> {len(df):,}"})

    # 2) 기존 31컬럼 전수 일치
    try:
        pd.testing.assert_frame_equal(
            full_original[orig_cols].reset_index(drop=True),
            df[orig_cols].reset_index(drop=True),
            check_dtype=True,
        )
        passed2, detail2 = True, f"{len(orig_cols)}개 컬럼 전수 일치"
    except AssertionError as exc:
        passed2, detail2 = False, str(exc)[:300]
    results.append({"name": "기존 31컬럼 값 전수 일치", "passed": passed2, "detail": detail2})

    # 3) leakage 검증
    hist_df = pd.read_parquet(HIST_PATH)
    try:
        verify_no_leakage(hist_df, n_pitchers=30)
        results.append({"name": "verify_no_leakage(30명)", "passed": True, "detail": "통과"})
    except AssertionError as exc:
        results.append({"name": "verify_no_leakage(30명)", "passed": False, "detail": str(exc)[:300]})

    # 4) prior_n_apps==0 -> prior_wpa_std NaN
    first_app = df["prior_n_apps"] == 0
    n_violation = int((first_app & df["prior_wpa_std"].notna()).sum())
    results.append({"name": "prior_n_apps==0 -> prior_wpa_std NaN", "passed": n_violation == 0,
                     "detail": f"위반 {n_violation}건"})

    # 5) n_pitchers_used 단조 증가
    ordered = df.sort_values("_orig_idx")
    non_monotonic = ordered.groupby(["game_id", "half"])["n_pitchers_used"].apply(lambda s: (s.diff().dropna() < 0).any())
    n_bad = int(non_monotonic.sum())
    results.append({"name": "n_pitchers_used 단조 증가", "passed": n_bad == 0, "detail": f"위반 그룹 {n_bad}개"})

    # 6) 비율 컬럼 [0,1] 또는 NaN
    ratio_cols = ["prior_bb_rate", "prior_so_rate", "prior_hr_rate", "bullpen_available_ratio"]
    bad_cols = [c for c in ratio_cols if not df[c].dropna().between(0, 1).all()]
    results.append({"name": "비율 컬럼 [0,1] 또는 NaN", "passed": len(bad_cols) == 0,
                     "detail": "정상" if not bad_cols else f"위반: {bad_cols}"})

    # 7) WPA 도메인 순서
    order = evaluate_wpa_order(df)
    results.append({"name": "pa_result 도메인 순서(Tier 1)", "passed": order["tier1_pass"],
                     "detail": f"warnings={order['warnings']}" if order["tier1_pass"] else str(order["tier1_violations"])})

    # 8) telescoping
    tel = telescoping_check(df)
    tel_pass = tel["n_error_ge_0_1"] == 0
    results.append({"name": "Telescoping 검증", "passed": tel_pass,
                     "detail": f"게임 {tel['n_games']}개, 평균절대오차 {tel['mean_abs_error']:.6f}, "
                               f"오차>=0.1인 게임 {tel['n_error_ge_0_1']}건"})

    print("\n" + "=" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{'PASS' if r['passed'] else 'FAIL'}] {r['name']} — {r['detail']}")
    print("=" * 70)

    n_fail = sum(1 for r in results if not r["passed"])
    if n_fail:
        raise AssertionError(f"검증 {n_fail}건 실패 — 저장 중단")

    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("저장 완료: %s (%d행, %d컬럼)", OUTPUT_PATH, *df.shape)

    import json
    with open("data_analysis/results/all_enriched_validation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
