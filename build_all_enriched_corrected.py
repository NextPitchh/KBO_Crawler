"""
build_all_enriched_corrected.py

Task 4-1: all_pa_enriched.parquet 의 we_before / we_after 를 아웃 차원 보정
(Option B, get_we_with_out_correction) 버전으로 재계산한다.

- 원본 all_pa_enriched.parquet 은 절대 수정하지 않는다.
- 새 파일 all_pa_enriched_corrected.parquet 로 저장한다.
- 기존 68컬럼은 값 그대로 유지하고 아래 3개 컬럼만 추가한다:
    we_before_corrected, we_after_corrected, reward_wpa_computed_corrected

재계산 로직은 state_transition.py 의 Step 3 / Step 7 (_compute_we_after) 을
그대로 옮기되 WE 조회 함수만 get_we_with_boundary → get_we_with_out_correction
으로 교체한다. 단, **각 게임의 마지막 PA(_orig_idx 최대)** 는 terminal PA
보정으로 이미 schedule API 최종 스코어(1.0/0.0/0.5)로 확정돼 있으므로
we_after_corrected 를 원본 we_after 로 그대로 둔다 (ground truth 보존).

실행:
    uv run python build_all_enriched_corrected.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_analysis.methods.we_re_lookup import get_we_with_out_correction

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

SRC = "data_analysis/results/all_pa_enriched.parquet"
DST = "data_analysis/results/all_pa_enriched_corrected.parquet"

GROUP_KEYS = ["game_id", "inning", "home_or_away"]


def _we_after_corrected_row(row: pd.Series) -> float:
    """state_transition._compute_we_after 의 Option B 버전."""
    sd_after = int(row["score_diff_attacker"]) + int(row["runs_scored"])
    inning = int(row["inning"])
    half = str(row["half"])

    if not row["inning_ended"]:
        return get_we_with_out_correction(
            inning, half, sd_after,
            int(row["_next_out_count"]), str(row["_next_base_state"]),
        )

    # 이닝 종료 처리 (원본 로직과 동일 분기)
    if half == "top":
        next_inning, next_half = inning, "bot"
    else:
        next_inning, next_half = inning + 1, "top"

    if inning == 9 and half == "bot" and sd_after > 0:
        return 1.0
    if next_inning > 12:
        if sd_after > 0:
            return 1.0
        if sd_after < 0:
            return 0.0
        return 0.5

    opponent_we = get_we_with_out_correction(
        next_inning, next_half, -sd_after, 0, "0"
    )
    return 1.0 - opponent_we


def main() -> None:
    df = pd.read_parquet(SRC)
    logger.info("로드: %d행 %d컬럼", *df.shape)
    orig_cols = list(df.columns)

    df = df.sort_values("_orig_idx", kind="stable").reset_index(drop=True)

    # ── we_before_corrected : state_transition Step 3 와 동일, 함수만 교체 ──
    df["we_before_corrected"] = [
        get_we_with_out_correction(int(i), str(h), int(s), int(o), str(b))
        for i, h, s, o, b in zip(
            df["inning"], df["half"], df["score_diff_attacker"],
            df["out_count"], df["base_state"],
        )
    ]

    # ── we_after_corrected : state_transition Step 7 와 동일, 함수만 교체 ──
    df["_next_out_count"] = df.groupby(GROUP_KEYS)["out_count"].shift(-1)
    df["_next_base_state"] = df.groupby(GROUP_KEYS)["base_state"].shift(-1)
    df["we_after_corrected"] = df.apply(_we_after_corrected_row, axis=1)

    # ── 게임 마지막 PA : terminal PA 보정(ground truth) 보존 ────────────────
    last_idx = df.groupby("game_id")["_orig_idx"].idxmax()
    n_pinned = len(last_idx)
    df.loc[last_idx, "we_after_corrected"] = df.loc[last_idx, "we_after"].to_numpy()
    logger.info("게임 마지막 PA %d건 we_after 원본 유지 (terminal 보정 보존)", n_pinned)

    df["we_before_corrected"] = df["we_before_corrected"].clip(0.0, 1.0).round(6)
    df["we_after_corrected"] = df["we_after_corrected"].clip(0.0, 1.0).round(6)
    df["reward_wpa_computed_corrected"] = (
        df["we_after_corrected"] - df["we_before_corrected"]
    ).round(6)

    df = df.drop(columns=["_next_out_count", "_next_base_state"])

    # ── 무결성 체크 ────────────────────────────────────────────────────────
    assert list(df.columns)[: len(orig_cols)] == orig_cols, "기존 컬럼 순서/구성 변경됨"
    pd.testing.assert_frame_equal(
        df[orig_cols].reset_index(drop=True),
        pd.read_parquet(SRC).sort_values("_orig_idx", kind="stable").reset_index(drop=True)[orig_cols],
        check_dtype=True,
    )
    for c in ("we_before_corrected", "we_after_corrected"):
        assert df[c].between(0.0, 1.0).all(), f"{c} 범위 위반"
        assert df[c].notna().all(), f"{c} 결측"
    assert df["reward_wpa_computed_corrected"].between(-1.0, 1.0).all(), "ΔWE 범위 위반"

    # out_count==1 행은 we_before 가 원본과 동일해야 함 (보정항 0)
    m1 = df["out_count"] == 1
    max_dev = (df.loc[m1, "we_before_corrected"] - df.loc[m1, "we_before"]).abs().max()
    assert max_dev < 1e-6, f"1사 we_before 편차 {max_dev} (0 이어야 함)"
    logger.info("무결성 체크 통과 (1사 we_before 최대편차 %.2e)", max_dev)

    df.to_parquet(DST, index=False)
    logger.info("저장: %s (%d행 %d컬럼)", DST, *df.shape)

    # ── 요약 ──────────────────────────────────────────────────────────────
    d_before = (df["we_before_corrected"] - df["we_before"]).abs()
    d_after = (df["we_after_corrected"] - df["we_after"]).abs()
    print("\n[변경 요약]")
    print(f"  we_before 변경 행: {(d_before > 1e-6).sum():,} / {len(df):,} "
          f"(평균 |Δ| {d_before[d_before>1e-6].mean():.4f}, 최대 {d_before.max():.4f})")
    print(f"  we_after  변경 행: {(d_after > 1e-6).sum():,} / {len(df):,} "
          f"(평균 |Δ| {d_after[d_after>1e-6].mean():.4f}, 최대 {d_after.max():.4f})")
    print(f"  reward_wpa_computed          평균 {df['reward_wpa_computed'].mean():+.5f}  std {df['reward_wpa_computed'].std():.5f}")
    print(f"  reward_wpa_computed_corrected 평균 {df['reward_wpa_computed_corrected'].mean():+.5f}  std {df['reward_wpa_computed_corrected'].std():.5f}")


if __name__ == "__main__":
    main()
