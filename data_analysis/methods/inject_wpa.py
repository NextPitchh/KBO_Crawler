"""
data_analysis/methods/inject_wpa.py

Phase 2 산출물(hsk_pa_with_states.parquet)에 reward_wpa_computed 컬럼을 추가하고
hsk_pa_with_wpa.parquet으로 저장.

reward_wpa_computed = we_after - we_before  (공격팀 관점 ΔWE)

실행: python -m data_analysis.methods.inject_wpa
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def inject_computed_wpa(
    input_path: str = "data_analysis/results/hsk_pa_with_states.parquet",
    output_path: str = "data_analysis/results/hsk_pa_with_wpa.parquet",
) -> pd.DataFrame:
    """
    hsk_pa_with_states.parquet을 읽어 reward_wpa_computed를 추가한 뒤 저장.

    Parameters
    ----------
    input_path  : Phase 2 산출 parquet (we_before / we_after 포함)
    output_path : 최종 parquet 저장 경로

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_parquet(input_path)
    logger.info("로드: %d 행, %d 컬럼", *df.shape)

    required = {"we_before", "we_after"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 parquet에 필요한 컬럼이 없습니다: {missing}")

    df["reward_wpa_computed"] = df["we_after"] - df["we_before"]

    df.to_parquet(output_path, index=False)
    logger.info(
        "저장 완료: %s  (reward_wpa_computed 결측=%d)",
        output_path,
        int(df["reward_wpa_computed"].isna().sum()),
    )

    return df


if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    df_out = inject_computed_wpa()

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  inject_wpa sanity check")
    print(SEP)

    # 1. 결측 없음
    n_nan = int(df_out["reward_wpa_computed"].isna().sum())
    assert n_nan == 0, f"reward_wpa_computed 결측 {n_nan}건"
    print(f"[OK] reward_wpa_computed 결측 0건")

    # 2. 범위 [-1, +1]
    out_of_range = ~df_out["reward_wpa_computed"].between(-1.0, 1.0)
    n_oor = int(out_of_range.sum())
    assert n_oor == 0, (
        f"reward_wpa_computed 범위 위반 {n_oor}건:\n"
        f"{df_out.loc[out_of_range, 'reward_wpa_computed'].describe()}"
    )
    print(f"[OK] reward_wpa_computed 모두 [-1, +1] 이내")

    # 3. 평균 근사 0
    mean_val = df_out["reward_wpa_computed"].mean()
    print(f"[INFO] reward_wpa_computed 평균: {mean_val:.6f}  (기대: ~0)")

    # 4. pa_result별 평균 ΔWE 부호 순서 확인
    by_result = (
        df_out.groupby("pa_result")["reward_wpa_computed"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n[pa_result별 평균 reward_wpa_computed]")
    print(by_result.to_string())

    for good, bad in [("HR", "OUT"), ("HR", "SO"), ("1B", "OUT")]:
        if good in by_result.index and bad in by_result.index:
            assert by_result[good] > by_result[bad], (
                f"부호 순서 위반: {good}({by_result[good]:.4f}) <= "
                f"{bad}({by_result[bad]:.4f})"
            )
    print("\n[OK] 타석 결과별 부호 순서 (HR > 1B > OUT) 확인")

    print(f"\n모든 sanity check 통과!")
    print(SEP)
