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

    n_missing = df["reward_wpa_computed"].isna().sum()
    if n_missing > 0:
        raise ValueError(f"reward_wpa_computed 결측 {n_missing}건 — Phase 2-B 검토 필요")

    out_of_range = ((df["reward_wpa_computed"] < -1.0) | (df["reward_wpa_computed"] > 1.0)).sum()
    if out_of_range > 0:
        logger.warning("reward_wpa_computed [-1, 1] 범위 초과: %d건", out_of_range)

    if "reward_wpa" not in df.columns:
        raise ValueError("reward_wpa 컬럼이 없음 — 입력 데이터 검토 필요")

    df.to_parquet(output_path, index=False)
    logger.info("저장 완료: %s (%d 행)", output_path, len(df))

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    df_out = inject_computed_wpa()

    print("=== Phase 3 완료 ===")
    print(f"행 수: {len(df_out)}")
    print(f"reward_wpa (네이버 원본) 결측률: {df_out['reward_wpa'].isna().mean():.2%}")
    print(f"reward_wpa_computed 결측률: {df_out['reward_wpa_computed'].isna().mean():.2%}")
    print(f"\nreward_wpa_computed 통계:")
    print(df_out['reward_wpa_computed'].describe())
    print(f"\n[-1, 1] 범위 초과: {((df_out['reward_wpa_computed'] < -1) | (df_out['reward_wpa_computed'] > 1)).sum()}건")

    print(f"\n=== pa_result별 평균 reward_wpa_computed ===")
    print(
        df_out.groupby('pa_result')['reward_wpa_computed']
        .agg(['mean', 'count'])
        .sort_values('mean', ascending=False)
    )
