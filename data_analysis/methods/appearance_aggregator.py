"""
data_analysis/methods/appearance_aggregator.py

PA-level(hsk_pa_with_wpa.parquet) → 등판(appearance) 단위 집계.
등판 = (pitcher_id, game_id, half) 조합.

정규시즌 필터(season_filter.py)를 집계 이전에 적용한다 — 시범경기 등판이
섞이면 pitcher_history.py의 prior_* 누적 통계가 왜곡되므로, 필터는 반드시
이 단계(집계 입력)에서 걸러야 한다. 10등판 미만 투수는 여기서 제외하지
않는다(표본 부족은 별도 shrinkage로 처리 — 데이터 자체를 버리지 않음).

입력  : data_analysis/results/hsk_pa_with_wpa.parquet (11,984 PA, 정규시즌 필터 적용 전)
출력  : data_analysis/results/pitcher_appearances.parquet (정규시즌 PA만)

실행:
    uv run python -m data_analysis.methods.appearance_aggregator
"""

from __future__ import annotations

import logging

import pandas as pd

from .season_filter import filter_regular_season

logger = logging.getLogger(__name__)

INPUT_PATH  = "data_analysis/results/hsk_pa_with_wpa.parquet"
OUTPUT_PATH = "data_analysis/results/pitcher_appearances.parquet"

_HALF_ORDER = {"top": 0, "bot": 1}  # 초(top)가 말(bot)보다 시간상 앞선다

# pa_result 레이블(pa_aggregator.PA_RESULT_PATTERNS 참고) → 출력 컬럼명
_OUTCOME_COL_TO_LABEL = {
    "n_bb": "BB", "n_so": "SO", "n_hr": "HR", "n_out": "OUT",
    "n_1b": "1B", "n_2b": "2B", "n_3b": "3B", "n_gdp": "GDP", "n_sf": "SF",
}

GROUP_KEYS = ["pitcher_id", "game_id", "half"]

OUTPUT_COLS = [
    "pitcher_id", "game_id", "half", "date",
    "app_wpa", "n_pa",
    "n_bb", "n_so", "n_hr", "n_out", "n_1b", "n_2b", "n_3b", "n_gdp", "n_sf",
    "outs_recorded", "innings_pitched", "total_pitches",
    "start_inning", "end_inning", "appearance_order", "is_starter",
]


def aggregate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """PA-level DataFrame → 등판 단위 DataFrame으로 집계."""
    required = {
        "pitcher_id", "game_id", "half", "inning", "pa_result",
        "reward_wpa_computed", "total_pitch_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 DataFrame에 필요한 컬럼이 없습니다: {missing}")

    df = df.copy()

    # ── 경기 내 시간순 정렬(초→말) 보장 후 등판의 최초 출현 위치 기록 ─────────
    df["_half_order"] = df["half"].map(_HALF_ORDER)
    df = df.sort_values(
        ["game_id", "inning", "_half_order"], kind="stable"
    ).reset_index(drop=True)
    df["_chrono_pos"] = df.index

    # ── 등판 단위 기본 집계 ────────────────────────────────────────────────
    grouped = df.groupby(GROUP_KEYS, as_index=False).agg(
        date=("game_id", lambda s: int(s.iloc[0][:8])),
        app_wpa=("reward_wpa_computed", "sum"),
        n_pa=("pa_result", "count"),
        total_pitches=("total_pitch_count", "max"),
        start_inning=("inning", "min"),
        end_inning=("inning", "max"),
        _first_pos=("_chrono_pos", "min"),
    )

    # ── pa_result 카테고리별 카운트(단일 groupby pass) ────────────────────
    outcome_counts = (
        df.groupby(GROUP_KEYS)["pa_result"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in _OUTCOME_COL_TO_LABEL.values():
        if label not in outcome_counts.columns:
            outcome_counts[label] = 0
    rename_map = {v: k for k, v in _OUTCOME_COL_TO_LABEL.items()}
    outcome_counts = outcome_counts.rename(columns=rename_map)
    grouped = grouped.merge(
        outcome_counts[GROUP_KEYS + list(_OUTCOME_COL_TO_LABEL.keys())],
        on=GROUP_KEYS, how="left",
    )

    # ── 파생 지표 ──────────────────────────────────────────────────────────
    grouped["outs_recorded"] = (
        grouped["n_out"] + grouped["n_so"] + grouped["n_sf"] + grouped["n_gdp"] * 2
    )
    grouped["innings_pitched"] = grouped["outs_recorded"] / 3.0

    # (game_id, half) 안에서 최초 출현 위치 기준 등판 순번 부여
    grouped["appearance_order"] = (
        grouped.groupby(["game_id", "half"])["_first_pos"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    grouped["is_starter"] = grouped["appearance_order"] == 1

    grouped = grouped.drop(columns=["_first_pos"])
    grouped = grouped[OUTPUT_COLS].sort_values(
        ["date", "game_id", "half", "appearance_order"]
    ).reset_index(drop=True)

    logger.info(
        "등판 집계 완료: PA %d행 → 등판 %d건", len(df), len(grouped)
    )
    return grouped


def build_appearances(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    logger.info("로드: %d 행, %d 컬럼", *df.shape)

    df = filter_regular_season(df)  # 시범경기 PA 제외 (season_filter.py)

    app_df = aggregate_appearances(df)

    app_df.to_parquet(output_path, index=False)
    logger.info("저장 완료: %s (%d 건)", output_path, len(app_df))
    return app_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out = build_appearances()

    print("\n" + "=" * 60)
    print(f"[등판 수] {len(out):,}건 (투수 {out['pitcher_id'].nunique():,}명)")
    print(f"\n[n_pa 분포]\n{out['n_pa'].describe().to_string()}")
    print(f"\n[app_wpa 분포]\n{out['app_wpa'].describe().to_string()}")
    print(f"\n[is_starter] True: {int(out['is_starter'].sum()):,}건")
    print("=" * 60)
