"""
data_analysis/methods/bullpen_state.py

각 PA 시점의 불펜 소모 상황(수비팀 기준)을 계산한다.

입력  : data_analysis/results/hsk_pa_with_wpa.parquet (11,984 PA, 읽기 전용)
        data_analysis/results/game_bullpen.parquet     (128경기 preview 커버리지)
출력  : data_analysis/results/pa_bullpen_state.parquet

수비팀 규칙(외부 진실 — 재해석 금지):
  half="top" → 수비=홈팀(is_home=True)
  half="bot" → 수비=원정팀(is_home=False)

game_id 위치 기반 팀 코드(YYYYMMDD + Away2 + Home2 + ...)는 preview 커버리지와
무관하게 153경기 전체에서 계산 가능 — n_pitchers_used 등 "전 경기 공통" 컬럼과
estimated 폴백의 (시즌, 팀) 매칭 모두 이 파싱에 의존한다.

실행:
    uv run python -m data_analysis.methods.bullpen_state
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PA_INPUT_PATH      = "data_analysis/results/hsk_pa_with_wpa.parquet"
BULLPEN_INPUT_PATH = "data_analysis/results/game_bullpen.parquet"
OUTPUT_PATH        = "data_analysis/results/pa_bullpen_state.parquet"

_HALF_ORDER = {"top": 0, "bot": 1}
GROUP_KEYS = ["game_id", "half"]

OUTPUT_COLS = [
    "_orig_idx", "game_id", "inning", "home_or_away", "half", "pitcher_id",
    "n_pitchers_used", "current_pitcher_pa_in_app", "is_pitcher_change",
    "bullpen_listed", "bullpen_used", "bullpen_available",
    "bullpen_available_ratio", "bullpen_source",
]


def _defense_team_code(game_id: str) -> tuple[str, str]:
    """game_id 위치 기반 (away_code, home_code) 파싱. YYYYMMDD+Away2+Home2+..."""
    return game_id[8:10], game_id[10:12]


# ────────────────────────────────────────────────────────────────────────── #
#  Step 1: 전 경기 공통 컬럼 — 등판 투수 수 / 현재 투수 타석 수 / 투수교체
# ────────────────────────────────────────────────────────────────────────── #

def _add_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    (game_id, half) 그룹 내 타석 순서대로 누적. 153경기 전체(preview 커버리지
    무관)에서 계산 가능.
    """
    # 첫 등장 여부 — 그 (game_id, half) 안에서 이 투수의 첫 타석인가
    df["_first_occ"] = ~df.duplicated(subset=GROUP_KEYS + ["pitcher_id"], keep="first")
    df["n_pitchers_used"] = df.groupby(GROUP_KEYS)["_first_occ"].cumsum().astype(int)

    # 현재 투수가 이번 등판(Task3 정의: pitcher_id×game_id×half 조합)에서
    # 소화한 타석 수 — 등판 도중 다른 투수가 끼어들어도 같은 조합이면 계속 누적
    df["current_pitcher_pa_in_app"] = (
        df.groupby(GROUP_KEYS + ["pitcher_id"]).cumcount() + 1
    )

    # 직전 PA 대비 투수 교체 여부 — 그룹(half) 내부만 비교, 그룹 첫 PA는 False
    # (직전 PA 자체가 존재하지 않으므로 "교체 이벤트"로 볼 수 없음)
    prev_pitcher = df.groupby(GROUP_KEYS)["pitcher_id"].shift(1)
    df["is_pitcher_change"] = (df["pitcher_id"] != prev_pitcher) & prev_pitcher.notna()

    return df


# ────────────────────────────────────────────────────────────────────────── #
#  Step 2: 팀 식별자 부착 (game_id 파싱 — 153경기 전체)
# ────────────────────────────────────────────────────────────────────────── #

def _add_team_identity(df: pd.DataFrame) -> pd.DataFrame:
    away_home = df["game_id"].apply(_defense_team_code)
    away_code = away_home.str[0]
    home_code = away_home.str[1]

    df["_defense_is_home"] = df["half"] == "top"
    df["defense_team_code"] = np.where(df["_defense_is_home"], home_code, away_code)
    df["season_year"] = df["game_id"].str[:4].astype(int)
    return df


# ────────────────────────────────────────────────────────────────────────── #
#  Step 3: 실측 불펜(preview) 컬럼 — 128경기 커버리지
# ────────────────────────────────────────────────────────────────────────── #

def _build_roster_maps(bullpen_df: pd.DataFrame) -> tuple[dict, dict]:
    listed_size = (
        bullpen_df.groupby(["game_id", "is_home"])["pitcher_id"].nunique().to_dict()
    )
    listed_set = (
        bullpen_df.groupby(["game_id", "is_home"])["pitcher_id"].apply(set).to_dict()
    )
    return listed_size, listed_set


def _add_preview_bullpen_columns(df: pd.DataFrame, bullpen_df: pd.DataFrame) -> pd.DataFrame:
    listed_size, listed_set = _build_roster_maps(bullpen_df)

    df["bullpen_listed_raw"] = df.apply(
        lambda r: listed_size.get((r["game_id"], r["_defense_is_home"]), np.nan), axis=1
    )
    df["_is_listed_pitcher"] = df.apply(
        lambda r: r["pitcher_id"] in listed_set.get((r["game_id"], r["_defense_is_home"]), set()),
        axis=1,
    )

    # 명단 투수 중 이미 등판한 수 — 명단 투수의 "첫 등장" 시점에만 카운트
    df["_listed_first_occ"] = df["_first_occ"] & df["_is_listed_pitcher"]
    df["bullpen_used_raw"] = (
        df.groupby(GROUP_KEYS)["_listed_first_occ"].cumsum().astype(int)
    )
    return df


# ────────────────────────────────────────────────────────────────────────── #
#  Step 4: 폴백(estimated) — preview 없는 25경기 + bullpen_listed=0 이상치
# ────────────────────────────────────────────────────────────────────────── #

def _build_median_lookup(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    유효(bullpen_listed_raw > 0)한 (game_id, half) 단위로 중복 제거 후,
    (시즌, 팀) 별 중앙값과 전체 중앙값을 계산한다.
    """
    valid_mask = df["bullpen_listed_raw"].notna() & (df["bullpen_listed_raw"] > 0)
    valid_unique = (
        df.loc[valid_mask, ["game_id", "half", "defense_team_code", "season_year", "bullpen_listed_raw"]]
        .drop_duplicates(subset=GROUP_KEYS)
    )
    team_season_median = (
        valid_unique.groupby(["season_year", "defense_team_code"])["bullpen_listed_raw"].median()
    )
    global_median = float(valid_unique["bullpen_listed_raw"].median())
    return team_season_median, global_median


def _apply_fallback(df: pd.DataFrame) -> pd.DataFrame:
    # bullpen_listed_raw=0(2020-05-30 등 원본 API 이슈)도 preview 무효로 간주 →
    # "같은 시즌·같은 팀" 폴백 경로로 전환한다.
    valid_mask = df["bullpen_listed_raw"].notna() & (df["bullpen_listed_raw"] > 0)

    team_season_median, global_median = _build_median_lookup(df)

    def _estimate(season_year, team_code):
        key = (season_year, team_code)
        if key in team_season_median.index:
            return float(team_season_median.loc[key])
        return global_median

    df["bullpen_source"] = np.where(valid_mask, "preview", "estimated")

    df["bullpen_listed"] = df["bullpen_listed_raw"]
    fallback_rows = ~valid_mask
    df.loc[fallback_rows, "bullpen_listed"] = df.loc[fallback_rows].apply(
        lambda r: _estimate(r["season_year"], r["defense_team_code"]), axis=1
    )

    # estimated 행의 bullpen_used: 실제 명단이 없으므로 "선발 제외 등판 투수 수"로 근사
    estimated_used = (df["n_pitchers_used"] - 1).clip(lower=0)
    df["bullpen_used"] = np.where(valid_mask, df["bullpen_used_raw"], estimated_used)

    df["bullpen_available"] = (df["bullpen_listed"] - df["bullpen_used"]).clip(lower=0)
    df["bullpen_available_ratio"] = (
        (df["bullpen_available"] / df["bullpen_listed"]).clip(lower=0, upper=1)
    )
    df.loc[df["bullpen_listed"] == 0, "bullpen_available_ratio"] = np.nan

    return df


# ────────────────────────────────────────────────────────────────────────── #
#  메인
# ────────────────────────────────────────────────────────────────────────── #

def build_bullpen_state(
    pa_path: str = PA_INPUT_PATH,
    bullpen_path: str = BULLPEN_INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(pa_path)
    bullpen_df = pd.read_parquet(bullpen_path)
    logger.info("로드: PA %d행 / bullpen 명단 %d행", len(df), len(bullpen_df))

    required = {"_orig_idx", "game_id", "inning", "home_or_away", "half", "pitcher_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 PA DataFrame에 필요한 컬럼이 없습니다: {missing}")

    # 원본 순서(=state_transition.py가 이미 보장한 경기 시간순) 그대로 사용.
    # 안전을 위해 명시적으로 한 번 더 정렬한다(그룹 내 cumsum 로직이 순서에 의존).
    df["_half_order"] = df["half"].map(_HALF_ORDER)
    df = df.sort_values(["game_id", "inning", "_half_order"], kind="stable").reset_index(drop=True)
    df = df.drop(columns=["_half_order"])

    df = _add_common_columns(df)
    df = _add_team_identity(df)
    df = _add_preview_bullpen_columns(df, bullpen_df)
    df = _apply_fallback(df)

    out = df[OUTPUT_COLS].sort_values("_orig_idx").reset_index(drop=True)

    n_source = out["bullpen_source"].value_counts()
    logger.info(
        "불펜 상태 계산 완료 | preview=%d행(%.1f%%) / estimated=%d행(%.1f%%)",
        n_source.get("preview", 0), 100 * n_source.get("preview", 0) / len(out),
        n_source.get("estimated", 0), 100 * n_source.get("estimated", 0) / len(out),
    )

    out.to_parquet(output_path, index=False)
    logger.info("저장 완료: %s (%d 행)", output_path, len(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out = build_bullpen_state()

    print("\n" + "=" * 60)
    print(f"[pa_bullpen_state] {len(out):,} 행")
    print(f"\n[bullpen_source 분포]\n{out['bullpen_source'].value_counts().to_string()}")
    print(f"\n[n_pitchers_used 분포]\n{out['n_pitchers_used'].describe().to_string()}")
    print(f"\n[bullpen_available_ratio 결측률] {out['bullpen_available_ratio'].isna().mean():.1%}")
    non_monotonic = (
        out.sort_values(["game_id", "_orig_idx"])
        .groupby(["game_id", "half"])["n_pitchers_used"]
        .apply(lambda s: (s.diff().dropna() < 0).any())
    )
    print(f"\n[n_pitchers_used 단조성 위반 그룹 수] {int(non_monotonic.sum())}")
    print("=" * 60)
