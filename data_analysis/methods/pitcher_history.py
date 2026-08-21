"""
data_analysis/methods/pitcher_history.py

각 등판(appearance)에 대해 "그 경기 이전까지"의 시점 안전(leakage-free)
누적 투수 이력을 계산한다.

핵심: expanding() + shift(1). 현재 등판을 절대 포함하지 않는다.
정렬: (pitcher_id, date, game_id) — 날짜 동률 시 game_id로 tie-break.

입력  : data_analysis/results/pitcher_appearances.parquet
출력  : data_analysis/results/pitcher_history.parquet
        data_analysis/results/league_baseline.json

실행:
    uv run python -m data_analysis.methods.pitcher_history
"""

from __future__ import annotations

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd

from .season_filter import is_regular_season

logger = logging.getLogger(__name__)

INPUT_PATH          = "data_analysis/results/pitcher_appearances.parquet"
OUTPUT_PATH          = "data_analysis/results/pitcher_history.parquet"
LEAGUE_BASELINE_PATH = "data_analysis/results/league_baseline.json"

ESTABLISHED_MIN_APPS = 10  # "표본 확보" 기준 (사전 확인된 사실: 52명/34%)


# ────────────────────────────────────────────────────────────────────────── #
#  Step 1: 시점 안전 누적 통계
# ────────────────────────────────────────────────────────────────────────── #

def compute_prior_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    등판 단위 DataFrame(pitcher_appearances)에 prior_* 컬럼을 추가한다.

    모든 prior_* 값은 "현재 등판 이전"의 등판만으로 계산되며(expanding + shift(1)),
    현재 등판 자체의 정보는 절대 섞이지 않는다.
    """
    required = {"pitcher_id", "date", "game_id", "app_wpa", "n_pa",
                "n_bb", "n_so", "n_hr", "innings_pitched"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 DataFrame에 필요한 컬럼이 없습니다: {missing}")

    df = df.sort_values(
        ["pitcher_id", "date", "game_id"], kind="stable"
    ).reset_index(drop=True)

    g = df.groupby("pitcher_id", sort=False)

    # 이전 등판 횟수 (cumcount는 현재 행 이전 개수를 셈 → 그 자체로 leakage-free)
    df["prior_n_apps"] = g.cumcount()

    # 변동성/평균 WPA — expanding().shift(1): 현재 행의 값은 통계에서 제외
    df["prior_wpa_mean"] = g["app_wpa"].transform(lambda s: s.expanding().mean().shift(1))
    df["prior_wpa_std"]  = g["app_wpa"].transform(lambda s: s.expanding().std().shift(1))

    # 누적 카운트(현재 행 제외) — cumsum().shift(1)
    cum_n_pa = g["n_pa"].transform(lambda s: s.cumsum().shift(1))
    cum_n_bb = g["n_bb"].transform(lambda s: s.cumsum().shift(1))
    cum_n_so = g["n_so"].transform(lambda s: s.cumsum().shift(1))
    cum_n_hr = g["n_hr"].transform(lambda s: s.cumsum().shift(1))
    cum_innings = g["innings_pitched"].transform(lambda s: s.cumsum().shift(1))

    # 비율 지표: 첫 등판(분모 0/NaN)은 정의상 결측(NaN) — 임의로 0 채우지 않음
    df["prior_bb_rate"] = cum_n_bb / cum_n_pa
    df["prior_so_rate"] = cum_n_so / cum_n_pa
    df["prior_hr_rate"] = cum_n_hr / cum_n_pa
    df["prior_avg_pa_per_app"] = cum_n_pa / df["prior_n_apps"].replace(0, np.nan)

    # 순수 누적 카운트: "이전 등판 없음" = 0 (구조적으로 의미 있는 값이므로 0 채움)
    df["prior_n_pa"] = cum_n_pa.fillna(0)
    df["prior_innings"] = cum_innings.fillna(0)

    logger.info(
        "prior_* 계산 완료: %d 투수, %d 등판 (첫 등판 %d건 → prior_n_apps=0)",
        df["pitcher_id"].nunique(), len(df), int((df["prior_n_apps"] == 0).sum()),
    )
    return df


# ────────────────────────────────────────────────────────────────────────── #
#  Step 2: leakage 검증
# ────────────────────────────────────────────────────────────────────────── #

def verify_no_leakage(
    df: pd.DataFrame,
    n_pitchers: int = 15,
    seed: int = 42,
) -> None:
    """
    랜덤 n_pitchers명의 투수를 뽑아, 각 등판의 prior_wpa_std가 "그 이전 등판들만"으로
    독립 재계산한 값과 정확히 일치하는지 assert.

    하나라도 불일치하면 즉시 AssertionError를 발생시킨다.
    """
    rng = np.random.default_rng(seed)
    unique_pitchers = df["pitcher_id"].unique()
    sample_size = min(n_pitchers, len(unique_pitchers))
    sampled = rng.choice(unique_pitchers, size=sample_size, replace=False)

    n_apps_checked = 0
    for pid in sampled:
        sub = (
            df[df["pitcher_id"] == pid]
            .sort_values(["date", "game_id"], kind="stable")
            .reset_index(drop=True)
        )
        for i in range(len(sub)):
            prior = sub.iloc[:i]  # 현재 등판(i) 이전 것들만
            expected_std = prior["app_wpa"].std() if len(prior) >= 2 else np.nan
            actual_std = sub.loc[i, "prior_wpa_std"]

            expected_nan = pd.isna(expected_std)
            actual_nan = pd.isna(actual_std)
            if expected_nan != actual_nan:
                raise AssertionError(
                    f"Leakage 발견! pitcher_id={pid}, i={i}: "
                    f"expected={expected_std} actual={actual_std} (NaN 불일치)"
                )
            if not expected_nan and not np.isclose(expected_std, actual_std, atol=1e-9):
                raise AssertionError(
                    f"Leakage 발견! pitcher_id={pid}, i={i}: "
                    f"expected={expected_std} actual={actual_std}"
                )
            n_apps_checked += 1

    print(
        f"[검증 통과] 투수 {sample_size}명 / 등판 {n_apps_checked}건 — "
        f"leakage 없음 확인 (prior_wpa_std 정확 일치)"
    )


# ────────────────────────────────────────────────────────────────────────── #
#  Step 3: 리그 베이스라인
# ────────────────────────────────────────────────────────────────────────── #

def compute_league_baseline(
    df: pd.DataFrame,
    min_apps: int = ESTABLISHED_MIN_APPS,
) -> dict:
    """
    리그 평균/표준편차 산출 — 정규시즌 · prior_n_apps >= min_apps · 투수 단위.

    v1(등판 단위 sd)의 문제: bb_rate_sd(0.155)가 평균(0.094)의 1.65배,
    hr_rate_sd(0.070)가 평균(0.026)의 2.7배로 나왔다. 원인은 등판 수가
    적은 투수의 극단값(1타석 1볼넷 → bb_rate=1.0)이 등판 단위 분포를
    오염시켰기 때문. min_apps 필터로 표본 부족 등판을 걷어낸 뒤에도,
    "등판" 단위로 sd를 구하면 등판이 많은 투수가 과대 반영된다.

    수정: 투수 단위로 접어(각 투수의 마지막 시점 prior_* 값 하나만 사용)
    sd를 계산한다. 평균(league_*_rate)은 여전히 (사건 수 합/타석 수 합)
    가중 평균 — 표본이 큰 투수의 비율이 더 정확한 추정이므로 가중치를
    유지하되, sd만 투수 단위로 분리한다.

    이 함수는 호출 시점의 df가 이미 정규시즌으로 필터되어 있다고 가정하지
    않고, season_filter.is_regular_season()으로 다시 한 번 걸러낸다
    (방어적 — 이 함수의 계약 자체가 "정규시즌만"이라는 걸 코드로 못박는다).
    """
    required = {"pitcher_id", "date", "game_id", "prior_n_apps", "prior_wpa_std",
                "prior_bb_rate", "prior_so_rate", "prior_hr_rate", "prior_n_pa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 DataFrame에 필요한 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["_is_regular"] = df["game_id"].apply(is_regular_season)

    established = df[df["_is_regular"] & (df["prior_n_apps"] >= min_apps)]
    per_pitcher = (
        established.sort_values(["pitcher_id", "date", "game_id"], kind="stable")
        .groupby("pitcher_id")
        .last()
    )

    def _weighted_rate(rate_col: str) -> float:
        return float(
            (per_pitcher[rate_col] * per_pitcher["prior_n_pa"]).sum()
            / per_pitcher["prior_n_pa"].sum()
        )

    league_wpa_std = float(per_pitcher["prior_wpa_std"].median())
    league_wpa_std_sd = float(per_pitcher["prior_wpa_std"].std())
    league_bb_rate = _weighted_rate("prior_bb_rate")
    league_so_rate = _weighted_rate("prior_so_rate")
    league_hr_rate = _weighted_rate("prior_hr_rate")
    league_bb_rate_sd = float(per_pitcher["prior_bb_rate"].std())
    league_so_rate_sd = float(per_pitcher["prior_so_rate"].std())
    league_hr_rate_sd = float(per_pitcher["prior_hr_rate"].std())

    baseline = {
        "league_wpa_std": league_wpa_std,
        "league_wpa_std_sd": league_wpa_std_sd,
        "league_bb_rate": league_bb_rate,
        "league_bb_rate_sd": league_bb_rate_sd,
        "league_so_rate": league_so_rate,
        "league_so_rate_sd": league_so_rate_sd,
        "league_hr_rate": league_hr_rate,
        "league_hr_rate_sd": league_hr_rate_sd,
        "n_established_pitchers": int(len(per_pitcher)),
        "n_regular_season_appearances": int(df["_is_regular"].sum()),
        "n_total_appearances": int(len(df)),
        "established_min_apps": min_apps,
        "filter_applied": {"regular_season_only": True, "min_prior_apps": min_apps},
        "opening_dates_source": "config/season_opening_dates.json",
    }

    for name, mean_v, sd_v in [
        ("bb_rate", league_bb_rate, league_bb_rate_sd),
        ("so_rate", league_so_rate, league_so_rate_sd),
        ("hr_rate", league_hr_rate, league_hr_rate_sd),
    ]:
        if sd_v >= mean_v:
            logger.warning(
                "league_%s_sd(%.4f) >= league_%s(%.4f) — 극단값이 여전히 남아있을 수 있음, 원인 분석 필요",
                name, sd_v, name, mean_v,
            )
        else:
            logger.info("league_%s_sd(%.4f) < league_%s(%.4f) — 정상 범위", name, sd_v, name, mean_v)

    logger.info(
        "리그 베이스라인(정규시즌·prior_n_apps>=%d·투수 단위, 투수 %d명) 산출 완료",
        min_apps, len(per_pitcher),
    )
    return baseline


# ────────────────────────────────────────────────────────────────────────── #
#  메인 빌드
# ────────────────────────────────────────────────────────────────────────── #

def build_pitcher_history(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    baseline_path: str = LEAGUE_BASELINE_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    logger.info("로드: %d 등판", len(df))

    hist_df = compute_prior_stats(df)
    verify_no_leakage(hist_df)

    baseline = compute_league_baseline(hist_df)

    # 최초 1회만 v1으로 백업 — 이후 재실행 시 v1(수정 전 원본)을 덮어쓰지 않는다
    if os.path.isfile(baseline_path):
        backup_path = os.path.join(os.path.dirname(baseline_path), "league_baseline_v1.json")
        if not os.path.isfile(backup_path):
            shutil.copy(baseline_path, backup_path)
            logger.info("기존 league_baseline.json → %s 백업", backup_path)
        else:
            logger.info("league_baseline_v1.json 이미 존재 — 백업 스킵(최초본 보존)")

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    logger.info("저장 완료: %s", baseline_path)

    hist_df.to_parquet(output_path, index=False)
    logger.info("저장 완료: %s (%d 행)", output_path, len(hist_df))
    return hist_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out = build_pitcher_history()

    print("\n" + "=" * 60)
    print(f"[pitcher_history] {len(out):,} 행")
    print(f"\n[prior_n_apps 분포]\n{out['prior_n_apps'].describe().to_string()}")
    print(f"\n[prior_wpa_std 결측률] {out['prior_wpa_std'].isna().mean():.1%}")
    with open(LEAGUE_BASELINE_PATH, encoding="utf-8") as f:
        print(f"\n[league_baseline]\n{json.dumps(json.load(f), ensure_ascii=False, indent=2)}")
    print("=" * 60)
