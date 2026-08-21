"""
data_analysis/methods/season_filter.py

정규시즌 / 시범경기 판별 및 필터.

개막일은 config/season_opening_dates.json 외부 설정 파일로 관리한다 —
연도별 개막일을 코드에 하드코딩하지 않는다. 향후 연도를 추가할 때는
그 설정 파일만 수정하면 된다. 설정에 없는 연도의 game_id가 들어오면
조용히 넘어가지 않고 예외를 발생시킨다.

판별 로직: game_id 앞 8자리(YYYYMMDD)의 날짜가 그 연도 개막일 이상이면
정규시즌, 미만이면 시범경기.

교차 검증: game_bullpen.parquet의 (game_id, team_code)별 등록 인원 수가
BULLPEN_SIZE_THRESHOLD(20명)을 초과하면 불펜 인원 기준으로도 시범경기로
판별된다. preview 커버리지(128/153경기)가 있는 경기에 한해 개막일 판별과
비교하며, 불일치 시에도 중단하지 않고 개막일 기준을 우선 적용한다
(단, 반드시 보고).

실행:
    uv run python -m data_analysis.methods.season_filter
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/season_opening_dates.json"
BULLPEN_SIZE_THRESHOLD = 20  # 이 값을 초과하면 불펜 인원 기준 시범경기 판정

PA_INPUT_PATH      = "data_analysis/results/hsk_pa_with_wpa.parquet"
GAME_BULLPEN_PATH  = "data_analysis/results/game_bullpen.parquet"


# ────────────────────────────────────────────────────────────────────────── #
#  개막일 설정 로드
# ────────────────────────────────────────────────────────────────────────── #

def load_opening_dates(config_path: str = CONFIG_PATH) -> dict[int, date]:
    """
    config/season_opening_dates.json → {연도(int): 개막일(date)}.
    "_"로 시작하는 키(주석용 필드)는 무시한다.
    """
    import json

    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        int(year): date.fromisoformat(iso)
        for year, iso in raw.items()
        if not year.startswith("_")
    }


_OPENING_DATES: dict[int, date] | None = None


def _get_opening_dates() -> dict[int, date]:
    global _OPENING_DATES
    if _OPENING_DATES is None:
        _OPENING_DATES = load_opening_dates()
    return _OPENING_DATES


# ────────────────────────────────────────────────────────────────────────── #
#  판별
# ────────────────────────────────────────────────────────────────────────── #

def game_date(game_id: str) -> date:
    """game_id 앞 8자리(YYYYMMDD) → date."""
    return date(int(game_id[:4]), int(game_id[4:6]), int(game_id[6:8]))


def is_regular_season(game_id: str) -> bool:
    """
    game_id의 경기일이 해당 연도 개막일 이상이면 정규시즌(True), 미만이면
    시범경기(False). 설정 파일에 해당 연도가 없으면 ValueError로 중단한다
    (임의 규칙으로 조용히 처리하지 않는다).
    """
    gdate = game_date(game_id)
    opening_dates = _get_opening_dates()
    if gdate.year not in opening_dates:
        raise ValueError(
            f"{CONFIG_PATH}에 {gdate.year}년 개막일이 없습니다 (game_id={game_id}). "
            f"해당 연도를 설정 파일에 추가하세요."
        )
    return gdate >= opening_dates[gdate.year]


def add_regular_season_flag(df: pd.DataFrame, game_id_col: str = "game_id") -> pd.DataFrame:
    """df에 is_regular_season(bool) 컬럼을 추가한다."""
    df = df.copy()
    df["is_regular_season"] = df[game_id_col].apply(is_regular_season)
    return df


def filter_regular_season(df: pd.DataFrame, game_id_col: str = "game_id") -> pd.DataFrame:
    """정규시즌 행만 남기고 시범경기 행을 제외한다."""
    flagged = add_regular_season_flag(df, game_id_col)
    n_before = len(flagged)
    out = (
        flagged[flagged["is_regular_season"]]
        .drop(columns=["is_regular_season"])
        .reset_index(drop=True)
    )
    n_excluded = n_before - len(out)
    logger.info(
        "정규시즌 필터 적용: %d행 → %d행 (시범경기 제외 %d행, %.1f%%)",
        n_before, len(out), n_excluded, 100 * n_excluded / n_before if n_before else 0.0,
    )
    return out


# ────────────────────────────────────────────────────────────────────────── #
#  교차 검증: 불펜 인원 수(>20명) 기준 독립 판별
# ────────────────────────────────────────────────────────────────────────── #

def cross_validate_with_bullpen_size(
    game_ids: list[str] | None = None,
    pa_path: str = PA_INPUT_PATH,
    bullpen_path: str = GAME_BULLPEN_PATH,
    threshold: int = BULLPEN_SIZE_THRESHOLD,
) -> pd.DataFrame:
    """
    개막일 기준 판별 vs (game_id, team_code)별 불펜 인원 수 > threshold 기준
    판별을 비교한다. game_bullpen.parquet 커버리지(preview 있는 경기)에
    한해서만 비교 가능 — 커버리지 밖 경기는 결과에서 제외된다.

    Returns
    -------
    DataFrame: game_id, game_date, opening_date, max_bullpen_size,
               is_regular_by_date, is_regular_by_bullpen, mismatch
    """
    if game_ids is None:
        pa = pd.read_parquet(pa_path)
        game_ids = sorted(pa["game_id"].unique())

    bullpen = pd.read_parquet(bullpen_path)
    bullpen_size = bullpen.groupby(["game_id", "team_code"])["pitcher_id"].nunique()
    max_size_per_game = bullpen_size.groupby("game_id").max()

    rows = []
    for gid in game_ids:
        if gid not in max_size_per_game.index:
            continue  # preview 없음 → 교차검증 불가
        max_size = int(max_size_per_game.loc[gid])
        gd = game_date(gid)
        opening = _get_opening_dates()[gd.year]
        by_date = is_regular_season(gid)
        by_bullpen = max_size <= threshold
        rows.append({
            "game_id": gid,
            "game_date": gd.isoformat(),
            "opening_date": opening.isoformat(),
            "max_bullpen_size": max_size,
            "is_regular_by_date": by_date,
            "is_regular_by_bullpen": by_bullpen,
            "mismatch": by_date != by_bullpen,
        })

    result = pd.DataFrame(rows)
    n_mismatch = int(result["mismatch"].sum()) if len(result) else 0
    if n_mismatch:
        logger.warning(
            "개막일 vs 불펜인원(>%d명) 판별 불일치 %d건 — 개막일 기준을 우선 적용, 상세는 결과 표 참고",
            threshold, n_mismatch,
        )
    else:
        logger.info(
            "개막일 vs 불펜인원(>%d명) 판별 완전 일치 (%d경기 비교)", threshold, len(result)
        )
    return result


# ────────────────────────────────────────────────────────────────────────── #
#  엔트리포인트
# ────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pa = pd.read_parquet(PA_INPUT_PATH)
    games = pa[["game_id"]].drop_duplicates().reset_index(drop=True)
    games["is_regular_season"] = games["game_id"].apply(is_regular_season)
    games["year"] = games["game_id"].str[:4].astype(int)

    preseason_games = games[~games["is_regular_season"]]
    n_preseason_pa = pa[pa["game_id"].isin(preseason_games["game_id"])].shape[0]

    print("\n" + "=" * 60)
    print(f"[전체 경기] {len(games)}개 / [시범경기] {len(preseason_games)}개")
    print(f"[시범경기 PA] {n_preseason_pa}행 ({100 * n_preseason_pa / len(pa):.1f}%)")
    print(f"\n[연도별 시범경기 수]\n{preseason_games['year'].value_counts().sort_index().to_string()}")
    print(f"\n[시범경기 game_id 목록]\n{preseason_games['game_id'].tolist()}")

    cv = cross_validate_with_bullpen_size(game_ids=games["game_id"].tolist())
    print(f"\n[교차검증] 비교 가능 {len(cv)}경기 / 불일치 {int(cv['mismatch'].sum())}건")
    if cv["mismatch"].any():
        print(cv[cv["mismatch"]].to_string(index=False))
    print("=" * 60)
