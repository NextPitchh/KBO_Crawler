"""
kbo_crawler/pilot_sampler.py

전 구단·전 연도 파일럿 검증용 최소 표본 선정기.

game_id_collector.py가 만든 data/game_ids/game_index.csv에서 정규시즌 경기만
대상으로, 연도별 20경기(총 200경기) 기본 표본을 뽑은 뒤, 팀별 등장 횟수가
20회 미만인 팀이 있으면 그 팀이 포함된 경기를 풀에서 우선 추가해 보정한다.

시드 고정(SEED=42)으로 재현성을 보장한다.

산출:
  data/game_ids/pilot_games.txt  (선정된 game_id, 정렬됨)

실행:
    uv run python -m kbo_crawler.pilot_sampler
"""

from __future__ import annotations

import csv
import logging
import os
import random
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GAME_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "game_ids", "game_index.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "game_ids", "pilot_games.txt")

SEED = 42
GAMES_PER_YEAR = 20
MIN_TEAM_APPEARANCES = 20

logger = logging.getLogger(__name__)


def _load_regular_season_games(path: str = GAME_INDEX_PATH) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_regular_season"] == "True":
                rows.append(row)
    return rows


def _base_sample(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """연도별 GAMES_PER_YEAR개를 무작위로 뽑는다. (selected, remaining_pool) 반환."""
    by_year: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_year[row["year"]].append(row)

    selected: list[dict] = []
    remaining: list[dict] = []

    for year in sorted(by_year):
        pool = by_year[year][:]
        rng.shuffle(pool)
        selected.extend(pool[:GAMES_PER_YEAR])
        remaining.extend(pool[GAMES_PER_YEAR:])

    return selected, remaining


def _team_counts(games: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for g in games:
        counts[g["away_code"]] += 1
        counts[g["home_code"]] += 1
    return counts


def _top_up_underrepresented_teams(
    selected: list[dict],
    remaining: list[dict],
    rng: random.Random,
    min_appearances: int = MIN_TEAM_APPEARANCES,
) -> list[dict]:
    """
    팀별 등장 횟수가 min_appearances 미만이면, remaining 풀에서 해당 팀이
    포함된 경기를 우선 추가한다. 매 반복마다 "가장 많은 부족 팀을 동시에
    커버하는" 경기를 고른다(탐욕적 보정).
    """
    selected = selected[:]
    remaining = remaining[:]
    counts = _team_counts(selected)

    all_teams = set()
    for g in remaining + selected:
        all_teams.add(g["away_code"])
        all_teams.add(g["home_code"])

    while True:
        deficient = {t for t in all_teams if counts.get(t, 0) < min_appearances}
        if not deficient:
            break

        candidates = [
            g for g in remaining
            if g["away_code"] in deficient or g["home_code"] in deficient
        ]
        if not candidates:
            logger.warning(
                "풀 소진 — 여전히 %d회 미만인 팀: %s",
                min_appearances,
                sorted(deficient),
            )
            break

        def _coverage(g: dict) -> int:
            return int(g["away_code"] in deficient) + int(g["home_code"] in deficient)

        max_cov = max(_coverage(g) for g in candidates)
        best = [g for g in candidates if _coverage(g) == max_cov]
        chosen = rng.choice(best)

        selected.append(chosen)
        remaining.remove(chosen)
        counts[chosen["away_code"]] += 1
        counts[chosen["home_code"]] += 1

    return selected


def select_pilot_games(
    game_index_path: str = GAME_INDEX_PATH,
    seed: int = SEED,
) -> list[dict]:
    rng = random.Random(seed)
    rows = _load_regular_season_games(game_index_path)
    logger.info("정규시즌 경기 %d개 로드", len(rows))

    base_selected, remaining = _base_sample(rows, rng)
    logger.info("연도별 %d경기 기본 표본 → %d경기", GAMES_PER_YEAR, len(base_selected))

    final_selected = _top_up_underrepresented_teams(base_selected, remaining, rng)
    logger.info(
        "팀별 최소 %d회 보정 후 최종 %d경기 (추가 %d경기)",
        MIN_TEAM_APPEARANCES, len(final_selected), len(final_selected) - len(base_selected),
    )

    return final_selected


def _print_summary(games: list[dict]) -> None:
    team_counts = _team_counts(games)
    year_counts: dict[str, int] = defaultdict(int)
    for g in games:
        year_counts[g["year"]] += 1

    print("\n" + "=" * 60)
    print(f"[파일럿 표본] 총 {len(games)}경기")

    print("\n[연도별 경기 수]")
    for year in sorted(year_counts):
        print(f"  {year}: {year_counts[year]}경기")

    print("\n[팀별 등장 횟수]")
    for team in sorted(team_counts):
        flag = "" if team_counts[team] >= MIN_TEAM_APPEARANCES else "  <-- 미달!"
        print(f"  {team}: {team_counts[team]}회{flag}")
    print("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    games = select_pilot_games()
    game_ids = sorted(g["game_id"] for g in games)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(game_ids) + "\n")
    logger.info("저장 완료: %s (%d경기)", OUTPUT_PATH, len(game_ids))

    _print_summary(games)


if __name__ == "__main__":
    main()
