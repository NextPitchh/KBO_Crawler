"""
kbo_crawler/final_scores_scan.py

전 경기(2016-2025) 최종 스코어를 월별 schedule API로 일괄 수집한다
(경기당 1회씩 6,000+번 호출하는 대신, 이미 검증된 120회 월별 요청 패턴을
game_id_collector.py에서 그대로 재사용 — 코드 수정 없이 재사용).

터미널 PA 득점 보정(state_transition 이후 후처리)의 ground truth로 쓰인다.

산출: data/game_ids/final_scores.csv
      컬럼: game_id, home_score, away_score, winner(HOME/AWAY/DRAW)

실행:
    uv run python -m kbo_crawler.final_scores_scan
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.game_id_collector import NaverScheduleFetcher  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "game_ids", "final_scores.csv")


def extract_scores(raw: dict) -> list[dict]:
    rows = []
    result = raw.get("result") or raw
    for game in result.get("games", []) or []:
        game_id = game.get("gameId", "")
        if not game_id:
            continue
        if game.get("cancel"):
            continue
        home_score = game.get("homeTeamScore")
        away_score = game.get("awayTeamScore")
        if home_score is None or away_score is None:
            continue
        winner = game.get("winner", "")
        rows.append({
            "game_id": game_id, "home_score": home_score, "away_score": away_score,
            "winner": winner,
        })
    return rows


async def scan_all() -> list[dict]:
    fetcher = NaverScheduleFetcher()
    monthly_results = await fetcher.fetch_all()

    all_rows: dict[str, dict] = {}
    for entry in monthly_results:
        for row in extract_scores(entry["data"]):
            all_rows[row["game_id"]] = row

    return sorted(all_rows.values(), key=lambda r: r["game_id"])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    rows = asyncio.run(scan_all())
    logger.info("최종 스코어 수집 완료: %d경기", len(rows))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game_id", "home_score", "away_score", "winner"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("저장 완료: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
