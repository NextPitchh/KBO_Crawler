"""
kbo_crawler/game_status_scan.py

schedule API 월별 일정을 전수 재조회하여(취소/서스펜드 경기 포함 — 기존
game_id_collector.py의 ScheduleParser는 statusCode=='RESULT' and not cancel
경기만 남기므로 취소 경기가 game_index.csv에서 원천적으로 빠져 있었다),
각 경기를 normal/called/suspended/cancelled로 분류한다.

기존 game_id_collector.py(NaverScheduleFetcher)는 수정하지 않고 그대로
재사용하며, 이 모듈은 필터링 없는 새 파서만 추가한다.

분류 규칙:
  - cancelled : cancel == True
  - suspended : cancel == False and suspended == True
                (경기가 중단되어 이후 이어하기로 재개 예정 — statusCode가
                아직 RESULT가 아닌 경우도 포함)
  - called    : statusCode == 'RESULT' and not cancel and not suspended
                and statusInfo의 최종 이닝 < 9 (콜드게임, 승자 확정)
  - normal    : statusCode == 'RESULT' and not cancel and not suspended
                and 최종 이닝 >= 9

실행:
    uv run python -m kbo_crawler.game_status_scan
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.game_id_collector import NaverScheduleFetcher, YEARS, MONTHS  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "game_ids", "game_index.csv")
_STATUS_INFO_RE = re.compile(r"(\d+)회")


def classify_status(game: dict) -> str:
    if game.get("cancel"):
        return "cancelled"
    if game.get("suspended"):
        return "suspended"
    if game.get("statusCode") != "RESULT":
        return "not_completed"

    status_info = game.get("statusInfo") or ""
    m = _STATUS_INFO_RE.search(status_info)
    if not m:
        return "normal"  # 파싱 불가 — 보수적으로 normal(추가 확인 필요 시 로그 참고)

    final_inning = int(m.group(1))
    return "called" if final_inning < 9 else "normal"


def extract_all_games_with_status(raw: dict) -> list[dict]:
    """cancel 여부와 무관하게 모든 경기를 상태 분류와 함께 반환."""
    rows: list[dict] = []
    result = raw.get("result") or raw
    games: list = result.get("games", []) or []

    for game in games:
        game_id = game.get("gameId", "")
        if not game_id:
            continue
        rows.append({
            "game_id": game_id,
            "date": game.get("gameDate", ""),
            "away_code": game.get("awayTeamCode", ""),
            "home_code": game.get("homeTeamCode", ""),
            "away_name": game.get("awayTeamName", ""),
            "home_name": game.get("homeTeamName", ""),
            "status_code": game.get("statusCode", ""),
            "status_info": game.get("statusInfo", ""),
            "cancel": game.get("cancel", False),
            "suspended": game.get("suspended", False),
            "game_status": classify_status(game),
        })
    return rows


async def scan_all() -> list[dict]:
    fetcher = NaverScheduleFetcher()
    monthly_results = await fetcher.fetch_all()

    all_rows: dict[str, dict] = {}
    for entry in monthly_results:
        for row in extract_all_games_with_status(entry["data"]):
            all_rows[row["game_id"]] = row

    return sorted(all_rows.values(), key=lambda r: r["game_id"])


def merge_into_game_index(rows: list[dict], game_index_path: str = OUTPUT_PATH) -> None:
    """
    기존 game_index.csv(정규시즌 판별 등 포함)에 game_status 컬럼을 병합한다.
    기존 파일의 행 구성(취소 경기 제외)은 그대로 유지하고, 새 컬럼만 덧붙인다.
    별도로 취소/서스펜드 경기 목록은 game_index_all_statuses.csv에 전체 저장.
    """
    status_map = {r["game_id"]: r for r in rows}

    existing_rows = []
    with open(game_index_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["game_status", "status_info"]
        for row in reader:
            status_row = status_map.get(row["game_id"])
            row["game_status"] = status_row["game_status"] if status_row else "unknown"
            row["status_info"] = status_row["status_info"] if status_row else ""
            existing_rows.append(row)

    with open(game_index_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
    logger.info("game_index.csv에 game_status 컬럼 병합 완료: %s", game_index_path)

    # 전체 상태(취소 포함) 별도 저장
    all_path = os.path.join(os.path.dirname(game_index_path), "game_index_all_statuses.csv")
    with open(all_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("전체 상태(취소/서스펜드 포함) 저장 완료: %s", all_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    rows = asyncio.run(scan_all())
    logger.info("전체 %d경기(취소/서스펜드 포함) 스캔 완료", len(rows))

    merge_into_game_index(rows)

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["game_status"]] += 1

    print("\n" + "=" * 50)
    print("[전체 game_status 분포 (2016-2025, 전 구단)]")
    for status, cnt in sorted(counts.items()):
        print(f"  {status}: {cnt}건")
    print("=" * 50)


if __name__ == "__main__":
    main()
