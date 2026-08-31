"""
kbo_crawler/game_id_collector.py

전 구단(10개 팀) game_id 전수 수집기 (2016-01-01 ~ 2025-12-31).

get_hsk_game_ids.py와 동일한 스케줄 API를 사용하되, HH/SK 팀 필터를 제거하고
KBO 리그의 모든 경기를 수집한다. 종료(statusCode == "RESULT")·비취소
(cancel == False) 경기만 대상으로 한다.

실제 API:
  GET https://api-gw.sports.naver.com/schedule/games
  ?fields=basic,schedule,baseball,manualRelayUrl
  &upperCategoryId=kbaseball
  &categoryId=kbo
  &fromDate=2016-01-01
  &toDate=2016-01-31
  &size=500

산출:
  data/game_ids/all_games_{year}.txt   (연도별 game_id, 한 줄에 하나)
  data/game_ids/game_index.csv         (game_id, date, year, away_code,
                                         home_code, is_regular_season)

실행:
    uv run python -m kbo_crawler.game_id_collector
"""

from __future__ import annotations

import asyncio
import calendar
import csv
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Optional

import aiohttp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_analysis.methods.season_filter import is_regular_season  # noqa: E402

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
YEARS = range(2016, 2026)   # 2016 ~ 2025
MONTHS = range(1, 13)       # 1월 ~ 12월 전체
MIN_DELAY = 1.0
MAX_DELAY = 1.5

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "game_ids")
GAME_INDEX_PATH = os.path.join(OUTPUT_DIR, "game_index.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://sports.naver.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ────────────────────────────────────────────────
# 날짜 유틸
# ────────────────────────────────────────────────
def _month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


# ────────────────────────────────────────────────
# Fetcher
# ────────────────────────────────────────────────
class NaverScheduleFetcher:
    """네이버 스포츠 월별 일정 API 비동기 크롤러 (전 구단)."""

    def __init__(self, min_delay: float = MIN_DELAY, max_delay: float = MAX_DELAY):
        self.min_delay = min_delay
        self.max_delay = max_delay

    async def _random_delay(self) -> None:
        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))

    async def fetch_month(
        self,
        session: aiohttp.ClientSession,
        year: int,
        month: int,
    ) -> Optional[dict]:
        from_date, to_date = _month_range(year, month)
        params = {
            "fields": "basic,schedule,baseball,manualRelayUrl",
            "upperCategoryId": "kbaseball",
            "categoryId": "kbo",
            "fromDate": from_date,
            "toDate": to_date,
            "size": 500,
        }

        try:
            async with session.get(
                SCHEDULE_URL,
                params=params,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                logger.warning("%d-%02d → HTTP %d", year, month, resp.status)
                return None
        except Exception as exc:
            logger.error("%d-%02d → %s", year, month, exc)
            return None

    async def fetch_all(self) -> list[dict]:
        results: list[dict] = []
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            for year in YEARS:
                for month in MONTHS:
                    data = await self.fetch_month(session, year, month)
                    if data is not None:
                        results.append({"year": year, "month": month, "data": data})
                    await self._random_delay()
                logger.info("%d년 수집 완료", year)
        return results


# ────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────
class ScheduleParser:
    """월별 일정 JSON에서 종료·비취소 경기 정보를 추출 (전 구단)."""

    @staticmethod
    def extract_games(raw: dict) -> list[dict]:
        """game_id, date, away_code, home_code 딕셔너리 리스트 반환."""
        rows: list[dict] = []
        result = raw.get("result") or raw
        games: list = result.get("games", []) or []

        for game in games:
            game_id = game.get("gameId", "")
            status_code = game.get("statusCode", "")
            is_cancelled = game.get("cancel", False)

            if not game_id or status_code != "RESULT" or is_cancelled:
                continue

            rows.append({
                "game_id": game_id,
                "date": game.get("gameDate", ""),
                "away_code": game.get("awayTeamCode", ""),
                "home_code": game.get("homeTeamCode", ""),
                "away_name": game.get("awayTeamName", ""),
                "home_name": game.get("homeTeamName", ""),
            })

        return rows


# ────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────
class GameIdCollector:
    """Fetcher + Parser를 조합해 전 구단 game_id 목록·인덱스 CSV를 생성한다."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.fetcher = NaverScheduleFetcher()
        self.parser = ScheduleParser()
        os.makedirs(self.output_dir, exist_ok=True)

    async def run(self) -> list[dict]:
        logger.info("=== 전 구단 game_id 수집 시작 (%d~%d) ===", min(YEARS), max(YEARS))

        monthly_results = await self.fetcher.fetch_all()

        all_rows: dict[str, dict] = {}
        for entry in monthly_results:
            for row in self.parser.extract_games(entry["data"]):
                all_rows[row["game_id"]] = row  # game_id 중복 제거

        rows = sorted(all_rows.values(), key=lambda r: r["game_id"])
        logger.info("총 %d경기 발견", len(rows))

        self._write_year_files(rows)
        self._write_game_index(rows)
        self._print_validation(rows)

        return rows

    def _write_year_files(self, rows: list[dict]) -> None:
        by_year: dict[int, list[str]] = defaultdict(list)
        for row in rows:
            year = int(row["game_id"][:4])
            by_year[year].append(row["game_id"])

        for year, ids in sorted(by_year.items()):
            path = os.path.join(self.output_dir, f"all_games_{year}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(ids)) + "\n")
            logger.info("%d년 → %d경기 → %s", year, len(ids), path)

    def _write_game_index(self, rows: list[dict]) -> None:
        with open(GAME_INDEX_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_id", "date", "year", "away_code", "home_code",
                "away_name", "home_name", "is_regular_season",
            ])
            for row in rows:
                year = int(row["game_id"][:4])
                try:
                    regular = is_regular_season(row["game_id"])
                except ValueError:
                    regular = ""  # 개막일 설정 없는 연도 — 판별 불가
                writer.writerow([
                    row["game_id"], row["date"], year,
                    row["away_code"], row["home_code"],
                    row["away_name"], row["home_name"], regular,
                ])
        logger.info("game_index.csv 저장 완료: %s", GAME_INDEX_PATH)

    def _print_validation(self, rows: list[dict]) -> None:
        by_year_total: dict[int, int] = defaultdict(int)
        by_year_regular: dict[int, int] = defaultdict(int)
        team_years: dict[str, set[int]] = defaultdict(set)
        team_names: dict[str, set[str]] = defaultdict(set)

        for row in rows:
            year = int(row["game_id"][:4])
            by_year_total[year] += 1
            try:
                if is_regular_season(row["game_id"]):
                    by_year_regular[year] += 1
            except ValueError:
                pass
            for code, name in (
                (row["away_code"], row["away_name"]),
                (row["home_code"], row["home_name"]),
            ):
                team_years[code].add(year)
                team_names[code].add(name)

        print("\n" + "=" * 70)
        print("[연도별 경기 수] (전체 / 정규시즌)")
        for year in sorted(by_year_total):
            print(f"  {year}: {by_year_total[year]:>4}경기 총 / {by_year_regular[year]:>4}경기 정규시즌")

        print(f"\n[팀 코드 전체 목록] ({len(team_years)}개)")
        for code in sorted(team_years):
            years_str = ",".join(str(y) for y in sorted(team_years[code]))
            names_str = "/".join(sorted(team_names[code]))
            print(f"  {code:>4} ({names_str}): {years_str}")

        print("=" * 70)


# ────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    collector = GameIdCollector()
    asyncio.run(collector.run())


if __name__ == "__main__":
    main()
