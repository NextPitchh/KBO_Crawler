"""
HH(한화) vs SK(SSG/SK와이번스) 맞대결 게임 ID 수집기 (2015~2024)

[수정 내역]
기존 코드는 3가지 이유로 0경기가 발견되었음:
  1. 쿼리 파라미터: year/month → fromDate/toDate 로 변경
  2. JSON 경로: result.gameList[].games[] → result.games[] 로 변경
  3. 상태 필터: "END"/"AFTER_GAME" → "RESULT" 로 변경

실제 API:
  GET https://api-gw.sports.naver.com/schedule/games
  ?fields=basic,schedule,baseball,manualRelayUrl
  &upperCategoryId=kbaseball
  &categoryId=kbo
  &fromDate=2024-03-01
  &toDate=2024-03-31
  &size=500

Usage:
    uv run python get_hsk_game_ids.py
"""

import asyncio
import calendar
import random
from typing import Optional

import aiohttp

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
TARGET_TEAMS = frozenset({"HH", "SK"})
YEARS = range(2015, 2025)          # 2015 ~ 2024
MONTHS = range(3, 12)              # 3월 ~ 11월
OUTPUT_FILE = "hsk_game_ids_2015_2024.txt"
MIN_DELAY = 0.3
MAX_DELAY = 0.8

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
    """해당 월의 fromDate, toDate 문자열을 반환."""
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


# ────────────────────────────────────────────────
# Fetcher
# ────────────────────────────────────────────────
class NaverScheduleFetcher:
    """네이버 스포츠 월별 일정 API 비동기 크롤러."""

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

        # ★ 수정: 실제 API 파라미터에 맞춤
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
                print(f"  [WARN] {year}-{month:02d} → HTTP {resp.status}")
                return None
        except Exception as exc:
            print(f"  [ERROR] {year}-{month:02d} → {exc}")
            return None

    async def fetch_all(self) -> list[dict]:
        """연도·월 전체 일정을 순차적으로 수집 (딜레이 포함)."""
        results: list[dict] = []
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            for year in YEARS:
                for month in MONTHS:
                    data = await self.fetch_month(session, year, month)
                    if data is not None:
                        results.append({"year": year, "month": month, "data": data})
                    await self._random_delay()
                print(f"[INFO] {year}년 수집 완료")
        return results


# ────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────
class ScheduleParser:
    """월별 일정 JSON에서 HH ↔ SK 종료 경기 ID를 추출."""

    @staticmethod
    def _is_target_matchup(home: str, away: str) -> bool:
        return {home, away} == TARGET_TEAMS

    def extract_game_ids(self, raw: dict) -> list[str]:
        game_ids: list[str] = []

        # ★ 수정: 실제 응답 구조는 result.games[] (gameList 아님)
        result = raw.get("result") or raw
        games: list = result.get("games", []) or []

        for game in games:
            home = game.get("homeTeamCode", "")
            away = game.get("awayTeamCode", "")
            game_id = game.get("gameId", "")

            # ★ 수정: statusCode == "RESULT" 가 종료 상태
            #         cancel == true 인 경기는 제외
            status_code = game.get("statusCode", "")
            is_cancelled = game.get("cancel", False)

            if (
                game_id
                and status_code == "RESULT"
                and not is_cancelled
                and self._is_target_matchup(home, away)
            ):
                game_ids.append(game_id)

        return game_ids


# ────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────
class GameIdCollector:
    """Fetcher + Parser를 조합해 gameId 목록을 생성·저장."""

    def __init__(self, output_path: str = OUTPUT_FILE):
        self.output_path = output_path
        self.fetcher = NaverScheduleFetcher()
        self.parser = ScheduleParser()

    async def run(self) -> list[str]:
        print(f"=== HH vs SK 맞대결 수집 시작 ({min(YEARS)}~{max(YEARS)}) ===\n")

        monthly_results = await self.fetcher.fetch_all()

        all_ids: set[str] = set()
        for entry in monthly_results:
            ids = self.parser.extract_game_ids(entry["data"])
            all_ids.update(ids)
            if ids:
                print(
                    f"  {entry['year']}-{entry['month']:02d} → "
                    f"{len(ids)}경기 발견: {ids}"
                )

        sorted_ids = sorted(all_ids)

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_ids) + ("\n" if sorted_ids else ""))

        print(f"\n{'=' * 45}")
        print(f"총 {len(sorted_ids)}경기 발견 → {self.output_path} 저장 완료")
        print("=" * 45)
        return sorted_ids


# ────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    collector = GameIdCollector()
    asyncio.run(collector.run())