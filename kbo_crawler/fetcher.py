"""
NaverSportsAPIFetcher
  - 네이버 스포츠 문자중계 API 비동기 크롤러
  - JSON 실제 경로: result.textRelayData.textRelays
  - Rate-limit 우회: 요청 사이 랜덤 딜레이 0.5s ~ 1.5s
  - 대용량(7,200경기+) 대비 Session 을 외부에서 주입받아 Connection Pool 재사용

JSON 확정 경로:
  root
  └─ result
       ├─ textRelayData
       │    └─ textRelays[]     ← 투구 이벤트 배열
       └─ metricOption
            └─ wpaByPlate       ← 타석 WPA 변동량
"""

import asyncio
import logging
import random
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

MAX_INNINGS = 12


class NaverSportsAPIFetcher:
    """
    네이버 스포츠 문자중계 API 비동기 Fetcher.

    Usage — 단일 실행 (Session 내부 관리):
        fetcher = NaverSportsAPIFetcher()
        async with fetcher.create_session() as session:
            data = await fetcher.fetch_game(session, game_id)

    Usage — 대량 수집 (Session 외부 공유, Connection Pool 재사용):
        async with fetcher.create_session() as shared_session:
            for game_id in game_ids:
                data = await fetcher.fetch_game(shared_session, game_id)
    """

    BASE_URL = "https://api-gw.sports.naver.com/schedule/games/{game_id}/relay"

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

    def __init__(self, min_delay: float = 0.5, max_delay: float = 1.5):
        self.min_delay = min_delay
        self.max_delay = max_delay

    # ------------------------------------------------------------------ #
    #  Session 팩토리                                                       #
    # ------------------------------------------------------------------ #

    def create_session(self) -> aiohttp.ClientSession:
        """공유 Session 생성 — async with 문으로 사용."""
        connector = aiohttp.TCPConnector(ssl=False, limit=10)
        return aiohttp.ClientSession(connector=connector, headers=self.HEADERS)

    # ------------------------------------------------------------------ #
    #  내부 유틸                                                           #
    # ------------------------------------------------------------------ #

    async def _random_delay(self) -> None:
        """Rate-limit 우회용 랜덤 대기 (0.5s ~ 1.5s)."""
        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))

    # ------------------------------------------------------------------ #
    #  단일 이닝 요청                                                       #
    # ------------------------------------------------------------------ #

    async def fetch_inning(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
        inning: int,
    ) -> Optional[dict]:
        """단일 이닝 JSON 원본을 반환한다. 실패 시 None."""
        url = self.BASE_URL.format(game_id=game_id)
        params = {"inning": inning}

        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    payload = await resp.json(content_type=None)
                    logger.debug("OK  game=%s inning=%d", game_id, inning)
                    return payload

                logger.warning(
                    "HTTP %d  game=%s inning=%d", resp.status, game_id, inning
                )
                return None

        except asyncio.TimeoutError:
            logger.error("Timeout  game=%s inning=%d", game_id, inning)
            return None
        except aiohttp.ClientError as exc:
            logger.error("ClientError  game=%s inning=%d  %s", game_id, inning, exc)
            return None

    # ------------------------------------------------------------------ #
    #  한 경기 전체 이닝 수집                                               #
    # ------------------------------------------------------------------ #

    async def fetch_game(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
    ) -> list[dict]:
        """
        1~12이닝을 순차 요청하여 {'inning': int, 'data': dict} 리스트 반환.

        조기 종료 조건:
          - result.textRelayData.textRelays 가 비어 있으면
            해당 이닝까지만 경기가 진행된 것으로 판단하고 루프를 종료한다.
          - 단, 1회는 빈 응답이어도 스킵만 하고 종료하지 않는다
            (우천 취소 등으로 1회 API 가 빈 채로 내려오는 케이스 방어).
        """
        results: list[dict] = []

        for inning in range(1, MAX_INNINGS + 1):
            raw = await self.fetch_inning(session, game_id, inning)

            if raw is None:
                # 네트워크 오류 → 해당 이닝 스킵, 진행 유지
                await self._random_delay()
                continue

            # ── 올바른 JSON 깊이 탐색 ──────────────────────────────────
            # result
            #   └─ textRelayData          ← 반드시 한 단계 더 들어가야 함
            #        └─ textRelays        ← 실제 투구 이벤트 배열
            relay_root = raw.get("result", {}).get("textRelayData", {})
            relays = relay_root.get("textRelays", [])

            if not relays and inning > 1:
                logger.info(
                    "game=%s inning=%d textRelays 없음 → 경기 종료로 판단, 루프 중단",
                    game_id,
                    inning,
                )
                break

            results.append({"inning": inning, "data": raw})
            await self._random_delay()

        return results
