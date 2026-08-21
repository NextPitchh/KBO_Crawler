"""
PreviewAPIFetcher
  - 네이버 스포츠 경기 프리뷰(선발투수/불펜명단/시즌스탯) API 크롤러
  - JSON 실제 경로: result.previewData
  - 2017-05-30 경기부터 제공 (그 이전 game_id는 요청 자체를 스킵)
  - fetcher.py의 요청 헤더 · 세션 구성을 재사용한다.
  - 요청 사이 최소 1초 대기, 실패 시 지수 백오프로 최대 3회 재시도.

저장 형식: data/preview/{game_id}.json (원본 그대로 보존)

실행:
    uv run python -m kbo_crawler.preview_fetcher
"""

import asyncio
import json
import logging
import os
import random

import aiohttp
from tqdm import tqdm

from .fetcher import NaverSportsAPIFetcher

logger = logging.getLogger(__name__)

PREVIEW_AVAILABLE_FROM = 20170530  # YYYYMMDD — 이 날짜 이전 game_id는 API 미제공


def game_date(game_id: str) -> int:
    """game_id 앞 8자리(YYYYMMDD) → int."""
    return int(game_id[:8])


def is_preview_supported(game_id: str) -> bool:
    """preview API가 제공되는 경기인지(2017-05-30 이후) 여부."""
    return game_date(game_id) >= PREVIEW_AVAILABLE_FROM


# ────────────────────────────────────────────────────────────────────────── #
#  저수준 Fetcher
# ────────────────────────────────────────────────────────────────────────── #

class PreviewAPIFetcher:
    """
    단일 game_id 당 1건의 프리뷰(선발/불펜/시즌 스탯) JSON을 수집한다.
    fetcher.py의 NaverSportsAPIFetcher와 동일한 요청 헤더를 재사용한다.
    """

    BASE_URL = "https://api-gw.sports.naver.com/schedule/games/{game_id}/preview"
    HEADERS = NaverSportsAPIFetcher.HEADERS

    def __init__(self, min_interval: float = 1.0, max_retries: int = 3):
        self.min_interval = min_interval
        self.max_retries = max_retries

    def create_session(self) -> aiohttp.ClientSession:
        """공유 Session 생성 — async with 문으로 사용."""
        connector = aiohttp.TCPConnector(ssl=False, limit=10)
        return aiohttp.ClientSession(connector=connector, headers=self.HEADERS)

    async def fetch_preview(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
    ) -> dict | None:
        """
        프리뷰 JSON 원본을 반환한다. 실패 시 지수 백오프로 최대
        self.max_retries회 재시도. 모두 실패하면 None.

        요청 성공/실패와 무관하게 매 시도 후 self.min_interval초를
        대기하여 요청 간 최소 간격을 보장한다.
        """
        url = self.BASE_URL.format(game_id=game_id)

        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        payload = await resp.json(content_type=None)
                        logger.debug("OK  game=%s", game_id)
                        await asyncio.sleep(self.min_interval)
                        return payload

                    logger.warning(
                        "HTTP %d  game=%s (attempt %d/%d)",
                        resp.status, game_id, attempt + 1, self.max_retries,
                    )
            except asyncio.TimeoutError:
                logger.error(
                    "Timeout  game=%s (attempt %d/%d)",
                    game_id, attempt + 1, self.max_retries,
                )
            except aiohttp.ClientError as exc:
                logger.error(
                    "ClientError  game=%s (attempt %d/%d)  %s",
                    game_id, attempt + 1, self.max_retries, exc,
                )

            # 마지막 시도가 아니면 지수 백오프 후 재시도
            if attempt < self.max_retries - 1:
                backoff = self.min_interval * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(backoff)

        logger.error("game=%s: %d회 재시도 모두 실패", game_id, self.max_retries)
        await asyncio.sleep(self.min_interval)
        return None


# ────────────────────────────────────────────────────────────────────────── #
#  오케스트레이터 (Resume 지원 + skipped 로그)
# ────────────────────────────────────────────────────────────────────────── #

class PreviewCrawlerPipeline:
    """
    사용 예:
        pipeline = PreviewCrawlerPipeline(game_ids=[...], output_dir="data/preview/")
        asyncio.run(pipeline.run())
    """

    def __init__(
        self,
        game_ids: list[str],
        output_dir: str = "data/preview/",
    ):
        self.game_ids = game_ids
        self.output_dir = output_dir
        self.fetcher = PreviewAPIFetcher()
        self.skipped_log_path = os.path.join(self.output_dir, "_skipped_early.log")

        os.makedirs(self.output_dir, exist_ok=True)

    def _json_path(self, game_id: str) -> str:
        return os.path.join(self.output_dir, f"{game_id}.json")

    def _already_collected(self, game_id: str) -> bool:
        return os.path.isfile(self._json_path(game_id))

    def _load_skipped_ids(self) -> set[str]:
        if not os.path.isfile(self.skipped_log_path):
            return set()
        with open(self.skipped_log_path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _log_skipped(self, game_id: str, already_logged: set[str]) -> None:
        """2017-05-30 이전 game_id를 skipped 로그에 기록(중복 방지)."""
        if game_id in already_logged:
            return
        with open(self.skipped_log_path, "a", encoding="utf-8") as f:
            f.write(game_id + "\n")
        already_logged.add(game_id)

    async def run(self) -> None:
        collected = 0
        skipped_existing = 0
        skipped_early = 0
        failed = 0

        already_logged = self._load_skipped_ids()

        async with self.fetcher.create_session() as session:
            for game_id in tqdm(self.game_ids, desc="프리뷰 수집", unit="game"):

                if not is_preview_supported(game_id):
                    self._log_skipped(game_id, already_logged)
                    skipped_early += 1
                    continue

                if self._already_collected(game_id):
                    logger.debug("game=%s: 이미 수집됨 → 스킵", game_id)
                    skipped_existing += 1
                    continue

                payload = await self.fetcher.fetch_preview(session, game_id)
                if payload is None:
                    failed += 1
                    continue

                with open(self._json_path(game_id), "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                collected += 1

        logger.info(
            "프리뷰 크롤링 완료 | 신규: %d경기 / 스킵(기존): %d경기 / "
            "스킵(2017-05-30 이전): %d경기 / 실패: %d경기",
            collected, skipped_existing, skipped_early, failed,
        )


# ────────────────────────────────────────────────────────────────────────── #
#  엔트리포인트
# ────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    game_id_file = "hsk_game_ids_2016_2024.txt"
    with open(game_id_file, encoding="utf-8") as f:
        game_ids = [line.strip() for line in f if line.strip()]
    logger.info("대상 경기 %d개 로드 완료", len(game_ids))

    pipeline = PreviewCrawlerPipeline(game_ids=game_ids, output_dir="data/preview/")
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
