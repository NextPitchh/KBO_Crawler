"""
KBO 문자중계 Pitch-by-Pitch 크롤러 — 진입점

# ── 프로젝트 세팅 (최초 1회) ─────────────────────────────────────────────
# uv init kbo-catboost
# cd kbo-catboost
# uv add aiohttp pandas tqdm

# ── 실행 ──────────────────────────────────────────────────────────────────
# uv run python main.py
"""

import asyncio
import logging
import sys

from kbo_crawler.pipeline import KBODataPipeline

# ── Windows asyncio + aiohttp 고질적 에러 방지 ───────────────────────────
# aiohttp 세션 종료 시 "RuntimeError: Event loop is closed" 방어
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── 로깅 설정 ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── 수집 대상 경기 ID (테스트용 하드코딩) ────────────────────────────────
# 실제 운영 시 외부 파일이나 DB 에서 로드하도록 확장 가능
GAME_IDS: list[str] = [
    "20150311SKHH0",
]

OUTPUT_DIR = "data/pbp/"


async def main() -> None:
    pipeline = KBODataPipeline(game_ids=GAME_IDS, output_dir=OUTPUT_DIR)
    await pipeline.run()

    print(f"\n수집 완료 — 저장 위치: {OUTPUT_DIR}")
    print("각 경기 데이터는 {game_id}.csv 로 개별 저장됩니다.")


if __name__ == "__main__":
    asyncio.run(main())
