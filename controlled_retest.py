"""
controlled_retest.py

Stage 0 파일럿(concurrency=5, 200경기, 96.9경기/시간)과 Stage 2 concurrency
테스트(concurrency=16, 100경기, 3,592경기/시간) 사이의 37배 성능 격차 원인을
규명하기 위한 통제 재측정.

통제 조건:
  - 대상: 파일럿에 실제 사용된 200경기 중 50경기(동일 game_id, 시드 고정 샘플)
  - 수집 범위: PBP(전 이닝) + preview 둘 다 (파일럿과 동일 범위)
  - concurrency=16 (Stage 2 최고 성능 단계와 동일)
  - 캐시 무효화: data/pbp_pilot/, data/preview_pilot/ 기존 파일은 절대 건드리지
    않고, 완전히 새 디렉토리(data/controlled_retest/)에 처음부터 다시 수집한다
    (동일 game_id라도 새 디렉토리엔 파일이 없으므로 Resume 스킵 없이 100%
    신규 네트워크 요청이 발생 — "임시 디렉토리로 이동 후 재수집"과 동일한 효과를
    기존 파일럿 데이터 훼손 위험 없이 달성)

측정: PBP 총 소요시간, preview 총 소요시간, 경기당 평균, 실제 이닝 요청 수,
      HTTP 상태코드 분포, Timeout/재시도 수.

실행:
    uv run python controlled_retest.py
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import sys
import time
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.concurrency_probe import InstrumentedFetcher
from kbo_crawler.preview_fetcher import PreviewAPIFetcher, is_preview_supported
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import _sort_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("controlled_retest")

GAME_ID_FILE = os.path.join(PROJECT_ROOT, "data", "game_ids", "controlled_retest_50.txt")
PBP_OUT = os.path.join(PROJECT_ROOT, "data", "controlled_retest", "pbp")
PREVIEW_OUT = os.path.join(PROJECT_ROOT, "data", "controlled_retest", "preview")

CONCURRENCY = 16


async def _crawl_pbp_one(fetcher, session, sem, abort_event, game_id, stats):
    async with sem:
        t0 = time.time()
        innings_data, status_counter, aborted = await fetcher.fetch_game(session, game_id, abort_event)
        elapsed = time.time() - t0

        n_requests = sum(status_counter.values())
        stats["pbp_requests_per_game"][game_id] = n_requests
        stats["pbp_status_counter"].update(status_counter)
        stats["pbp_elapsed_per_game"][game_id] = elapsed

        if not innings_data:
            stats["pbp_failed"].append(game_id)
            return

        parser = PitchDataParser()
        rows = parser.parse_game(game_id, innings_data)
        if not rows:
            stats["pbp_failed"].append(game_id)
            return

        df = _sort_columns(pd.DataFrame(rows))
        df.to_csv(os.path.join(PBP_OUT, f"{game_id}.csv"), index=False, encoding="utf-8-sig")


async def crawl_pbp_controlled(game_ids: list[str]) -> dict:
    os.makedirs(PBP_OUT, exist_ok=True)
    fetcher = InstrumentedFetcher()
    sem = asyncio.Semaphore(CONCURRENCY)
    abort_event = asyncio.Event()

    stats = {
        "pbp_requests_per_game": {}, "pbp_status_counter": Counter(),
        "pbp_elapsed_per_game": {}, "pbp_failed": [],
    }

    t0 = time.time()
    async with fetcher.create_session() as session:
        tasks = [_crawl_pbp_one(fetcher, session, sem, abort_event, gid, stats) for gid in game_ids]
        await asyncio.gather(*tasks)
    stats["pbp_total_elapsed"] = time.time() - t0
    return stats


async def crawl_preview_controlled(game_ids: list[str]) -> dict:
    os.makedirs(PREVIEW_OUT, exist_ok=True)
    fetcher = PreviewAPIFetcher(min_interval=1.0)
    sem = asyncio.Semaphore(CONCURRENCY)

    n_requests = 0
    n_skipped_early = 0
    n_failed = 0

    async def _one(session, gid):
        nonlocal n_requests, n_skipped_early, n_failed
        if not is_preview_supported(gid):
            n_skipped_early += 1
            return
        async with sem:
            n_requests += 1
            payload = await fetcher.fetch_preview(session, gid)
            if payload is None:
                n_failed += 1
                return
            with open(os.path.join(PREVIEW_OUT, f"{gid}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

    t0 = time.time()
    async with fetcher.create_session() as session:
        await asyncio.gather(*(_one(session, gid) for gid in game_ids))
    total_elapsed = time.time() - t0

    return {
        "preview_total_elapsed": total_elapsed,
        "preview_n_requests": n_requests,
        "preview_n_skipped_early": n_skipped_early,
        "preview_n_failed": n_failed,
    }


def main() -> None:
    with open(GAME_ID_FILE, encoding="utf-8") as f:
        game_ids = [line.strip() for line in f if line.strip()]
    logger.info("통제 재측정 대상 %d경기 (concurrency=%d)", len(game_ids), CONCURRENCY)

    # 사전 확인: 기존 pbp_pilot/preview_pilot 파일에 절대 손대지 않음(읽기조차 안 함)
    assert not os.path.isdir(PBP_OUT) or not os.listdir(PBP_OUT), "출력 디렉토리가 비어있지 않음 — 재측정 오염 위험"

    pbp_stats = asyncio.run(crawl_pbp_controlled(game_ids))
    preview_stats = asyncio.run(crawl_preview_controlled(game_ids))

    n = len(game_ids)
    total_pbp_requests = sum(pbp_stats["pbp_requests_per_game"].values())
    total_pbp_elapsed = pbp_stats["pbp_total_elapsed"]
    total_preview_elapsed = preview_stats["preview_total_elapsed"]
    combined_elapsed = total_pbp_elapsed + total_preview_elapsed

    result = {
        "n_games": n,
        "concurrency": CONCURRENCY,
        "pbp_total_elapsed_sec": total_pbp_elapsed,
        "pbp_avg_sec_per_game": total_pbp_elapsed / n,
        "pbp_total_requests": total_pbp_requests,
        "pbp_avg_requests_per_game": total_pbp_requests / n,
        "pbp_status_counter": dict(pbp_stats["pbp_status_counter"]),
        "pbp_failed_games": pbp_stats["pbp_failed"],
        "preview_total_elapsed_sec": total_preview_elapsed,
        "preview_n_requests": preview_stats["preview_n_requests"],
        "preview_n_skipped_early": preview_stats["preview_n_skipped_early"],
        "preview_n_failed": preview_stats["preview_n_failed"],
        "combined_elapsed_sec": combined_elapsed,
        "combined_games_per_hour": n / combined_elapsed * 3600,
        "pbp_only_games_per_hour": n / total_pbp_elapsed * 3600,
    }

    with open("data/controlled_retest/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
