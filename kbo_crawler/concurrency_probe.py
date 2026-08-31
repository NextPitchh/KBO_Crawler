"""
kbo_crawler/concurrency_probe.py

Concurrency 안전 상한을 실측하기 위한 계단식 부하 테스트 도구.
기존 kbo_crawler/fetcher.py는 수정하지 않고, HTTP 상태 코드(429/5xx 구분)를
직접 관측해야 하므로 별도의 계측 fetcher를 이 모듈 안에 둔다(URL/HEADERS
상수는 fetcher.py에서 그대로 재사용).

안전장치: HTTP 429가 단 1건이라도 감지되면 asyncio.Event로 즉시 전역 중단
신호를 보내 진행 중인 모든 코루틴이 추가 요청을 멈춘다(이미 발사된 요청은
자연 완료되지만 새 요청은 발사하지 않음).

실행:
    uv run python -m kbo_crawler.concurrency_probe <stage_name> <concurrency> <game_id_file> <output_pbp_dir>
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import random
import sys
import time
from collections import Counter

import aiohttp
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.fetcher import NaverSportsAPIFetcher
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import _sort_columns
from kbo_crawler.inning_completeness import check_game_completeness

logger = logging.getLogger(__name__)

MAX_INNINGS = 12
MIN_DELAY = 1.0
MAX_DELAY = 1.8


class InstrumentedFetcher:
    """
    NaverSportsAPIFetcher와 동일한 URL/HEADERS를 쓰되, HTTP 상태 코드를
    세밀하게(200/429/5xx/기타4xx/timeout/clienterror) 분류해 반환한다.
    """

    BASE_URL = NaverSportsAPIFetcher.BASE_URL
    HEADERS = NaverSportsAPIFetcher.HEADERS

    def __init__(self, min_delay: float = MIN_DELAY, max_delay: float = MAX_DELAY):
        self.min_delay = min_delay
        self.max_delay = max_delay

    def create_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(ssl=False, limit=32)
        return aiohttp.ClientSession(connector=connector, headers=self.HEADERS)

    async def _random_delay(self) -> None:
        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))

    async def fetch_inning(
        self, session: aiohttp.ClientSession, game_id: str, inning: int,
    ) -> tuple[str, dict | None]:
        url = self.BASE_URL.format(game_id=game_id)
        params = {"inning": inning}
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    payload = await resp.json(content_type=None)
                    return "200", payload
                if resp.status == 429:
                    return "429", None
                if 500 <= resp.status < 600:
                    return "5xx", None
                return "4xx_other", None
        except asyncio.TimeoutError:
            return "timeout", None
        except aiohttp.ClientError:
            return "clienterror", None

    async def fetch_game(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
        abort_event: asyncio.Event,
    ) -> tuple[list[dict], Counter, bool]:
        """
        (innings_data, status_counter, aborted) 반환.
        abort_event가 set되면 더 이상 새 요청을 발사하지 않고 즉시 반환.
        """
        results: list[dict] = []
        status_counter: Counter = Counter()

        for inning in range(1, MAX_INNINGS + 1):
            if abort_event.is_set():
                return results, status_counter, True

            status, raw = await self.fetch_inning(session, game_id, inning)
            status_counter[status] += 1

            if status == "429":
                logger.error("game=%s inning=%d → HTTP 429 감지! 전역 중단 신호 전송", game_id, inning)
                abort_event.set()
                return results, status_counter, True

            if status != "200" or raw is None:
                await self._random_delay()
                continue

            relay_root = raw.get("result", {}).get("textRelayData", {})
            relays = relay_root.get("textRelays", [])
            if not relays and inning > 1:
                break

            results.append({"inning": inning, "data": raw})
            await self._random_delay()

        return results, status_counter, False


async def _process_one(
    fetcher: InstrumentedFetcher,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    abort_event: asyncio.Event,
    game_id: str,
    output_dir: str,
) -> dict:
    async with sem:
        if abort_event.is_set():
            return {"game_id": game_id, "outcome": "aborted", "status_counter": Counter()}

        t0 = time.time()
        innings_data, status_counter, aborted = await fetcher.fetch_game(session, game_id, abort_event)
        elapsed = time.time() - t0

        if aborted:
            return {"game_id": game_id, "outcome": "aborted", "status_counter": status_counter, "elapsed": elapsed}

        if not innings_data:
            return {"game_id": game_id, "outcome": "no_data", "status_counter": status_counter, "elapsed": elapsed}

        parser = PitchDataParser()
        rows = parser.parse_game(game_id, innings_data)
        if not rows:
            return {"game_id": game_id, "outcome": "no_rows", "status_counter": status_counter, "elapsed": elapsed}

        df = _sort_columns(pd.DataFrame(rows))
        csv_path = os.path.join(output_dir, f"{game_id}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        completeness = check_game_completeness(df[["inning", "home_or_away"]])

        return {
            "game_id": game_id, "outcome": "ok", "status_counter": status_counter,
            "elapsed": elapsed, "n_pitches": len(df),
            "completeness_status": completeness["status"],
        }


async def run_stage(
    game_ids: list[str],
    concurrency: int,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    fetcher = InstrumentedFetcher()
    sem = asyncio.Semaphore(concurrency)
    abort_event = asyncio.Event()

    t0 = time.time()
    async with fetcher.create_session() as session:
        tasks = [
            _process_one(fetcher, session, sem, abort_event, gid, output_dir)
            for gid in game_ids
        ]
        results = await asyncio.gather(*tasks)
    total_elapsed = time.time() - t0

    total_status_counter: Counter = Counter()
    for r in results:
        total_status_counter.update(r.get("status_counter", Counter()))

    n_total = len(results)
    n_ok = sum(1 for r in results if r["outcome"] == "ok")
    n_aborted = sum(1 for r in results if r["outcome"] == "aborted")
    n_no_data = sum(1 for r in results if r["outcome"] in ("no_data", "no_rows"))
    n_incomplete = sum(1 for r in results if r.get("completeness_status") == "incomplete_gap")

    n_429 = total_status_counter.get("429", 0)
    n_5xx = total_status_counter.get("5xx", 0)
    n_timeout = total_status_counter.get("timeout", 0)
    n_clienterror = total_status_counter.get("clienterror", 0)

    failure_rate = (n_no_data + n_incomplete) / n_total if n_total else 0.0
    inning_missing_rate = n_incomplete / n_total if n_total else 0.0

    return {
        "concurrency": concurrency,
        "n_games": n_total,
        "n_ok": n_ok,
        "n_aborted": n_aborted,
        "n_no_data": n_no_data,
        "n_incomplete_gap": n_incomplete,
        "n_http_429": n_429,
        "n_http_5xx": n_5xx,
        "n_timeout": n_timeout,
        "n_clienterror": n_clienterror,
        "failure_rate": failure_rate,
        "inning_missing_rate": inning_missing_rate,
        "total_elapsed_sec": total_elapsed,
        "avg_sec_per_game": total_elapsed / n_total if n_total else 0.0,
        "aborted_due_to_429": n_429 > 0,
        "raw_results": results,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if len(sys.argv) != 5:
        print("Usage: python -m kbo_crawler.concurrency_probe <stage_name> <concurrency> <game_id_file> <output_pbp_dir>")
        sys.exit(1)

    stage_name, concurrency_str, game_id_file, output_dir = sys.argv[1:5]
    concurrency = int(concurrency_str)

    with open(game_id_file, encoding="utf-8") as f:
        game_ids = [line.strip() for line in f if line.strip()]

    logger.info("[%s] concurrency=%d, 대상 %d경기", stage_name, concurrency, len(game_ids))
    result = asyncio.run(run_stage(game_ids, concurrency, output_dir))
    result.pop("raw_results")

    print("\n" + "=" * 60)
    print(f"[{stage_name}] 결과")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
