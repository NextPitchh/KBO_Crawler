"""
kbo_crawler/inning_retry.py

Timeout으로 누락된 이닝만 선별 재요청하는 복구 모듈.
기존 kbo_crawler/fetcher.py, parser.py, pipeline.py는 전혀 수정하지 않고
그대로 재사용한다 — 이 모듈은 그 위에 "누락 이닝만 재시도"하는 얇은 레이어다.

정책:
  - 누락 이닝별 최대 3회 재시도, 지수 백오프(1s, 2s, 4s 기준 + jitter)
  - 재시도 후에도 실패하면 게임 전체를 failed로 분류 → 그 게임의 CSV는
    저장/갱신하지 않는다(부분 데이터 사용 금지 — 절대 금지 규칙 준수)
  - 성공적으로 채운 경우, 전체 이닝을 처음부터 다시 파싱한다
    (PitchDataParser는 stateful하므로 순서대로 전체 재파싱해야
    recent_5_pitch_speed_avg 등 누적 파생 피처가 정확함)

실행(단독):
    uv run python -m kbo_crawler.inning_retry <pbp_dir> <game_id> [<game_id> ...]
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

import aiohttp
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.fetcher import NaverSportsAPIFetcher
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import _sort_columns
from kbo_crawler.inning_completeness import check_game_completeness

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_BACKOFF = 1.0  # seconds


async def _retry_fetch_inning(
    fetcher: NaverSportsAPIFetcher,
    session: aiohttp.ClientSession,
    game_id: str,
    inning: int,
    max_retries: int = MAX_RETRIES,
) -> Optional[dict]:
    """단일 누락 이닝을 지수 백오프로 최대 max_retries회 재시도."""
    import random

    for attempt in range(max_retries):
        raw = await fetcher.fetch_inning(session, game_id, inning)
        if raw is not None:
            relay_root = raw.get("result", {}).get("textRelayData", {})
            if relay_root.get("textRelays"):
                return raw
        backoff = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
        logger.warning(
            "game=%s inning=%d 재시도 %d/%d 실패 → %.1fs 대기",
            game_id, inning, attempt + 1, max_retries, backoff,
        )
        await asyncio.sleep(backoff)
    return None


async def recover_game(
    fetcher: NaverSportsAPIFetcher,
    session: aiohttp.ClientSession,
    game_id: str,
    missing_innings: list[int],
    existing_innings_data: list[dict],
) -> tuple[Optional[list[dict]], str]:
    """
    missing_innings를 개별 재요청하여 existing_innings_data에 병합.
    전부 성공하면 (병합된 innings_data, "recovered") 반환.
    하나라도 최종 실패하면 (None, "failed") 반환 — 부분 데이터 사용 금지.
    """
    recovered = list(existing_innings_data)
    still_missing: list[int] = []

    for inning in missing_innings:
        raw = await _retry_fetch_inning(fetcher, session, game_id, inning)
        if raw is None:
            still_missing.append(inning)
            continue
        recovered.append({"inning": inning, "data": raw})

    if still_missing:
        logger.error(
            "game=%s: 재시도 후에도 누락 이닝 %s 복구 실패 → 게임 전체 failed 처리",
            game_id, still_missing,
        )
        return None, "failed"

    recovered.sort(key=lambda x: x["inning"])
    return recovered, "recovered"


async def refetch_and_recover_game(
    fetcher: NaverSportsAPIFetcher,
    session: aiohttp.ClientSession,
    game_id: str,
) -> tuple[Optional[pd.DataFrame], str]:
    """
    게임을 처음부터 다시 크롤링하고, 완전성 체크 → 누락 시 해당 이닝만
    재시도로 복구 → 최종 DataFrame 반환. (기존 CSV의 원본 raw JSON을
    보관하지 않으므로, 전체 재크롤링 후 그 결과에 대해 gap-fill을 적용한다.)
    """
    innings_data = await fetcher.fetch_game(session, game_id)
    if not innings_data:
        return None, "failed_no_innings"

    parser = PitchDataParser()
    rows = parser.parse_game(game_id, innings_data)
    if not rows:
        return None, "failed_no_rows"

    df = pd.DataFrame(rows)
    check = check_game_completeness(df[["inning", "home_or_away"]])

    if check["status"] == "complete":
        return _sort_columns(df), "complete_no_retry_needed"

    if check["status"] != "incomplete_gap":
        # short_needs_verification 등 — 이 함수는 구조적 gap만 다룬다
        return _sort_columns(df), check["status"]

    missing = check["missing_innings"]
    logger.info("game=%s: 누락 이닝 %s 재시도 시작", game_id, missing)

    recovered_innings, status = await recover_game(
        fetcher, session, game_id, missing, innings_data
    )
    if status == "failed":
        return None, "failed"

    parser2 = PitchDataParser()
    rows2 = parser2.parse_game(game_id, recovered_innings)
    if not rows2:
        return None, "failed_no_rows_after_recovery"

    df2 = pd.DataFrame(rows2)
    recheck = check_game_completeness(df2[["inning", "home_or_away"]])
    if recheck["status"] == "incomplete_gap":
        logger.error("game=%s: 복구 후에도 gap 존재 %s → failed", game_id, recheck["missing_innings"])
        return None, "failed_still_incomplete"

    return _sort_columns(df2), "recovered"


async def recover_games_in_dir(pbp_dir: str, game_ids: list[str]) -> dict:
    """
    지정된 game_id 목록에 대해 전체 재크롤링+gap-fill을 수행하고,
    성공 시 pbp_dir의 CSV를 갱신한다. 결과 요약 dict 반환.
    """
    fetcher = NaverSportsAPIFetcher(min_delay=1.0, max_delay=1.8)
    results: dict[str, dict] = {}

    async with fetcher.create_session() as session:
        for gid in game_ids:
            before_path = os.path.join(pbp_dir, f"{gid}.csv")
            n_pa_before = None
            if os.path.isfile(before_path):
                try:
                    n_pa_before = len(pd.read_csv(before_path, usecols=["inning"], low_memory=False))
                except Exception:
                    pass

            df, status = await refetch_and_recover_game(fetcher, session, gid)

            if df is not None and status in ("recovered", "complete_no_retry_needed"):
                df.to_csv(before_path, index=False, encoding="utf-8-sig")
                n_pitches_after = len(df)
            else:
                n_pitches_after = None

            results[gid] = {
                "status": status,
                "n_pitches_before": n_pa_before,
                "n_pitches_after": n_pitches_after,
            }
            logger.info("game=%s → %s (투구 %s → %s)", gid, status, n_pa_before, n_pitches_after)

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if len(sys.argv) < 3:
        print("Usage: python -m kbo_crawler.inning_retry <pbp_dir> <game_id> [<game_id> ...]")
        sys.exit(1)

    pbp_dir = sys.argv[1]
    game_ids = sys.argv[2:]
    results = asyncio.run(recover_games_in_dir(pbp_dir, game_ids))

    print("\n" + "=" * 60)
    for gid, r in results.items():
        print(f"{gid}: {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
