"""
HSK 데이터 병렬 크롤링 & PA 집계 오케스트레이터
  - hsk_game_ids_2016_2024.txt(153경기) 전체를 asyncio.Semaphore(5)로 병렬 크롤링
  - 경기별 pitch-level CSV → data_analysis/results/pbp/{game_id}.csv
  - 전체 집계 후 PA-level Parquet → data_analysis/results/hsk_pa.parquet

실행:
    uv run python data_analysis/methods/run_hsk.py
"""

import asyncio
import glob
import logging
import os
import sys

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

# ── 프로젝트 루트를 sys.path에 추가 (kbo_crawler 패키지 임포트용) ─────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METHODS_DIR  = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, METHODS_DIR)

from kbo_crawler.fetcher import NaverSportsAPIFetcher
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import ORDERED_COLS, _sort_columns
from pa_aggregator import aggregate_pa

# ── 경로 상수 ────────────────────────────────────────────────────────────
GAME_ID_FILE = os.path.join(PROJECT_ROOT, "hsk_game_ids_2016_2024.txt")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data_analysis", "results", "pbp")
PA_OUTPUT    = os.path.join(PROJECT_ROOT, "data_analysis", "results", "hsk_pa.parquet")

CONCURRENCY  = 5   # 동시 경기 수 (Rate Limit 안전선)

# ── 로깅 설정 ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_hsk")


# ────────────────────────────────────────────────────────────────────────
#  단일 경기 처리
# ────────────────────────────────────────────────────────────────────────

async def process_one(
    session,
    fetcher: NaverSportsAPIFetcher,
    sem: asyncio.Semaphore,
    game_id: str,
) -> int:
    """
    한 경기를 크롤링 → 파싱 → CSV 저장.
    이미 CSV 존재 시 스킵(Resume). 저장된 투구 수를 반환.
    """
    csv_path = os.path.join(OUTPUT_DIR, f"{game_id}.csv")
    if os.path.isfile(csv_path):
        logger.debug("SKIP %s (이미 존재)", game_id)
        return 0

    async with sem:
        try:
            parser = PitchDataParser()   # 경기당 독립 인스턴스 (상태 격리)
            innings_data = await fetcher.fetch_game(session, game_id)

            if not innings_data:
                logger.warning("game=%s: 수집된 이닝 없음", game_id)
                return 0

            rows = parser.parse_game(game_id, innings_data)
            if not rows:
                logger.warning("game=%s: 파싱된 투구 Row 없음", game_id)
                return 0

            df = _sort_columns(pd.DataFrame(rows))
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info("game=%s → %d투구 저장", game_id, len(df))
            return len(df)

        except Exception as exc:
            logger.error("game=%s 처리 실패: %s", game_id, exc, exc_info=True)
            return 0


# ────────────────────────────────────────────────────────────────────────
#  전체 병렬 크롤링
# ────────────────────────────────────────────────────────────────────────

async def crawl_all(game_ids: list[str]) -> int:
    """모든 경기를 Semaphore(CONCURRENCY)로 병렬 크롤링. 총 신규 투구 수 반환."""
    sem = asyncio.Semaphore(CONCURRENCY)
    fetcher = NaverSportsAPIFetcher()

    async with fetcher.create_session() as session:
        tasks = [
            process_one(session, fetcher, sem, gid)
            for gid in game_ids
        ]
        results = []
        for coro in async_tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="크롤링",
            unit="game",
        ):
            cnt = await coro
            results.append(cnt)

    return sum(results)


# ────────────────────────────────────────────────────────────────────────
#  PA 집계 및 Parquet 저장
# ────────────────────────────────────────────────────────────────────────

def aggregate_all_csvs() -> pd.DataFrame:
    """results/pbp/*.csv → concat → aggregate_pa() → hsk_pa.parquet."""
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"CSV 없음: {OUTPUT_DIR}")

    logger.info("CSV %d개 로드 중...", len(csv_files))
    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            logger.warning("CSV 로드 실패 %s: %s", f, exc)

    pitch_df = pd.concat(dfs, ignore_index=True)
    logger.info("pitch-level 총 %d행 → PA 집계 시작", len(pitch_df))

    pa_df = aggregate_pa(pitch_df)
    logger.info("PA-level 총 %d타석", len(pa_df))

    pa_df.to_parquet(PA_OUTPUT, index=False)
    logger.info("저장 완료: %s", PA_OUTPUT)
    return pa_df


# ────────────────────────────────────────────────────────────────────────
#  엔트리포인트
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 게임 ID 로드
    with open(GAME_ID_FILE, encoding="utf-8") as f:
        game_ids = [line.strip() for line in f if line.strip()]
    logger.info("대상 경기 %d개 로드 완료", len(game_ids))

    # 크롤링
    total_pitches = asyncio.run(crawl_all(game_ids))
    logger.info("크롤링 완료 — 신규 수집 투구: %d", total_pitches)

    # PA 집계
    pa_df = aggregate_all_csvs()

    # 간단 요약
    print("\n" + "=" * 60)
    print(f"[완료] PA 타석 수: {len(pa_df):,}")
    if "pa_result" in pa_df.columns:
        print("\n[pa_result 분포]")
        print(pa_df["pa_result"].value_counts().to_string())
    if "pitches_per_pa" in pa_df.columns:
        print(f"\n[pitches_per_pa] 평균: {pa_df['pitches_per_pa'].mean():.2f} "
              f"/ 최대: {pa_df['pitches_per_pa'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
