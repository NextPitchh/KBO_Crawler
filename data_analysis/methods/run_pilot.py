"""
data_analysis/methods/run_pilot.py

파일럿 200경기(전 구단·전 연도) 검증 오케스트레이터.

기존 파이프라인 코드(kbo_crawler/*, data_analysis/methods/*)는 전혀 수정하지
않고, 이미 파라미터화되어 있는 함수/클래스를 파일럿 전용 경로로 호출한다.
game_id 목록만 hsk 153경기 → pilot_games.txt(200경기)로 교체.

산출 경로 (기존 153경기 산출물과 완전히 분리):
  data/pbp_pilot/{game_id}.csv
  data/preview_pilot/{game_id}.json
  data_analysis/results/pilot/pilot_pa.parquet
  data_analysis/results/pilot/pilot_pa_with_states.parquet
  data_analysis/results/pilot/pilot_pa_with_wpa.parquet
  data_analysis/results/pilot/pilot_game_lineup.parquet
  data_analysis/results/pilot/pilot_game_bullpen.parquet
  data_analysis/results/pilot/pilot_pitcher_appearances.parquet
  data_analysis/results/pilot/pilot_pitcher_history.parquet
  data_analysis/results/pilot/pilot_league_baseline.json
  data_analysis/results/pilot/pilot_pa_bullpen_state.parquet
  data_analysis/results/pilot/pilot_hsk_pa_enriched.parquet
  data_analysis/results/pilot/pilot_enrichment_report.md
  data_analysis/results/pilot/pilot_failures.json   (단계별 실패 game_id + 원인)

실행:
    uv run python -m data_analysis.methods.run_pilot
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import sys

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.fetcher import NaverSportsAPIFetcher
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import _sort_columns
from kbo_crawler.preview_fetcher import PreviewCrawlerPipeline, is_preview_supported

from data_analysis.methods.pa_aggregator import aggregate_pa
from data_analysis.methods.state_transition import build_state_transitions
from data_analysis.methods.inject_wpa import inject_computed_wpa
from data_analysis.methods.preview_parser import build_lineup_tables
from data_analysis.methods.appearance_aggregator import build_appearances
from data_analysis.methods.pitcher_history import build_pitcher_history
from data_analysis.methods.bullpen_state import build_bullpen_state
from data_analysis.methods.build_enriched_dataset import build_enriched_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pilot")

# ── 경로 상수 ────────────────────────────────────────────────────────────
GAME_ID_FILE = os.path.join(PROJECT_ROOT, "data", "game_ids", "pilot_games.txt")

PBP_DIR      = os.path.join(PROJECT_ROOT, "data", "pbp_pilot")
PREVIEW_DIR  = os.path.join(PROJECT_ROOT, "data", "preview_pilot")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "data_analysis", "results", "pilot")

PA_PATH             = os.path.join(RESULTS_DIR, "pilot_pa.parquet")
PA_STATES_PATH       = os.path.join(RESULTS_DIR, "pilot_pa_with_states.parquet")
PA_WPA_PATH          = os.path.join(RESULTS_DIR, "pilot_pa_with_wpa.parquet")
LINEUP_PATH          = os.path.join(RESULTS_DIR, "pilot_game_lineup.parquet")
BULLPEN_LISTED_PATH  = os.path.join(RESULTS_DIR, "pilot_game_bullpen.parquet")
APPEARANCES_PATH     = os.path.join(RESULTS_DIR, "pilot_pitcher_appearances.parquet")
PITCHER_HISTORY_PATH = os.path.join(RESULTS_DIR, "pilot_pitcher_history.parquet")
LEAGUE_BASELINE_PATH = os.path.join(RESULTS_DIR, "pilot_league_baseline.json")
BULLPEN_STATE_PATH   = os.path.join(RESULTS_DIR, "pilot_pa_bullpen_state.parquet")
ENRICHED_PATH        = os.path.join(RESULTS_DIR, "pilot_hsk_pa_enriched.parquet")
ENRICH_REPORT_PATH   = os.path.join(RESULTS_DIR, "pilot_enrichment_report.md")
FAILURES_PATH        = os.path.join(RESULTS_DIR, "pilot_failures.json")

PBP_CONCURRENCY = 5
MIN_DELAY = 1.0   # 요청 간격 절대 최소 1초 (절대 금지 규칙 준수)
MAX_DELAY = 1.8


# ────────────────────────────────────────────────────────────────────────
#  Step 1: PBP 크롤링 (실패 game_id + 원인 기록)
# ────────────────────────────────────────────────────────────────────────

async def _process_one_pbp(session, fetcher, parser_cls, sem, game_id: str, failures: dict) -> int:
    csv_path = os.path.join(PBP_DIR, f"{game_id}.csv")
    if os.path.isfile(csv_path):
        return 0  # 이미 수집됨 — 실패 아님, 스킵

    async with sem:
        try:
            innings_data = await fetcher.fetch_game(session, game_id)
            if not innings_data:
                failures[game_id] = "no_innings_returned"
                return 0

            parser = parser_cls()
            rows = parser.parse_game(game_id, innings_data)
            if not rows:
                failures[game_id] = "parsed_zero_rows"
                return 0

            df = _sort_columns(pd.DataFrame(rows))
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return len(df)

        except Exception as exc:
            failures[game_id] = f"exception:{type(exc).__name__}:{str(exc)[:200]}"
            logger.error("game=%s PBP 크롤링 실패: %s", game_id, exc, exc_info=True)
            return 0


async def crawl_pbp(game_ids: list[str]) -> dict:
    os.makedirs(PBP_DIR, exist_ok=True)
    sem = asyncio.Semaphore(PBP_CONCURRENCY)
    fetcher = NaverSportsAPIFetcher(min_delay=MIN_DELAY, max_delay=MAX_DELAY)
    failures: dict[str, str] = {}

    async with fetcher.create_session() as session:
        tasks = [
            _process_one_pbp(session, fetcher, PitchDataParser, sem, gid, failures)
            for gid in game_ids
        ]
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="PBP 크롤링", unit="game"):
            await coro

    logger.info("PBP 크롤링 완료 | 실패 %d경기", len(failures))
    return failures


def aggregate_pbp_to_pa() -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(PBP_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"CSV 없음: {PBP_DIR}")

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    pitch_df = pd.concat(dfs, ignore_index=True)
    logger.info("pitch-level %d행(%d경기) → PA 집계", len(pitch_df), len(csv_files))

    pa_df = aggregate_pa(pitch_df)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pa_df.to_parquet(PA_PATH, index=False)
    logger.info("저장 완료: %s (%d PA)", PA_PATH, len(pa_df))
    return pa_df


# ────────────────────────────────────────────────────────────────────────
#  Step 2: preview 크롤링 (실패 game_id + 원인 기록)
# ────────────────────────────────────────────────────────────────────────

async def crawl_preview(game_ids: list[str]) -> dict:
    pipeline = PreviewCrawlerPipeline(game_ids=game_ids, output_dir=PREVIEW_DIR)
    pipeline.fetcher.min_interval = MIN_DELAY
    await pipeline.run()

    failures: dict[str, str] = {}
    for gid in game_ids:
        json_path = os.path.join(PREVIEW_DIR, f"{gid}.json")
        if os.path.isfile(json_path):
            continue
        if not is_preview_supported(gid):
            continue  # 2017-05-30 이전 — 설계상 정상 스킵, 실패 아님
        failures[gid] = "preview_fetch_failed_after_retries"

    logger.info("preview 크롤링 완료 | 실패 %d경기", len(failures))
    return failures


# ────────────────────────────────────────────────────────────────────────
#  메인
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(GAME_ID_FILE, encoding="utf-8") as f:
        game_ids = [line.strip() for line in f if line.strip()]
    logger.info("파일럿 대상 %d경기 로드", len(game_ids))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_failures: dict[str, dict] = {}

    # ── 1) PBP 크롤링 + 집계 ────────────────────────────────────────────
    pbp_failures = asyncio.run(crawl_pbp(game_ids))
    all_failures["pbp_crawl"] = pbp_failures
    aggregate_pbp_to_pa()

    # ── 2) WPA 계산 ──────────────────────────────────────────────────────
    build_state_transitions(input_path=PA_PATH, output_path=PA_STATES_PATH)
    inject_computed_wpa(input_path=PA_STATES_PATH, output_path=PA_WPA_PATH)

    # ── 3) preview 크롤링 + 파싱 ─────────────────────────────────────────
    preview_failures = asyncio.run(crawl_preview(game_ids))
    all_failures["preview_crawl"] = preview_failures
    build_lineup_tables(
        preview_dir=PREVIEW_DIR,
        lineup_out=LINEUP_PATH,
        bullpen_out=BULLPEN_LISTED_PATH,
    )

    # ── 4) 등판 집계 + pitcher_history ────────────────────────────────────
    build_appearances(input_path=PA_WPA_PATH, output_path=APPEARANCES_PATH)
    build_pitcher_history(
        input_path=APPEARANCES_PATH,
        output_path=PITCHER_HISTORY_PATH,
        baseline_path=LEAGUE_BASELINE_PATH,
    )
    build_bullpen_state(
        pa_path=PA_WPA_PATH,
        bullpen_path=BULLPEN_LISTED_PATH,
        output_path=BULLPEN_STATE_PATH,
    )

    # ── 5) enriched 통합 ─────────────────────────────────────────────────
    try:
        build_enriched_dataset(
            pa_path=PA_WPA_PATH,
            hist_path=PITCHER_HISTORY_PATH,
            bullpen_state_path=BULLPEN_STATE_PATH,
            game_bullpen_path=BULLPEN_LISTED_PATH,
            game_lineup_path=LINEUP_PATH,
            output_path=ENRICHED_PATH,
            report_path=ENRICH_REPORT_PATH,
        )
        all_failures["enrichment"] = "OK"
    except Exception as exc:
        all_failures["enrichment"] = f"FAILED: {exc}"
        logger.error("enriched 통합 실패: %s", exc, exc_info=True)

    # ── 실패 로그 저장 ───────────────────────────────────────────────────
    with open(FAILURES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_failures, f, ensure_ascii=False, indent=2)
    logger.info("실패 로그 저장: %s", FAILURES_PATH)

    print("\n" + "=" * 60)
    print(f"[파일럿 파이프라인 완료] {len(game_ids)}경기 대상")
    print(f"PBP 크롤링 실패: {len(pbp_failures)}경기")
    print(f"preview 크롤링 실패: {len(preview_failures)}경기")
    print(f"enrichment: {all_failures['enrichment']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
