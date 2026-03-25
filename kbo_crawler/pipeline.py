"""
KBODataPipeline
  - NaverSportsAPIFetcher + PitchDataParser 를 조율하는 오케스트레이터
  - 공유 aiohttp.Session 을 유지하여 Connection Pool 을 재사용
  - 경기별 개별 CSV 파티셔닝 저장 → 재시작(Resume) 지원
    저장 경로: {output_dir}/{game_id}.csv
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from .fetcher import NaverSportsAPIFetcher
from .parser import PitchDataParser

logger = logging.getLogger(__name__)

# 원하는 컬럼 순서 (전역 상수로 관리)
ORDERED_COLS: list[str] = [
    "game_id", "inning", "home_or_away",
    # 상황
    "score_diff", "out_count", "ball_count_B", "ball_count_S",
    "is_base1", "is_base2", "is_base3",
    # 프로필
    "pitcher_id", "batter_id", "batter_hit_type",
    "pitcher_vs_batter_avg", "batter_recent_avg",
    # 투구 & 피로도
    "pitch_speed", "pitch_type",
    "total_pitch_count", "recent_5_pitch_speed_avg", "inning_pitch_count",
    # 타겟 & 보상
    "pitch_result", "reward_wpa",
]


def _sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ORDERED_COLS 순서대로 컬럼을 정렬한다.
    df 에 없는 컬럼은 조용히 건너뛰고(KeyError 방어),
    예상치 못한 추가 컬럼은 뒤에 붙인다.
    """
    actual_ordered = [c for c in ORDERED_COLS if c in df.columns]
    extra_cols     = [c for c in df.columns if c not in ORDERED_COLS]
    return df[actual_ordered + extra_cols]


class KBODataPipeline:
    """
    사용 예:
        pipeline = KBODataPipeline(game_ids=[...], output_dir="data/pbp/")
        asyncio.run(pipeline.run())
    """

    def __init__(
        self,
        game_ids: list[str],
        output_dir: str = "data/pbp/",
    ):
        self.game_ids = game_ids
        self.output_dir = output_dir
        self.fetcher = NaverSportsAPIFetcher()
        self.parser = PitchDataParser()

        # 저장 디렉토리 사전 생성 (없으면 자동 생성)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  내부 유틸                                                           #
    # ------------------------------------------------------------------ #

    def _game_csv_path(self, game_id: str) -> str:
        """경기별 CSV 저장 경로 반환."""
        return os.path.join(self.output_dir, f"{game_id}.csv")

    def _already_collected(self, game_id: str) -> bool:
        """이미 CSV 가 존재하면 True (재시작 스킵 판단용)."""
        return os.path.isfile(self._game_csv_path(game_id))

    # ------------------------------------------------------------------ #
    #  단일 경기 처리 & 즉시 저장                                            #
    # ------------------------------------------------------------------ #

    async def _process_and_save_game(self, session, game_id: str) -> int:
        """
        한 경기를 크롤링 → 파싱 → 개별 CSV 저장까지 수행한다.
        저장된 투구 Row 수를 반환한다 (0 이면 데이터 없음).
        """
        innings_data = await self.fetcher.fetch_game(session, game_id)

        if not innings_data:
            logger.warning("game=%s: 수집된 이닝 없음", game_id)
            return 0

        # 게임 단위로 파생 변수 상태 초기화
        self.parser.reset()
        rows = self.parser.parse_game(game_id, innings_data)

        if not rows:
            logger.warning("game=%s: 파싱된 투구 Row 없음", game_id)
            return 0

        df = _sort_columns(pd.DataFrame(rows))

        csv_path = self._game_csv_path(game_id)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("game=%s → %d 투구 저장: %s", game_id, len(df), csv_path)

        return len(df)

    # ------------------------------------------------------------------ #
    #  전체 파이프라인 실행                                                   #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        모든 game_id 를 순차 처리한다.
        - 이미 CSV 가 존재하는 경기는 API 요청 없이 스킵 (Resume 지원)
        - Session 을 하나만 열어 Connection Pool 을 모든 경기에서 공유
        """
        total_pitches = 0
        skipped = 0

        async with self.fetcher.create_session() as session:
            for game_id in tqdm(self.game_ids, desc="경기 수집", unit="game"):

                # ── Resume 스킵 로직 ───────────────────────────────────
                if self._already_collected(game_id):
                    logger.info(
                        "game=%s: 이미 수집된 경기입니다 → 스킵", game_id
                    )
                    skipped += 1
                    continue

                count = await self._process_and_save_game(session, game_id)
                total_pitches += count

        logger.info(
            "파이프라인 완료 | 신규 수집: %d경기 / 스킵: %d경기 / 총 투구: %d",
            len(self.game_ids) - skipped, skipped, total_pitches,
        )
