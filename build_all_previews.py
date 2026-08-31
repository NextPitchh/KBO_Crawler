"""
build_all_previews.py

data/preview_full/{year}/*.json (10개 연도) 전체를 preview_parser.py의
parse_preview_json()으로 파싱해 통합 lineup/bullpen 테이블을 만든다.
기존 preview_parser.py는 단일 디렉토리만 처리하므로(연도별 서브디렉토리
구조와 안 맞음) 이 스크립트는 그 함수를 그대로 재사용하되 연도 루프만
새로 짠다 — 파싱 로직 자체는 수정하지 않는다.

산출: data_analysis/results/all_game_lineup.parquet
      data_analysis/results/all_game_bullpen.parquet

실행:
    uv run python build_all_previews.py
"""

from __future__ import annotations

import glob
import json
import logging
import os

import pandas as pd

from data_analysis.methods.preview_parser import parse_preview_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

YEARS = list(range(2016, 2026))
PREVIEW_DIR_TMPL = "data/preview_full/{year}"
LINEUP_OUT = "data_analysis/results/all_game_lineup.parquet"
BULLPEN_OUT = "data_analysis/results/all_game_bullpen.parquet"


def main() -> None:
    all_lineup_rows: list[dict] = []
    all_bullpen_rows: list[dict] = []
    parse_fail = 0
    per_year_counts: dict[int, int] = {}

    for year in YEARS:
        preview_dir = PREVIEW_DIR_TMPL.format(year=year)
        json_files = sorted(glob.glob(os.path.join(preview_dir, "*.json")))
        per_year_counts[year] = len(json_files)

        for path in json_files:
            game_id = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("game=%s: JSON 로드 실패 %s", game_id, exc)
                parse_fail += 1
                continue

            lineup_rows, bullpen_rows = parse_preview_json(game_id, data)
            if not lineup_rows:
                parse_fail += 1
                continue

            all_lineup_rows.extend(lineup_rows)
            all_bullpen_rows.extend(bullpen_rows)

        logger.info("%d년: %d개 JSON 처리", year, len(json_files))

    lineup_df = pd.DataFrame(all_lineup_rows)
    bullpen_df = pd.DataFrame(all_bullpen_rows)

    logger.info(
        "파싱 완료 | game_lineup: %d행 / game_bullpen: %d행 / 파싱 실패: %d",
        len(lineup_df), len(bullpen_df), parse_fail,
    )

    os.makedirs(os.path.dirname(LINEUP_OUT), exist_ok=True)
    lineup_df.to_parquet(LINEUP_OUT, index=False)
    bullpen_df.to_parquet(BULLPEN_OUT, index=False)
    logger.info("저장 완료: %s / %s", LINEUP_OUT, BULLPEN_OUT)

    # ── 검증 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[전체] game_lineup: {len(lineup_df):,}행 (경기 {lineup_df['game_id'].nunique():,}개)")
    print(f"[전체] game_bullpen: {len(bullpen_df):,}행")

    # generate_date < 경기일 검증
    lineup_df["_gdate"] = lineup_df["game_id"].str[:8].astype(int)
    violation = lineup_df[
        lineup_df["generate_date"].notna()
        & (lineup_df["generate_date"] >= lineup_df["_gdate"])
    ]
    print(f"\n[검증] generate_date < 경기일 위반: {len(violation)}건 "
          f"({'PASS' if len(violation)==0 else 'FAIL'})")

    # 2017-05-30 이전 경기 존재 여부
    pre_threshold = lineup_df[lineup_df["_gdate"] < 20170530]
    print(f"[검증] 2017-05-30 이전 경기 존재: {pre_threshold['game_id'].nunique()}개 "
          f"({'PASS(0건)' if pre_threshold['game_id'].nunique()==0 else 'FAIL — 크롤링 오류 의심'})")

    print(f"\n[연도별 JSON 파일 수]")
    for y, c in per_year_counts.items():
        print(f"  {y}: {c}")

    print(f"\n[팀별 커버리지] (lineup 기준 team_code 등장 횟수)")
    print(lineup_df["team_code"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
