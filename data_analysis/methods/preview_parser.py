"""
data_analysis/methods/preview_parser.py

kbo_crawler/preview_fetcher.py가 저장한 data/preview/{game_id}.json 원본을
두 개의 정형 테이블로 정규화한다.

  - game_lineup.parquet  : 경기×팀 단위 (선발투수 + 그 시점 시즌 누적 스탯)
  - game_bullpen.parquet : 경기×팀×투수 단위 (등록 불펜 명단)

JSON 실제 경로: result.previewData
  - generateDate            : 통계 기준일(YYYYMMDD 문자열) — leakage 검증용
  - gameInfo.{aCode,hCode,gdate}
  - {away,home}Starter.playerInfo.{pCode,name,hitType}
  - {away,home}Starter.currentSeasonStats.{era,whip,bb,kk,hr,inn,gameCount}
  - {away,home}TeamLineUp.pitcherBullpen[].{playerCode,playerName,hitType,batsThrows}

실행:
    uv run python -m data_analysis.methods.preview_parser
"""

from __future__ import annotations

import glob
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

PREVIEW_DIR       = "data/preview"
GAME_LINEUP_OUT   = "data_analysis/results/game_lineup.parquet"
GAME_BULLPEN_OUT  = "data_analysis/results/game_bullpen.parquet"

_SIDES = (("away", False), ("home", True))


# ────────────────────────────────────────────────────────────────────────── #
#  파싱 유틸
# ────────────────────────────────────────────────────────────────────────── #

def _parse_hand(hit_type: str | None) -> str:
    """
    hitType 문자열 → 투구 손 "L"/"R"/"U"(언더).

    두 가지 스키마 형식을 모두 지원한다:
      - 선발투수: "좌투좌타"/"우투좌타" 형식 (앞 글자로 판별)
      - 불펜명단: "좌완투수"/"우완투수"/"우완언더" 형식
    판별 불가 시 빈 문자열("") 반환.
    """
    if not isinstance(hit_type, str) or not hit_type:
        return ""
    if hit_type.startswith("좌"):
        return "L"
    if "언더" in hit_type:
        return "U"
    if hit_type.startswith("우"):
        return "R"
    return ""


def _parse_player_code(code, game_id: str) -> int | None:
    """
    선수 코드(pCode/playerCode)가 숫자 문자열인지 검증 후 int로 변환.
    숫자 문자열이 아니면 경고를 남기고 None 반환.
    """
    if code is None:
        return None
    code_str = str(code)
    if not code_str.isdigit():
        logger.warning("game=%s: 비정상 playerCode=%r — 숫자 문자열 아님", game_id, code)
        return None
    return int(code_str)


def _to_float(value) -> float | None:
    """None/빈 문자열은 NaN(None)으로, 그 외는 float 변환."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ────────────────────────────────────────────────────────────────────────── #
#  단일 경기 파싱
# ────────────────────────────────────────────────────────────────────────── #

def parse_preview_json(game_id: str, data: dict) -> tuple[list[dict], list[dict]]:
    """
    단일 게임 프리뷰 JSON → (lineup_rows, bullpen_rows).

    lineup_rows  : game_lineup.parquet에 append할 행 (경기당 2행, away/home)
    bullpen_rows : game_bullpen.parquet에 append할 행 (불펜 투수당 1행)
    """
    preview = data.get("result", {}).get("previewData")
    if preview is None:
        logger.warning("game=%s: previewData 없음 — 스킵", game_id)
        return [], []

    generate_date_raw = preview.get("generateDate")
    generate_date = int(generate_date_raw) if generate_date_raw else None

    game_info = preview.get("gameInfo", {})
    gdate = int(game_info["gdate"]) if game_info.get("gdate") is not None else None

    if generate_date is not None and gdate is not None and not (generate_date < gdate):
        logger.warning(
            "game=%s: generate_date(%s) < gdate(%s) 위반 — 시점 역전 의심",
            game_id, generate_date, gdate,
        )

    lineup_rows: list[dict] = []
    bullpen_rows: list[dict] = []

    for side, is_home in _SIDES:
        team_code = game_info.get("hCode") if is_home else game_info.get("aCode")
        starter = preview.get(f"{side}Starter", {})
        team_lineup = preview.get(f"{side}TeamLineUp", {})
        bullpen = team_lineup.get("pitcherBullpen") or []

        pinfo = starter.get("playerInfo", {})
        stats = starter.get("currentSeasonStats", {})

        lineup_rows.append({
            "game_id": game_id,
            "team_code": team_code,
            "is_home": is_home,
            "generate_date": generate_date,
            "starter_id": _parse_player_code(pinfo.get("pCode"), game_id),
            "starter_name": pinfo.get("name"),
            "starter_hand": _parse_hand(pinfo.get("hitType")),
            "starter_era": _to_float(stats.get("era")),
            "starter_whip": _to_float(stats.get("whip")),
            "starter_bb": _to_float(stats.get("bb")),
            "starter_kk": _to_float(stats.get("kk")),
            "starter_hr": _to_float(stats.get("hr")),
            "starter_inn": _to_float(stats.get("inn")),
            "starter_games": _to_float(stats.get("gameCount")),
            "bullpen_size": len(bullpen),
        })

        if not bullpen:
            logger.info("game=%s side=%s: pitcherBullpen 비어있음", game_id, side)

        for entry in bullpen:
            bullpen_rows.append({
                "game_id": game_id,
                "team_code": team_code,
                "is_home": is_home,
                "pitcher_id": _parse_player_code(entry.get("playerCode"), game_id),
                "pitcher_name": entry.get("playerName"),
                "throws": _parse_hand(entry.get("hitType")),
                "is_listed_bullpen": True,
            })

    return lineup_rows, bullpen_rows


# ────────────────────────────────────────────────────────────────────────── #
#  배치 빌드
# ────────────────────────────────────────────────────────────────────────── #

def build_lineup_tables(
    preview_dir: str = PREVIEW_DIR,
    lineup_out: str = GAME_LINEUP_OUT,
    bullpen_out: str = GAME_BULLPEN_OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    data/preview/*.json 전체를 파싱하여 game_lineup.parquet / game_bullpen.parquet
    으로 저장하고 두 DataFrame을 반환한다.
    """
    json_files = sorted(glob.glob(os.path.join(preview_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"프리뷰 JSON 없음: {preview_dir}")

    logger.info("프리뷰 JSON %d개 로드 중...", len(json_files))

    all_lineup_rows: list[dict] = []
    all_bullpen_rows: list[dict] = []
    empty_bullpen_games = 0
    parse_fail = 0

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

        n_empty = sum(1 for r in lineup_rows if r["bullpen_size"] == 0)
        empty_bullpen_games += 1 if n_empty > 0 else 0

        all_lineup_rows.extend(lineup_rows)
        all_bullpen_rows.extend(bullpen_rows)

    lineup_df = pd.DataFrame(all_lineup_rows)
    bullpen_df = pd.DataFrame(all_bullpen_rows)

    logger.info(
        "파싱 완료 | game_lineup: %d행 / game_bullpen: %d행 / "
        "파싱 실패 경기: %d / pitcherBullpen 비어있는 경기: %d",
        len(lineup_df), len(bullpen_df), parse_fail, empty_bullpen_games,
    )

    os.makedirs(os.path.dirname(lineup_out), exist_ok=True)
    lineup_df.to_parquet(lineup_out, index=False)
    bullpen_df.to_parquet(bullpen_out, index=False)
    logger.info("저장 완료: %s / %s", lineup_out, bullpen_out)

    return lineup_df, bullpen_df


# ────────────────────────────────────────────────────────────────────────── #
#  엔트리포인트
# ────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    lineup_df, bullpen_df = build_lineup_tables()

    print("\n" + "=" * 60)
    print(f"[game_lineup] {len(lineup_df):,} 행 (경기 {lineup_df['game_id'].nunique():,}개)")
    print(f"[game_bullpen] {len(bullpen_df):,} 행")
    print(f"\n[starter_hand 분포]\n{lineup_df['starter_hand'].value_counts().to_string()}")
    print(f"\n[bullpen throws 분포]\n{bullpen_df['throws'].value_counts().to_string()}")
    n_bad_starter = lineup_df["starter_id"].isna().sum()
    n_bad_bullpen = bullpen_df["pitcher_id"].isna().sum()
    print(f"\nplayerCode 파싱 실패: starter {n_bad_starter}건 / bullpen {n_bad_bullpen}건")
    print("=" * 60)
