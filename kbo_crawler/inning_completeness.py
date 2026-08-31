"""
kbo_crawler/inning_completeness.py

이닝 단위 Timeout으로 인한 "조용한 데이터 손실"을 탐지하는 완전성 검증기.

핵심 통찰: fetcher.py의 fetch_game()은 특정 이닝 요청이 Timeout/실패해도
루프를 중단하지 않고 다음 이닝으로 계속 진행한다(해당 이닝만 조용히 스킵).
즉, 수집된 이닝 번호 시퀀스에 "중간 구멍"이 생기면 그건 100% 크롤링 결함이지
정상적인 경기 진행일 수 없다(콜드게임도 이닝을 건너뛰지 않고, 중단된 지점까지만
연속으로 기록된다).

분류 규칙 (pitch-level DataFrame 기준, game_id 단위):
  - "incomplete_gap"           : 이닝 번호 시퀀스에 구멍이 있거나, 마지막 이닝이
                                  아닌 이닝에 top/bot 중 하나가 없음 → 명백한 크롤링 결함
  - "short_needs_verification" : 이닝 구멍은 없지만 최종 이닝이 9 미만
                                  → 우천취소/콜드게임(정상)일 수도, 이닝 전체가
                                    연속으로 Timeout난 것(결함)일 수도 있음 →
                                    schedule API ground truth 대조 필요
  - "complete"                 : 구멍 없음 + 최종 이닝 9 이상(또는 정상 종료 패턴)

기존 파이프라인 코드(kbo_crawler/fetcher.py 등)는 수정하지 않는다 — 이 모듈은
결과 CSV를 읽어 검증만 하는 독립 모듈이며, ground-truth 대조용 schedule API 호출
로직은 get_hsk_game_ids.py / game_id_collector.py의 기존 패턴을 재사용(복제)한다.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import re
from typing import Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://sports.naver.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

_STATUS_INFO_RE = re.compile(r"(\d+)회(초|말)?")


# ────────────────────────────────────────────────────────────────────────── #
#  단일 경기 구조적 완전성 체크
# ────────────────────────────────────────────────────────────────────────── #

def check_game_completeness(df: pd.DataFrame) -> dict:
    """
    pitch-level DataFrame(단일 game_id) → 완전성 판정 dict.

    필요 컬럼: inning, home_or_away
    """
    if df.empty or "inning" not in df.columns:
        return {
            "max_inning": 0, "missing_innings": [], "mid_game_half_missing": [],
            "last_inning_missing_half": [0, 1], "status": "incomplete_gap",
            "reason": "empty_or_missing_columns",
        }

    innings = sorted(int(i) for i in df["inning"].unique())
    max_inning = max(innings)
    expected = list(range(1, max_inning + 1))
    missing_innings = sorted(set(expected) - set(innings))

    half_presence = df.groupby("inning")["home_or_away"].apply(lambda s: set(int(x) for x in s.unique()))

    mid_game_half_missing: list[tuple[int, list[int]]] = []
    for inning in innings:
        if inning == max_inning:
            continue
        halves = half_presence.get(inning, set())
        missing_halves = sorted({0, 1} - halves)
        if missing_halves:
            mid_game_half_missing.append((inning, missing_halves))

    last_halves = half_presence.get(max_inning, set())
    last_missing = sorted({0, 1} - last_halves)

    structural_gap = bool(missing_innings) or bool(mid_game_half_missing)
    short_game = (max_inning < 9) and not structural_gap

    if structural_gap:
        status = "incomplete_gap"
    elif short_game:
        status = "short_needs_verification"
    else:
        status = "complete"

    return {
        "max_inning": max_inning,
        "missing_innings": missing_innings,
        "mid_game_half_missing": mid_game_half_missing,
        "last_inning_missing_half": last_missing,
        "status": status,
        "reason": "",
    }


def check_pbp_dir(pbp_dir: str) -> pd.DataFrame:
    """디렉토리 내 모든 {game_id}.csv에 완전성 체크를 적용해 요약 DataFrame 반환."""
    csv_files = sorted(glob.glob(os.path.join(pbp_dir, "*.csv")))
    rows = []
    for path in csv_files:
        game_id = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path, usecols=["inning", "home_or_away"], low_memory=False)
        except Exception as exc:
            rows.append({
                "game_id": game_id, "max_inning": 0, "missing_innings": [],
                "mid_game_half_missing": [], "last_inning_missing_half": [],
                "status": "incomplete_gap", "reason": f"csv_read_error:{exc}",
            })
            continue
        result = check_game_completeness(df)
        rows.append({"game_id": game_id, **result})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────── #
#  Ground-truth 대조 (schedule API statusInfo) — "short_needs_verification" 해소용
# ────────────────────────────────────────────────────────────────────────── #

def _parse_status_info(status_info: Optional[str]) -> Optional[tuple[int, str]]:
    """'9회말' → (9, '말'), '12회초' → (12, '초'). 파싱 불가 시 None."""
    if not status_info:
        return None
    m = _STATUS_INFO_RE.search(status_info)
    if not m:
        return None
    inning = int(m.group(1))
    half = m.group(2) or ""
    return inning, half


async def fetch_ground_truth_status(
    session: aiohttp.ClientSession, game_id: str
) -> Optional[dict]:
    """schedule API에서 해당 game_id의 실제 종료 상태(statusInfo 등)를 조회."""
    date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"
    params = {
        "fields": "basic,schedule,baseball,manualRelayUrl",
        "upperCategoryId": "kbaseball",
        "categoryId": "kbo",
        "fromDate": date_str,
        "toDate": date_str,
        "size": 500,
    }
    try:
        async with session.get(
            SCHEDULE_URL, params=params, headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except Exception as exc:
        logger.error("ground truth 조회 실패 game=%s: %s", game_id, exc)
        return None

    games = data.get("result", {}).get("games", []) or []
    for g in games:
        if g.get("gameId") == game_id:
            status_info = g.get("statusInfo")
            parsed = _parse_status_info(status_info)
            return {
                "game_id": game_id,
                "statusInfo": status_info,
                "true_final_inning": parsed[0] if parsed else None,
                "true_final_half": parsed[1] if parsed else None,
                "statusCode": g.get("statusCode"),
                "suspended": g.get("suspended"),
                "cancel": g.get("cancel"),
                "winner": g.get("winner"),
            }
    return None


async def verify_short_games(game_ids: list[str], min_delay: float = 1.0) -> pd.DataFrame:
    """final_inning < 9인 경기들의 ground truth를 조회해 콜드게임 vs 크롤링 결함을 구분."""
    rows = []
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for gid in game_ids:
            truth = await fetch_ground_truth_status(session, gid)
            rows.append({"game_id": gid, **(truth or {"truth_unavailable": True})})
            await asyncio.sleep(min_delay)
    return pd.DataFrame(rows)


def classify_short_games(short_df: pd.DataFrame, completeness_df: pd.DataFrame) -> pd.DataFrame:
    """
    ground truth(true_final_inning) vs 크롤링된 max_inning을 비교해
    'confirmed_cold_game'(정상) vs 'crawl_truncated'(결함) 분류.
    """
    merged = short_df.merge(
        completeness_df[["game_id", "max_inning"]], on="game_id", how="left"
    )

    def _classify(row) -> str:
        if row.get("truth_unavailable"):
            return "unverifiable"
        true_final = row.get("true_final_inning")
        if true_final is None:
            return "unverifiable"
        if int(true_final) <= int(row["max_inning"]):
            return "confirmed_cold_game_or_suspended"
        return "crawl_truncated"

    merged["classification"] = merged.apply(_classify, axis=1)
    return merged
