"""
data_analysis/methods/run_full_year.py

전체 확장(7,200경기) 실행 계획서(expansion_plan.md v2)에 따른 연도 단위
크롤링+처리 오케스트레이터. 기존 kbo_crawler/*, data_analysis/methods/*
모듈은 수정하지 않고 그대로 재사용한다.

한 연도 처리 흐름:
  1) all_games_{year}.txt 로드 → game_index.csv 기준 is_regular_season=True,
     game_status in {normal, called} 만 필터 (suspended/cancelled 제외)
  2) PBP 크롤링 (concurrency=16, 429 감지 시 즉시 중단 후 12→8로 하향 재시작,
     이닝 gap 발견 시 최대 3회 재시도, 재시도 후에도 실패면 CSV 미저장)
  3) preview 크롤링 (2017-05-30 이후만)
  4) PA 집계 → 상태 전이 → WPA 계산 → called 게임 WE 보정
  5) staging 저장: data_analysis/results/staging/pa_states_{year}.parquet
  6) 체크포인트 7항목 계산 → checkpoint_{year}.json

실행:
    uv run python -m data_analysis.methods.run_full_year <year>
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import random
import sys
import time
from collections import Counter

import aiohttp
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from kbo_crawler.concurrency_probe import InstrumentedFetcher
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import _sort_columns
from kbo_crawler.inning_completeness import check_game_completeness, check_pbp_dir
from kbo_crawler.preview_fetcher import PreviewAPIFetcher, is_preview_supported

from data_analysis.methods.pa_aggregator import aggregate_pa
from data_analysis.methods.state_transition import build_state_transitions
from data_analysis.methods.inject_wpa import inject_computed_wpa
from data_analysis.methods.terminal_pa_correction import load_final_scores, apply_terminal_pa_correction
from data_analysis.methods.extra_innings_policy import identify_extra_innings_games, apply_extra_innings_policy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_full_year")

GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "data", "game_ids")
GAME_INDEX_PATH = os.path.join(GAME_IDS_DIR, "game_index.csv")
EXCLUDED_PATH = os.path.join(GAME_IDS_DIR, "excluded_games.csv")
FINAL_SCORES_PATH = os.path.join(GAME_IDS_DIR, "final_scores.csv")

PBP_DIR_TMPL = os.path.join(PROJECT_ROOT, "data", "pbp_full", "{year}")
PREVIEW_DIR_TMPL = os.path.join(PROJECT_ROOT, "data", "preview_full", "{year}")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data_analysis", "results", "staging")

CONCURRENCY_LADDER = [16, 12, 8]
MAX_INNING_RETRIES = 3
BASE_BACKOFF = 1.0
MIN_DELAY = 1.0
MAX_DELAY = 1.8

EXPECTED_ORDER = ["HR", "3B", "2B", "1B", "BB", "SF", "OUT", "SO", "GDP"]
HIT_CATEGORIES = ["1B", "2B", "3B", "HR"]
OUT_CATEGORIES = ["OUT", "SO", "GDP"]
SEM_TOLERANCE = 2.0  # Tier 2 경고 허용 범위: 차이가 2×SEM 이내

# 인접 카테고리 역전을 Tier 2(경고)로 허용하는 쌍과 그 조건.
# 조건 함수는 ns(카테고리별 표본수 dict)를 받아 True/False 반환 —
# 조건을 만족해야만 fragile pair로 취급(그 외엔 Tier 1 유지).
#   (BB, SF)  : 153+파일럿+2025 통합 표본에서 평균 사실상 동일(1.07σ) 확인.
#   (OUT, SO) : 예방적 등록(둘 다 대량 표본이라 근소 역전 시 노이즈일 가능성 높음).
#   (HR, 3B)  : 2017년 0.006σ로 확인. 단 3B 표본이 충분히 쌓이면(n>=500)
#               역전은 더 이상 노이즈로 볼 수 없으므로 Tier 1로 엄격화한다
#               (전 구단 데이터 축적 시 자동으로 엄격해지도록 설계).
FRAGILE_PAIR_CONDITIONS = {
    frozenset({"BB", "SF"}): lambda ns: True,
    frozenset({"OUT", "SO"}): lambda ns: True,
    frozenset({"HR", "3B"}): lambda ns: ns.get("3B", 10**9) < 500,
}


# ────────────────────────────────────────────────────────────────────────── #
#  게임 목록 로드
# ────────────────────────────────────────────────────────────────────────── #

def load_year_targets(year: int) -> tuple[list[str], list[dict]]:
    """(크롤링 대상 game_id 목록, 제외된 게임 사유 리스트) 반환."""
    all_games_path = os.path.join(GAME_IDS_DIR, f"all_games_{year}.txt")
    with open(all_games_path, encoding="utf-8") as f:
        all_ids = [line.strip() for line in f if line.strip()]

    idx = pd.read_csv(GAME_INDEX_PATH, dtype=str)
    idx = idx[idx["game_id"].isin(all_ids)]

    targets: list[str] = []
    excluded: list[dict] = []

    idx_map = idx.set_index("game_id").to_dict("index")
    for gid in all_ids:
        row = idx_map.get(gid)
        if row is None:
            excluded.append({"game_id": gid, "reason": "not_in_game_index"})
            continue
        if row.get("is_regular_season") != "True":
            continue  # 정규시즌 아님 — 애초에 대상 아님(제외 사유 기록 불필요)
        status = row.get("game_status")
        if status == "suspended":
            excluded.append({"game_id": gid, "reason": "suspended_pending_resume"})
            continue
        if status == "cancelled":
            excluded.append({"game_id": gid, "reason": "cancelled_before_game"})
            continue
        targets.append(gid)

    return sorted(targets), excluded


def _append_excluded(excluded: list[dict], year: int) -> None:
    if not excluded:
        return
    file_exists = os.path.isfile(EXCLUDED_PATH)
    with open(EXCLUDED_PATH, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("game_id,year,reason\n")
        for e in excluded:
            f.write(f"{e['game_id']},{year},{e['reason']}\n")


# ────────────────────────────────────────────────────────────────────────── #
#  PBP 크롤링 (429 감지 → concurrency 하향 재시작, 이닝 gap 재시도)
# ────────────────────────────────────────────────────────────────────────── #

async def _retry_missing_innings(
    fetcher: InstrumentedFetcher,
    session: aiohttp.ClientSession,
    game_id: str,
    missing: list[int],
    abort_event: asyncio.Event,
) -> tuple[list[int] | None, bool]:
    """누락 이닝 재시도. (raw payload 리스트 or None, http_429_seen) 반환."""
    recovered_raw: dict[int, dict] = {}
    for inning in missing:
        for attempt in range(MAX_INNING_RETRIES):
            if abort_event.is_set():
                return None, True
            status, raw = await fetcher.fetch_inning(session, game_id, inning)
            if status == "429":
                abort_event.set()
                return None, True
            if status == "200" and raw is not None:
                relays = raw.get("result", {}).get("textRelayData", {}).get("textRelays", [])
                if relays:
                    recovered_raw[inning] = raw
                    break
            backoff = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(backoff)
        else:
            return None, False  # 3회 재시도 후에도 실패 (429는 아님)
    return recovered_raw, False


async def _process_one_game(
    fetcher: InstrumentedFetcher,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    abort_event: asyncio.Event,
    game_id: str,
    pbp_dir: str,
    results: dict,
) -> None:
    csv_path = os.path.join(pbp_dir, f"{game_id}.csv")
    if os.path.isfile(csv_path):
        results[game_id] = "skipped_existing"
        return

    async with sem:
        if abort_event.is_set():
            results[game_id] = "pending_429"
            return

        innings_data, status_counter, aborted = await fetcher.fetch_game(session, game_id, abort_event)
        if aborted:
            results[game_id] = "pending_429"
            return
        if not innings_data:
            results[game_id] = "failed_no_data"
            return

        parser = PitchDataParser()
        rows = parser.parse_game(game_id, innings_data)
        if not rows:
            results[game_id] = "failed_no_rows"
            return

        df = pd.DataFrame(rows)
        check = check_game_completeness(df[["inning", "home_or_away"]])

        if check["status"] == "incomplete_gap":
            missing = check["missing_innings"]
            recovered_raw, http_429 = await _retry_missing_innings(
                fetcher, session, game_id, missing, abort_event
            )
            if http_429:
                results[game_id] = "pending_429"
                return
            if recovered_raw is None:
                results[game_id] = "failed_incomplete_after_retry"
                return

            merged = list(innings_data) + [{"inning": i, "data": r} for i, r in recovered_raw.items()]
            merged.sort(key=lambda x: x["inning"])
            parser2 = PitchDataParser()
            rows = parser2.parse_game(game_id, merged)
            df = pd.DataFrame(rows)
            recheck = check_game_completeness(df[["inning", "home_or_away"]])
            if recheck["status"] == "incomplete_gap":
                results[game_id] = "failed_incomplete_after_retry"
                return

        df = _sort_columns(df)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        results[game_id] = "ok"


async def crawl_year_pbp(year: int, game_ids: list[str]) -> dict:
    pbp_dir = PBP_DIR_TMPL.format(year=year)
    os.makedirs(pbp_dir, exist_ok=True)

    remaining = list(game_ids)
    all_results: dict[str, str] = {}
    http_429_events: list[dict] = []
    concurrency_used = None

    for concurrency in CONCURRENCY_LADDER:
        if not remaining:
            break
        concurrency_used = concurrency
        fetcher = InstrumentedFetcher(min_delay=MIN_DELAY, max_delay=MAX_DELAY)
        sem = asyncio.Semaphore(concurrency)
        abort_event = asyncio.Event()
        results: dict[str, str] = {}

        logger.info("[%d] PBP 크롤링 시작 concurrency=%d 대상=%d경기", year, concurrency, len(remaining))
        async with fetcher.create_session() as session:
            tasks = [
                _process_one_game(fetcher, session, sem, abort_event, gid, pbp_dir, results)
                for gid in remaining
            ]
            await asyncio.gather(*tasks)

        if abort_event.is_set():
            http_429_events.append({
                "year": year, "concurrency": concurrency,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_completed_before_429": sum(1 for v in results.values() if v == "ok"),
            })
            logger.error(
                "[%d] HTTP 429 감지! concurrency=%d 중단. 하향 재시작 예정.", year, concurrency
            )

        all_results.update({k: v for k, v in results.items() if v != "pending_429"})
        remaining = [gid for gid in remaining if results.get(gid) in ("pending_429", None)]

        if not abort_event.is_set():
            break  # 429 없이 끝났으면 남은 게임 없음(전부 처리됨) → 종료

    return {
        "results": all_results,
        "http_429_events": http_429_events,
        "final_concurrency": concurrency_used,
        "unresolved": remaining,
    }


# ────────────────────────────────────────────────────────────────────────── #
#  preview 크롤링
# ────────────────────────────────────────────────────────────────────────── #

async def crawl_year_preview(year: int, game_ids: list[str]) -> dict:
    preview_dir = PREVIEW_DIR_TMPL.format(year=year)
    os.makedirs(preview_dir, exist_ok=True)
    fetcher = PreviewAPIFetcher(min_interval=MIN_DELAY)

    n_ok, n_skipped_existing, n_skipped_early, n_failed = 0, 0, 0, 0

    async with fetcher.create_session() as session:
        for gid in game_ids:
            json_path = os.path.join(preview_dir, f"{gid}.json")
            if not is_preview_supported(gid):
                n_skipped_early += 1
                continue
            if os.path.isfile(json_path):
                n_skipped_existing += 1
                continue
            payload = await fetcher.fetch_preview(session, gid)
            if payload is None:
                n_failed += 1
                continue
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            n_ok += 1

    return {
        "n_ok": n_ok, "n_skipped_existing": n_skipped_existing,
        "n_skipped_early": n_skipped_early, "n_failed": n_failed,
    }


# ────────────────────────────────────────────────────────────────────────── #
#  PA 집계 → 상태전이 → WPA → called 보정
# ────────────────────────────────────────────────────────────────────────── #

def process_year_downstream(year: int) -> pd.DataFrame:
    pbp_dir = PBP_DIR_TMPL.format(year=year)
    csv_files = sorted(glob.glob(os.path.join(pbp_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"CSV 없음: {pbp_dir}")

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    pitch_df = pd.concat(dfs, ignore_index=True)
    pa_df = aggregate_pa(pitch_df)

    os.makedirs(STAGING_DIR, exist_ok=True)
    tmp_pa_path = os.path.join(STAGING_DIR, f"_tmp_pa_{year}.parquet")
    tmp_states_path = os.path.join(STAGING_DIR, f"_tmp_pa_states_{year}.parquet")
    final_path = os.path.join(STAGING_DIR, f"pa_states_{year}.parquet")

    pa_df.to_parquet(tmp_pa_path, index=False)
    build_state_transitions(input_path=tmp_pa_path, output_path=tmp_states_path)
    wpa_df = inject_computed_wpa(input_path=tmp_states_path, output_path=final_path)

    # 연장전으로 간 게임은 9회말 동점 종료 시점을 we_after=0.5로 확정하고
    # 10회 이후 PA를 제외한다 — 터미널 PA 보정(ground truth 최종 스코어)은
    # 그 게임들에는 적용하지 않는다(실제 연장 결과가 아니라 이론적 0.5가
    # 정답이므로).
    extras_games = identify_extra_innings_games(wpa_df)

    # 터미널 PA 보정(schedule API 최종 스코어 ground truth) — 연장으로 가지
    # 않은 게임의 마지막 PA에 적용되며, called 게임의 마지막 PA도 이 로직으로
    # 함께 처리된다(called_game_correction.py를 포괄하는 상위 로직 — Round 3
    # 에서 통합, 별도 called 전용 보정은 더 이상 호출하지 않음).
    final_scores = load_final_scores(FINAL_SCORES_PATH)
    corrected, corrections = apply_terminal_pa_correction(
        wpa_df, final_scores, exclude_game_ids=extras_games
    )
    n_changed = sum(1 for c in corrections if c["delta"] != 0)
    logger.info("[%d] 터미널 PA 보정 %d건 처리(실질 변경 %d건)", year, len(corrections), n_changed)

    # 연장전 정책 적용(9회말 동점=0.5 확정 + 10회 이후 PA 제외)
    filtered, extras_corrections, extras_stats = apply_extra_innings_policy(corrected)
    logger.info(
        "[%d] 연장전 정책: 진입 %d경기 / 9회말 동점 확정 %d건 / 제외 PA %d건",
        year, extras_stats["n_extra_innings_games"],
        extras_stats["n_9th_tied_corrections"], extras_stats["n_excluded_pa"],
    )

    filtered.to_parquet(final_path, index=False)
    wpa_df = filtered

    os.remove(tmp_pa_path)
    os.remove(tmp_states_path)
    return wpa_df


# ────────────────────────────────────────────────────────────────────────── #
#  WPA 도메인 순서 — Tier 1(절대 위반)/Tier 2(경고, 2×SEM 이내 역전) 판정
# ────────────────────────────────────────────────────────────────────────── #

def evaluate_wpa_order(pa_df: pd.DataFrame) -> dict:
    means, stds, ns = {}, {}, {}
    for pa in EXPECTED_ORDER:
        sub = pa_df.loc[pa_df["pa_result"] == pa, "reward_wpa_computed"]
        if len(sub):
            means[pa] = float(sub.mean())
            stds[pa] = float(sub.std())
            ns[pa] = int(len(sub))

    violations: list[str] = []
    warnings: list[dict] = []

    for pa in HIT_CATEGORIES:
        if pa in means and means[pa] < 0:
            violations.append(f"안타 계열 {pa} 평균이 음수: {means[pa]:.4f}")

    for pa in OUT_CATEGORIES:
        if pa in means and means[pa] > 0:
            violations.append(f"아웃 계열 {pa} 평균이 양수: {means[pa]:.4f}")

    for pa in ("BB", "SF"):
        if pa in means and means[pa] < 0:
            violations.append(f"{pa} 평균이 음수: {means[pa]:.4f}")

    # 인접 카테고리 순서 검증 — HR>3B>2B>1B>BB>SF>OUT>SO>GDP 전체를 여기서
    # 한 번에 검증한다(HR/GDP가 극값인지도 인접 비교의 연쇄로 자연히 보장됨 —
    # 별도의 전역 max/min 체크는 fragile-pair 예외를 우회시키는 중복 로직이라 제거).
    present = [pa for pa in EXPECTED_ORDER if pa in means]
    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        if means[a] >= means[b]:
            continue  # 정상 순서

        diff = means[b] - means[a]
        n_a, n_b = ns[a], ns[b]
        sem = ((stds[a] ** 2) / n_a + (stds[b] ** 2) / n_b) ** 0.5 if n_a and n_b else float("inf")
        ratio = diff / sem if sem else float("inf")

        condition = FRAGILE_PAIR_CONDITIONS.get(frozenset({a, b}))
        is_fragile = condition is not None and condition(ns)

        if is_fragile and ratio <= SEM_TOLERANCE:
            warnings.append({
                "pair": f"{a}_{b}", "values": [means[a], means[b]],
                "diff": means[a] - means[b], "sem": sem, "ratio": ratio, "n": [n_a, n_b],
            })
        else:
            reason = "fragile pair 아님" if condition is None else (
                "허용범위(2×SEM) 초과" if condition(ns) else "조건 불충족(표본 충분)"
            )
            violations.append(
                f"{a}({means[a]:.4f}) < {b}({means[b]:.4f}) 순서 위반, "
                f"ratio={ratio:.2f}σ ({reason})"
            )

    return {
        "means": means, "tier1_pass": len(violations) == 0,
        "tier1_violations": violations, "warnings": warnings,
    }


# ────────────────────────────────────────────────────────────────────────── #
#  체크포인트
# ────────────────────────────────────────────────────────────────────────── #

def compute_checkpoint(
    year: int,
    targets: list[str],
    pbp_result: dict,
    preview_result: dict,
    pa_df: pd.DataFrame,
) -> dict:
    checkpoint: dict = {"year": year}

    # 1) 경기 수
    n_ok = sum(1 for v in pbp_result["results"].values() if v in ("ok", "skipped_existing"))
    checkpoint["1_game_count"] = {
        "target": len(targets), "success": n_ok,
        "ratio": n_ok / len(targets) if targets else 0.0,
        "pass": n_ok == len(targets),
    }

    # 2) 이닝 완전성
    pbp_dir = PBP_DIR_TMPL.format(year=year)
    completeness_df = check_pbp_dir(pbp_dir)
    n_gap = int((completeness_df["status"] == "incomplete_gap").sum())
    checkpoint["2_inning_completeness"] = {"n_incomplete_gap": n_gap, "pass": n_gap == 0}

    # 3) UNK 비율
    unk_rate = (pa_df["pa_result"] == "UNK").mean() if len(pa_df) else 0.0
    checkpoint["3_unk_rate"] = {"rate": float(unk_rate), "pass": unk_rate < 0.05}

    # 4) WPA 도메인 순서 — Tier 1(절대 위반)/Tier 2(2×SEM 이내 역전, 경고만)
    wpa_order = evaluate_wpa_order(pa_df)
    checkpoint["4_wpa_domain_order"] = {
        "means": wpa_order["means"], "pass": wpa_order["tier1_pass"],
        "wpa_order": {
            "tier1_pass": wpa_order["tier1_pass"],
            "tier1_violations": wpa_order["tier1_violations"],
            "warnings": wpa_order["warnings"],
        },
    }

    # SF 건수 sanity check (relay_text 정규식 연도별 안정성 확인용)
    n_sf = int((pa_df["pa_result"] == "SF").sum())
    checkpoint["sf_count_check"] = {
        "n_sf": n_sf, "n_pa": len(pa_df),
        "sf_rate": n_sf / len(pa_df) if len(pa_df) else 0.0,
    }

    # 5) 팀 커버리지
    idx = pd.read_csv(GAME_INDEX_PATH, dtype=str)
    year_games = idx[idx["game_id"].isin(targets)]
    teams_seen = set(year_games["away_code"]) | set(year_games["home_code"])
    checkpoint["5_team_coverage"] = {
        "teams_seen": sorted(teams_seen), "n_teams": len(teams_seen), "pass": len(teams_seen) >= 10,
    }

    # 6) preview 커버리지 — "preview 없음"이 전부 2017-05-30 이전 설계상 스킵인지,
    #    그리고 그 이후 경기의 실패율이 0%에 가까운지를 명시적으로 분리 검증한다.
    #    (2017년처럼 연도 내에 임계일이 걸쳐 있으면 전체 커버리지 %만으로는
    #    "정상적으로 낮은 것"과 "진짜 실패"를 구분할 수 없으므로 파일 존재 여부를
    #    game_id 단위로 직접 확인한다.)
    preview_dir = PREVIEW_DIR_TMPL.format(year=year)
    supported_targets = [gid for gid in targets if is_preview_supported(gid)]
    unsupported_targets = [gid for gid in targets if not is_preview_supported(gid)]

    missing_preview = [
        gid for gid in targets
        if not os.path.isfile(os.path.join(preview_dir, f"{gid}.json"))
    ]
    missing_but_supported = [gid for gid in missing_preview if is_preview_supported(gid)]
    missing_all_pre_threshold = len(missing_but_supported) == 0

    n_supported = len(supported_targets)
    n_missing_supported = len(missing_but_supported)
    post_threshold_fail_rate = n_missing_supported / n_supported if n_supported else 0.0

    checkpoint["6_preview_coverage"] = {
        **preview_result,
        "n_targets": len(targets),
        "n_unsupported_pre_20170530": len(unsupported_targets),
        "n_supported_post_20170530": n_supported,
        "n_missing_total": len(missing_preview),
        "n_missing_but_should_have_preview": n_missing_supported,
        "missing_but_should_have_preview_ids": missing_but_supported,
        "all_missing_are_pre_threshold": missing_all_pre_threshold,
        "post_threshold_fail_rate": post_threshold_fail_rate,
        "pass": missing_all_pre_threshold and post_threshold_fail_rate <= 0.01,
    }

    # 7) called 게임 보정
    called_ids = set(idx.loc[(idx["game_status"] == "called") & idx["game_id"].isin(targets), "game_id"])
    checkpoint["7_called_game_correction"] = {
        "n_called": len(called_ids), "called_ids": sorted(called_ids),
    }

    checkpoint["overall_pass"] = all(
        checkpoint[k]["pass"] for k in [
            "1_game_count", "2_inning_completeness", "3_unk_rate",
            "4_wpa_domain_order", "5_team_coverage", "6_preview_coverage",
        ]
    )
    checkpoint["http_429_events"] = pbp_result["http_429_events"]
    checkpoint["final_concurrency"] = pbp_result["final_concurrency"]
    checkpoint["unresolved_games"] = pbp_result["unresolved"]

    return checkpoint


# ────────────────────────────────────────────────────────────────────────── #
#  연도 실행
# ────────────────────────────────────────────────────────────────────────── #

def run_year(year: int) -> dict:
    t0 = time.time()
    pbp_dir = PBP_DIR_TMPL.format(year=year)
    n_existing = len(glob.glob(os.path.join(pbp_dir, "*.csv"))) if os.path.isdir(pbp_dir) else 0

    targets, excluded = load_year_targets(year)
    print(f"\n{year}: {n_existing}/{len(targets)} 완료 (기존 스킵) — 신규 대상 {len(targets)-n_existing}경기")
    _append_excluded(excluded, year)
    if excluded:
        logger.info("[%d] 제외 %d건: %s", year, len(excluded), Counter(e["reason"] for e in excluded))

    pbp_result = asyncio.run(crawl_year_pbp(year, targets))
    preview_result = asyncio.run(crawl_year_preview(year, targets))
    pa_df = process_year_downstream(year)
    checkpoint = compute_checkpoint(year, targets, pbp_result, preview_result, pa_df)

    elapsed = time.time() - t0
    checkpoint["elapsed_sec"] = elapsed

    os.makedirs(STAGING_DIR, exist_ok=True)
    checkpoint_path = os.path.join(STAGING_DIR, f"checkpoint_{year}.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)

    n_ok = sum(1 for v in pbp_result["results"].values() if v == "ok")
    n_skipped = sum(1 for v in pbp_result["results"].values() if v == "skipped_existing")
    n_failed = sum(1 for v in pbp_result["results"].values() if v.startswith("failed"))

    print(f"\n{'='*60}")
    print(f"[{year}] 완료 — 소요 {elapsed:.1f}초")
    print(f"  성공(신규): {n_ok} / 스킵(기존): {n_skipped} / 실패: {n_failed}")
    print(f"  PA: {len(pa_df):,}건")
    print(f"  체크포인트: {'PASS' if checkpoint['overall_pass'] else 'FAIL'}")
    for k in ["1_game_count", "2_inning_completeness", "3_unk_rate", "4_wpa_domain_order", "5_team_coverage", "6_preview_coverage"]:
        print(f"    {k}: {'PASS' if checkpoint[k]['pass'] else 'FAIL'}")
    pc = checkpoint["6_preview_coverage"]
    print(
        f"      preview 없음 총 {pc['n_missing_total']}건 "
        f"(2017-05-30 이전 스킵 {pc['n_missing_total'] - pc['n_missing_but_should_have_preview']}건 / "
        f"그 이후인데 실패 {pc['n_missing_but_should_have_preview']}건, "
        f"실패율 {pc['post_threshold_fail_rate']:.2%})"
    )
    print(f"  called 게임 보정: {checkpoint['7_called_game_correction']['n_called']}건")
    sf_chk = checkpoint["sf_count_check"]
    print(f"  SF 건수: {sf_chk['n_sf']}건 / PA {sf_chk['n_pa']:,}건 (SF율 {sf_chk['sf_rate']*100:.3f}%)")
    warnings = checkpoint["4_wpa_domain_order"]["wpa_order"]["warnings"]
    if warnings:
        print(f"  Tier 2 경고 (2×SEM 이내 역전, 진행은 계속):")
        for w in warnings:
            print(f"    {w['pair']}: {w['values'][0]:.4f} vs {w['values'][1]:.4f} "
                  f"(diff={w['diff']:.4f}, ratio={w['ratio']:.2f}σ, n={w['n']})")
    print(f"  저장: {checkpoint_path}")
    print(f"{'='*60}")

    if not checkpoint["4_wpa_domain_order"]["pass"]:
        print("\n" + "!" * 60)
        print(f"[중단] {year}년 WPA 도메인 순서 Tier 1 위반 — 절대 금지 규칙에 따라 다음 연도 진행 중단")
        for v in checkpoint["4_wpa_domain_order"]["wpa_order"]["tier1_violations"]:
            print(f"  - {v}")
        print("!" * 60)

    return checkpoint


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m data_analysis.methods.run_full_year <year>")
        sys.exit(1)
    year = int(sys.argv[1])
    checkpoint = run_year(year)
    if not checkpoint["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
