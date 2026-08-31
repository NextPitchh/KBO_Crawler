"""
data_analysis/methods/extra_innings_policy.py

연장전(10회 이후) 처리 정책:
  - 9회말까지는 WE 테이블 값을 그대로 사용(정교한 상태별 확률 보존).
  - 9회말이 동점으로 종료되면(연장 진입), 그 시점을 "논리적 경기 종료"로
    간주하고 we_after를 정확히 0.5로 확정한다. 근사가 아니라 "동점 = 승률
    정확히 0.5"라는 야구의 실제 구조를 반영한 것이다.
  - 9회말 끝내기(홈팀 역전승)는 기존 처리(terminal_pa_correction의
    ground-truth 기반 1.0 확정)를 그대로 유지한다 — 연장으로 가지 않으므로
    이 정책의 대상이 아니다.
  - 10회 이후 PA는 is_extra_innings=True로 표시한 뒤 최종 데이터셋에서
    제외한다(승부처 투수 교체를 다루는 본 프로젝트의 목적상, WE 테이블
    신뢰도가 떨어지는 연장 구간을 분석 대상에서 뺀다).

state_transition.py / inject_wpa.py / terminal_pa_correction.py는 수정하지
않는다 — 이 모듈은 그 산출물 위에서 동작하는 독립 후처리 단계다.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def identify_extra_innings_games(df: pd.DataFrame) -> set[str]:
    """10회 이상 PA가 하나라도 있는 game_id 집합(=연장전으로 간 게임)."""
    return set(df.loc[df["inning"] >= 10, "game_id"].unique())


def apply_extra_innings_policy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    1) is_extra_innings 플래그 부여(inning>=10)
    2) 연장 진입 게임의 9회말 마지막 PA를 we_after=0.5로 확정
    3) 10회 이상 PA를 제외한 최종 데이터프레임 반환

    Returns
    -------
    (필터링된 df, 9회말 동점 확정 내역 리스트, 통계 dict)
    """
    df = df.sort_values("_orig_idx").reset_index(drop=True)
    df["is_extra_innings"] = df["inning"] >= 10

    extras_games = identify_extra_innings_games(df)
    logger.info("연장전 진입 게임: %d개", len(extras_games))

    corrections: list[dict] = []
    for gid in extras_games:
        mask = (
            (df["game_id"] == gid) & (df["inning"] == 9)
            & (df["half"] == "bot") & (df["inning_ended"])
        )
        if not mask.any():
            logger.warning("game=%s: 연장전 진입인데 9회말 종료 PA를 찾지 못함 — 스킵", gid)
            continue

        idx = df.index[mask][0]
        row = df.loc[idx]
        old_we_after = float(row["we_after"])
        old_reward = float(row["reward_wpa_computed"])
        new_we_after = 0.5
        new_reward = new_we_after - float(row["we_before"])

        df.loc[idx, "we_after"] = new_we_after
        df.loc[idx, "reward_wpa_computed"] = new_reward

        corrections.append({
            "game_id": gid, "old_we_after": old_we_after, "new_we_after": new_we_after,
            "old_reward_wpa_computed": old_reward, "new_reward_wpa_computed": new_reward,
        })

    n_excluded_pa = int(df["is_extra_innings"].sum())
    n_total_games = df["game_id"].nunique()

    stats = {
        "n_extra_innings_games": len(extras_games),
        "n_total_games": n_total_games,
        "extra_innings_game_ratio": len(extras_games) / n_total_games if n_total_games else 0.0,
        "n_excluded_pa": n_excluded_pa,
        "n_total_pa_before": len(df),
        "n_9th_tied_corrections": len(corrections),
    }

    filtered = df[~df["is_extra_innings"]].reset_index(drop=True)
    stats["n_total_pa_after"] = len(filtered)

    logger.info(
        "연장전 정책 적용: 9회말 동점 확정 %d건 / 제외 PA %d건(%d경기 중 %d경기가 연장) / "
        "%d행 → %d행",
        len(corrections), n_excluded_pa, n_total_games, len(extras_games),
        stats["n_total_pa_before"], stats["n_total_pa_after"],
    )

    return filtered, corrections, stats
