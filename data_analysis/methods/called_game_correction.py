"""
data_analysis/methods/called_game_correction.py

콜드게임("called": statusInfo 최종 이닝 < 9, 승자 확정된 경기)의 마지막 PA는
state_transition.py의 기본 로직(다음 이닝/공수 전환을 가정)으로 계산되면
잘못된 we_after를 얻는다 — 실제로는 그 PA에서 경기가 완전히 종료되므로
결과가 이미 확정된 상태(공격팀 승리 시 we_after=1.0, 패배 시 0.0)여야 한다.

이 모듈은 state_transition.py / inject_wpa.py를 수정하지 않고, 그 산출물
(we_before/we_after/reward_wpa_computed 포함 PA parquet) 위에 사후 보정을
적용하는 독립 후처리 단계다. 기존 153경기 데이터에는 called 게임이 0건이므로
(inning_completeness 재검증 결과) 이 보정을 기존 153경기 산출물에 적용할
필요도, 재실행할 이유도 없다 — 파일럿/향후 확장 데이터에만 적용한다.

실행:
    uv run python -m data_analysis.methods.called_game_correction <pa_wpa_parquet> <called_game_ids...>
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def apply_called_game_correction(
    df: pd.DataFrame, called_game_ids: set[str]
) -> tuple[pd.DataFrame, list[dict]]:
    """
    called_game_ids에 속한 각 게임의 (chronologically) 마지막 PA에 대해
    we_after를 경기 최종 결과(공격팀 승/패 확정)로 덮어쓰고, 그에 맞춰
    reward_wpa_computed를 재계산한다. (_orig_idx 기준 정렬이 이미 경기
    내 시간순임을 전제로 한다 — state_transition.py가 보장.)

    Returns
    -------
    (보정된 df, 보정 내역 리스트)
    """
    df = df.sort_values("_orig_idx").reset_index(drop=True)
    corrections: list[dict] = []

    for gid in called_game_ids:
        sub = df[df["game_id"] == gid]
        if sub.empty:
            logger.warning("game=%s: called 대상이나 PA 데이터 없음 — 스킵", gid)
            continue

        last_idx = sub.index[-1]
        row = df.loc[last_idx]
        final_score_diff = float(row["score_diff_attacker"]) + float(row["runs_scored"])

        if final_score_diff > 0:
            new_we_after = 1.0
        elif final_score_diff < 0:
            new_we_after = 0.0
        else:
            # 콜드게임이 동점으로 최종 확정(무승부)된 이례적 케이스 —
            # schedule API winner="DRAW"와 일치. we_after를 정확히 0.5로 고정.
            new_we_after = 0.5
            logger.warning("game=%s: 콜드게임 최종 스코어 동점(무승부) — we_after=0.5로 보정", gid)

        old_we_after = float(row["we_after"])
        old_reward = float(row["reward_wpa_computed"])
        new_reward = new_we_after - float(row["we_before"])

        df.loc[last_idx, "we_after"] = new_we_after
        df.loc[last_idx, "reward_wpa_computed"] = new_reward

        corrections.append({
            "game_id": gid,
            "inning": int(row["inning"]),
            "half": row["half"],
            "old_we_after": old_we_after,
            "new_we_after": new_we_after,
            "old_reward_wpa_computed": old_reward,
            "new_reward_wpa_computed": new_reward,
            "delta": new_reward - old_reward,
        })

    logger.info("called 게임 보정 완료: %d/%d건 적용", len(corrections), len(called_game_ids))
    return df, corrections


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if len(sys.argv) < 3:
        print("Usage: python -m data_analysis.methods.called_game_correction <pa_wpa_parquet> <game_id...>")
        sys.exit(1)

    path = sys.argv[1]
    called_ids = set(sys.argv[2:])

    df = pd.read_parquet(path)
    corrected, corrections = apply_called_game_correction(df, called_ids)

    print("\n" + "=" * 70)
    for c in corrections:
        print(
            f"{c['game_id']} ({c['inning']}회{c['half']}) | "
            f"we_after: {c['old_we_after']:.4f} -> {c['new_we_after']:.4f} | "
            f"reward_wpa_computed: {c['old_reward_wpa_computed']:.4f} -> {c['new_reward_wpa_computed']:.4f} "
            f"(delta={c['delta']:+.4f})"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
