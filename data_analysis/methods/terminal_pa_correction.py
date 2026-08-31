"""
data_analysis/methods/terminal_pa_correction.py

모든 게임의 "후계 그룹 없는 진짜 마지막 PA"는 state_transition.py에서
_estimate_runs_from_pa_result() 폴백(HR/SF/만루BB 외에는 항상 runs=0)을
거친다. 이 모듈은 kbo_crawler/final_scores_scan.py가 수집한 schedule API
최종 스코어를 ground truth로 사용해 그 마지막 PA의 runs_scored/we_after를
직접 확정한다.

called_game_correction.py(콜드게임 마지막 PA를 경기 결과로 확정)와 원리가
동일하다 — 콜드게임은 "마지막 PA가 곧 경기 종료"라는 조건의 부분집합일 뿐이고,
사실 모든 게임의 마지막 PA가 같은 처리를 필요로 한다. 이 모듈이 상위 개념이며
called_game_correction.apply_called_game_correction()은 이 함수로 완전히
대체 가능하다(둘 다 "마지막 PA를 ground truth로 확정" 로직).

we_after 확정 규칙 (called_game_correction과 동일):
  공격팀 승 → 1.0 / 공격팀 패 → 0.0 / 무승부 → 0.5

state_transition.py / inject_wpa.py는 수정하지 않는다 — 이 모듈은 그
산출물 위에서 동작하는 독립 후처리 단계다.

실행:
    uv run python -m data_analysis.methods.terminal_pa_correction <pa_wpa_parquet> <final_scores_csv>
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def load_final_scores(path: str) -> dict[str, dict]:
    df = pd.read_csv(path, dtype={"game_id": str})
    return {
        row["game_id"]: {
            "home_score": int(row["home_score"]), "away_score": int(row["away_score"]),
            "winner": row["winner"],
        }
        for _, row in df.iterrows()
    }


def classify_terminal_pas(df: pd.DataFrame, final_scores: dict[str, dict]) -> pd.DataFrame:
    """
    각 게임의 마지막 PA(_orig_idx 최대)를 추출해 ground truth와 대조,
    유형을 분류한다: normal(일치) / missing_runs(득점 누락 의심 및 확인) /
    other_mismatch(그 외 불일치) / truth_unavailable(ground truth 없음).
    """
    df = df.sort_values("_orig_idx").reset_index(drop=True)
    last_idx = df.groupby("game_id")["_orig_idx"].idxmax()
    last_rows = df.loc[last_idx].copy()

    rows = []
    for _, row in last_rows.iterrows():
        gid = row["game_id"]
        truth = final_scores.get(gid)
        if truth is None:
            rows.append({**row.to_dict(), "category": "truth_unavailable", "true_runs_in_terminal_pa": None})
            continue

        true_final_diff = truth["home_score"] - truth["away_score"]  # H1: home-away
        cur_diff = int(row["score_diff"])  # PA 시작 시점 H1 diff

        if row["home_or_away"] == 1:  # 말(홈 공격)
            implied_runs = true_final_diff - cur_diff
        else:  # 초(원정 공격)
            implied_runs = cur_diff - true_final_diff

        recorded_runs = int(row["runs_scored"])
        matches = implied_runs == recorded_runs

        if matches:
            category = "normal"
        elif implied_runs < 0:
            category = "other_mismatch"  # 음수 = 매칭 오류/데이터 이상 가능성
        elif row["pa_result"] in ("1B", "2B", "3B", "BB") and row["is_base3"] == 1 and recorded_runs == 0:
            category = "missing_runs_confirmed"
        elif implied_runs != recorded_runs:
            category = "missing_runs_other"
        else:
            category = "normal"

        rows.append({
            **row.to_dict(), "category": category,
            "true_final_diff": true_final_diff, "implied_runs": implied_runs,
            "recorded_runs": recorded_runs,
        })

    return pd.DataFrame(rows)


def is_walkoff(row) -> bool:
    """9회 이상 말(홈 공격)에서 홈팀이 역전/결승 상황으로 경기가 끝난 경우."""
    return bool(row["inning"] >= 9 and row["half"] == "bot" and row.get("implied_runs", 0) > 0)


# ────────────────────────────────────────────────────────────────────────── #
#  보정 적용
# ────────────────────────────────────────────────────────────────────────── #

def apply_terminal_pa_correction(
    df: pd.DataFrame,
    final_scores: dict[str, dict],
    exclude_game_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    모든 게임의 마지막 PA에 대해 runs_scored/we_after/reward_wpa_computed를
    ground truth(schedule API 최종 스코어) 기준으로 확정한다.
    ground truth를 못 찾으면 해당 게임은 건드리지 않는다(기존 값 유지).

    exclude_game_ids: 이 집합에 속한 game_id는 건드리지 않는다 — 연장전으로
    간 게임은 extra_innings_policy.py가 "9회말 동점=0.5" 규칙으로 별도
    처리하므로(실제 연장 결과가 아니라 이론적 0.5를 확정해야 함), 여기서
    ground-truth 기반으로 먼저 덮어써버리면 안 된다.
    """
    df = df.sort_values("_orig_idx").reset_index(drop=True)
    last_idx = df.groupby("game_id")["_orig_idx"].idxmax()
    exclude_game_ids = exclude_game_ids or set()

    corrections: list[dict] = []

    for idx in last_idx:
        row = df.loc[idx]
        gid = row["game_id"]
        if gid in exclude_game_ids:
            continue
        truth = final_scores.get(gid)
        if truth is None:
            continue

        true_final_diff = truth["home_score"] - truth["away_score"]
        cur_diff = int(row["score_diff"])

        if row["home_or_away"] == 1:
            implied_runs = true_final_diff - cur_diff
        else:
            implied_runs = cur_diff - true_final_diff

        if implied_runs < 0:
            logger.warning(
                "game=%s: 마지막 PA implied_runs 음수(%d) — 보정 스킵(데이터 확인 필요)",
                gid, implied_runs,
            )
            continue

        # 공격팀 관점 최종 score_diff (해당 PA 종료 후)
        score_diff_after_attacker = int(row["score_diff_attacker"]) + implied_runs

        if score_diff_after_attacker > 0:
            new_we_after = 1.0
        elif score_diff_after_attacker < 0:
            new_we_after = 0.0
        else:
            new_we_after = 0.5

        old_runs = int(row["runs_scored"])
        old_we_after = float(row["we_after"])
        old_reward = float(row["reward_wpa_computed"])
        new_reward = new_we_after - float(row["we_before"])

        df.loc[idx, "runs_scored"] = implied_runs
        df.loc[idx, "we_after"] = new_we_after
        df.loc[idx, "reward_wpa_computed"] = new_reward

        corrections.append({
            "game_id": gid, "inning": int(row["inning"]), "half": row["half"],
            "pa_result": row["pa_result"],
            "old_runs_scored": old_runs, "new_runs_scored": implied_runs,
            "old_we_after": old_we_after, "new_we_after": new_we_after,
            "old_reward_wpa_computed": old_reward, "new_reward_wpa_computed": new_reward,
            "delta": new_reward - old_reward,
        })

    logger.info("터미널 PA 보정 완료: %d건 적용", len(corrections))
    return df, corrections


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if len(sys.argv) != 3:
        print("Usage: python -m data_analysis.methods.terminal_pa_correction <pa_wpa_parquet> <final_scores_csv>")
        sys.exit(1)

    pa_path, scores_path = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(pa_path)
    final_scores = load_final_scores(scores_path)

    corrected, corrections = apply_terminal_pa_correction(df, final_scores)
    corrected.to_parquet(pa_path, index=False)

    n_changed = sum(1 for c in corrections if c["delta"] != 0)
    print(f"\n총 {len(corrections)}건 처리, 실질 변경 {n_changed}건")


if __name__ == "__main__":
    main()
