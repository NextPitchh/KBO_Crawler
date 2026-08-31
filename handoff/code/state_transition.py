"""
data_analysis/methods/state_transition.py

각 PA(타석)의 before_state / after_state WE(기대승리확률)를 계산하여
hsk_pa_with_states.parquet으로 저장하는 모듈.

입력  : data_analysis/results/hsk_pa.parquet  (11,984 PA)
출력  : data_analysis/results/hsk_pa_with_states.parquet
추가 컬럼: base_state, half, score_diff_attacker, we_before,
           runs_scored, inning_ended, we_after, data_quality_flag

실행  : python -m data_analysis.methods.state_transition
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np
import pandas as pd

from .we_re_lookup import get_we_with_boundary, get_re  # noqa: F401

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: 유틸 함수
# ─────────────────────────────────────────────────────────────────────────────

def to_attacker_score_diff(score_diff: int, home_or_away: int) -> int:
    """
    H1 컨벤션 score_diff(home - away)를 공격팀 관점 score_diff로 변환.

    home_or_away=0 (원정 공격/초): 공격팀=원정 → 부호 반전 (away-home = -(home-away))
    home_or_away=1 (홈 공격/말): 공격팀=홈 → 그대로 (home-away = 공격팀 리드)
    """
    if home_or_away == 1:
        return int(score_diff)
    return -int(score_diff)


def base_str(is_b1: int, is_b2: int, is_b3: int) -> str:
    """
    주자 플래그(0/1) → 주자상태 문자열.

    plan2.md 2-4절 표기법:
      (0,0,0)→"0", (1,0,0)→"1", (0,1,0)→"2", (0,0,1)→"3"
      (1,1,0)→"12", (1,0,1)→"13", (0,1,1)→"23", (1,1,1)→"123"
    """
    parts: list[str] = []
    if is_b1:
        parts.append("1")
    if is_b2:
        parts.append("2")
    if is_b3:
        parts.append("3")
    return "".join(parts) if parts else "0"


def half_of(home_or_away: int) -> Literal["top", "bot"]:
    """0→"top" (초/원정 공격), 1→"bot" (말/홈 공격)."""
    return "top" if home_or_away == 0 else "bot"


# ─────────────────────────────────────────────────────────────────────────────
#  Step 6 보조: relay_text / pa_result 기반 득점 수 추정
# ─────────────────────────────────────────────────────────────────────────────

RUN_PATTERNS: list[tuple[str, int | str]] = [
    (r"만루\s*홈런", 4),
    (r"3점\s*홈런|쓰리런", 3),
    (r"2점\s*홈런|투런", 2),
    (r"홈런", 1),          # 기본 솔로 홈런
    (r"(\d+)점", "extract"),  # "2점 적시타" 등
]
_COMPILED_RUN_PATTERNS = [(re.compile(p), v) for p, v in RUN_PATTERNS]


def parse_runs_from_relay(text: str) -> int:
    """
    relay_text(한국어 중계 텍스트)에서 득점 수를 파싱.

    RUN_PATTERNS 순서로 첫 매칭 사용.
    "extract" 값은 첫 번째 캡처 그룹의 숫자를 반환.
    매칭 없으면 0.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    for pattern, value in _COMPILED_RUN_PATTERNS:
        m = pattern.search(text)
        if m:
            if value == "extract":
                return int(m.group(1))
            return int(value)
    return 0


def _estimate_runs_from_pa_result(
    pa_result: str,
    is_base1: int,
    is_base2: int,
    is_base3: int,
) -> int:
    """
    pa_result + 주자 상태로 득점 수 추정.

    relay_text / 후계 그룹 score_diff 모두 미제공 시 폴백으로 사용.
    HR: 타자 + 전 주자 득점.
    SF: 3루 주자 1명 득점.
    BB/IBB/HBP + 만루: 1점 (밀어내기).
    그 외: 0 (이닝 종료 타석에서 추가 득점 없음으로 가정).
    """
    bases_occupied = int(is_base1) + int(is_base2) + int(is_base3)
    if pa_result == "HR":
        return 1 + bases_occupied
    if pa_result == "SF":
        return 1
    if pa_result in ("BB", "IBB", "HBP") and bases_occupied == 3:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Step 7 보조: after_state WE 계산
# ─────────────────────────────────────────────────────────────────────────────

def _compute_we_after(row: pd.Series) -> float:
    """
    PA 종료 시점의 공격팀 관점 WE.

    같은 이닝 내 다음 PA가 있으면 그 before_state를 사용.
    이닝 종료 시 다음 이닝/공수 시작 시점의 상대팀 WE에서 역산.
    """
    score_diff_after_attacker = int(row["score_diff_attacker"]) + int(row["runs_scored"])

    if not row["inning_ended"]:
        return get_we_with_boundary(
            int(row["inning"]),
            str(row["half"]),
            score_diff_after_attacker,
            int(row["_next_out_count"]),
            str(row["_next_base_state"]),
        )

    # 이닝 종료 처리
    inning: int = int(row["inning"])
    cur_half: str = str(row["half"])

    if cur_half == "top":
        next_inning = inning
        next_half: str = "bot"
    else:
        next_inning = inning + 1
        next_half = "top"

    # 9회말 끝내기: 홈팀이 9회말 도중 역전/결승점 득점
    if inning == 9 and cur_half == "bot" and score_diff_after_attacker > 0:
        return 1.0

    # 12회 초과 → 경기 종료 (무승부 처리)
    if next_inning > 12:
        if score_diff_after_attacker > 0:
            return 1.0
        if score_diff_after_attacker < 0:
            return 0.0
        return 0.5

    # 정상 이닝 전환: 상대팀(다음 공격팀) 관점 WE = 1 − 현재 공격팀 WE
    # 다음 공격팀의 score_diff = −score_diff_after_attacker (공수 전환)
    opponent_we = get_we_with_boundary(
        next_inning,
        next_half,
        -score_diff_after_attacker,
        out_count=0,
        base_state="0",
    )
    return 1.0 - opponent_we


# ─────────────────────────────────────────────────────────────────────────────
#  메인 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_state_transitions(
    input_path: str = "data_analysis/results/hsk_pa.parquet",
    output_path: str = "data_analysis/results/hsk_pa_with_states.parquet",
) -> pd.DataFrame:
    """
    각 PA의 before / after WE를 계산하여 parquet으로 저장.

    Parameters
    ----------
    input_path  : 입력 parquet 경로 (기본: hsk_pa.parquet)
    output_path : 출력 parquet 경로

    Returns
    -------
    pd.DataFrame : 처리 완료 DataFrame
    """

    # ── 로드 ─────────────────────────────────────────────────────────────────
    df = pd.read_parquet(input_path)
    logger.info("로드: %d 행, %d 컬럼", *df.shape)

    # ── Step 2: 정렬 보장 + 원본 인덱스 보존 ─────────────────────────────────
    df["_orig_idx"] = df.index  # parquet 원본 행 번호 — 디버깅 시 역추적용
    # 시간순(초→말): home_or_away 0=초(top) 1=말(bot) 이므로 0이 앞에 와야 한다.
    # 숫자 역순에 의존하지 않도록 임시 키로 명시적 매핑 후 전부 오름차순 정렬.
    # parser.py에서 PA 역순 보정이 완료되었으므로 그룹 내 순서 반전은 불필요.
    df["_sort_half"] = df["home_or_away"].map({0: 0, 1: 1})  # 초(ha=0)→0, 말(ha=1)→1
    df = df.sort_values(
        ["game_id", "inning", "_sort_half"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    df = df.drop(columns=["_sort_half"])

    # ── Step 3: before_state 컬럼 생성 ───────────────────────────────────────
    df["base_state"] = df.apply(
        lambda r: base_str(int(r["is_base1"]), int(r["is_base2"]), int(r["is_base3"])),
        axis=1,
    )
    df["half"] = df["home_or_away"].map({0: "top", 1: "bot"})
    df["score_diff_attacker"] = df.apply(
        lambda r: to_attacker_score_diff(int(r["score_diff"]), int(r["home_or_away"])),
        axis=1,
    )
    df["we_before"] = df.apply(
        lambda r: get_we_with_boundary(
            int(r["inning"]),
            str(r["half"]),
            int(r["score_diff_attacker"]),
            int(r["out_count"]),
            str(r["base_state"]),
        ),
        axis=1,
    )

    # ── Step 4: runs_scored (같은 (game_id, inning, home_or_away) 내 diff) ──
    # H1 컨벤션: score_diff = home − away
    #   홈   공격(1/말): home 득점 → score_diff 증가 → runs = next − cur
    #   원정 공격(0/초): away 득점 → score_diff 감소 → runs = cur − next
    df["_next_score_diff"] = (
        df.groupby(["game_id", "inning", "home_or_away"])["score_diff"]
        .shift(-1)
    )
    df["runs_scored"] = np.where(
        df["home_or_away"] == 1,
        df["_next_score_diff"] - df["score_diff"],
        df["score_diff"] - df["_next_score_diff"],
    )
    # 이닝 마지막 PA는 NaN → Step 6에서 보정

    # ── Step 5: inning_ended 판별 ─────────────────────────────────────────────
    df["inning_ended"] = ~df.duplicated(
        subset=["game_id", "inning", "home_or_away"], keep="last"
    )
    n_groups = df.groupby(["game_id", "inning", "home_or_away"]).ngroups
    n_ended = int(df["inning_ended"].sum())
    assert n_groups == n_ended, (
        f"inning_ended 카운트 불일치: groups={n_groups}, ended={n_ended}"
    )
    logger.info("inning_ended: %d 그룹 확인", n_groups)

    # ── Step 6: 이닝 마지막 PA의 runs_scored 보정 ────────────────────────────
    # shift(-1)는 그룹 마지막 PA의 값을 얻지 못하므로,
    # 후계 그룹(다음 half-inning)의 첫 score_diff를 이용해 역산.
    #   원정공격(0/초) 마지막 PA → 후계 = (game_id, inning, 1)   홈/말 첫 score_diff
    #   홈공격(1/말) 마지막 PA   → 후계 = (game_id, inning+1, 0) 원정/초 첫 score_diff

    # 후계 그룹 첫 score_diff 조회 딕셔너리
    _group_first_sd: dict[tuple, float] = (
        df.groupby(["game_id", "inning", "home_or_away"])["score_diff"]
        .first()
        .to_dict()
    )

    def _get_succ_first_sd(
        game_id: str, inning: int, home_or_away: int
    ) -> float | None:
        if home_or_away == 0:  # 원정공격(초) 끝 → 같은 이닝 홈공격(말)
            return _group_first_sd.get((game_id, inning, 1))
        return _group_first_sd.get((game_id, inning + 1, 0))  # 홈공격(말) 끝 → 다음 이닝 원정공격(초)

    # 이닝 마지막 PA에 후계 그룹 첫 score_diff 부착
    df["_succ_first_sd"] = np.nan
    inning_end_mask = df["inning_ended"]
    df.loc[inning_end_mask, "_succ_first_sd"] = df.loc[inning_end_mask].apply(
        lambda r: _get_succ_first_sd(
            str(r["game_id"]), int(r["inning"]), int(r["home_or_away"])
        ),
        axis=1,
    )

    # 후계 그룹 데이터가 있는 케이스: cross-inning score_diff 역산
    top_end = inning_end_mask & (df["home_or_away"] == 0) & df["_succ_first_sd"].notna()
    bot_end = inning_end_mask & (df["home_or_away"] == 1) & df["_succ_first_sd"].notna()

    # top(초/원정): away 득점 → score_diff 감소 → runs = cur − succ
    df.loc[top_end, "runs_scored"] = (
        df.loc[top_end, "score_diff"] - df.loc[top_end, "_succ_first_sd"]
    )
    # bot(말/홈): home 득점 → score_diff 증가 → runs = succ − cur
    df.loc[bot_end, "runs_scored"] = (
        df.loc[bot_end, "_succ_first_sd"] - df.loc[bot_end, "score_diff"]
    )

    # 후계 그룹이 없는 케이스(경기 마지막 이닝): pa_result 기반 폴백
    still_nan_mask = df["inning_ended"] & df["runs_scored"].isna()
    n_fallback = int(still_nan_mask.sum())
    parse_fail = 0
    if n_fallback > 0:
        for idx in df[still_nan_mask].index:
            row = df.loc[idx]
            runs = _estimate_runs_from_pa_result(
                str(row["pa_result"]),
                int(row["is_base1"]),
                int(row["is_base2"]),
                int(row["is_base3"]),
            )
            df.loc[idx, "runs_scored"] = float(runs)
            if runs == 0 and row["pa_result"] not in (
                "OUT", "SO", "GDP", "UNK", "BB", "IBB", "HBP", "SF",
            ):
                parse_fail += 1

        fail_rate = parse_fail / n_fallback if n_fallback else 0.0
        logger.info("후계 그룹 없는 이닝 마지막 PA: %d건 → pa_result 폴백", n_fallback)
        if fail_rate > 0.1:
            logger.warning(
                "runs_scored 추정 불확실 비율 %.1f%% (10%% 초과)", fail_rate * 100
            )

    # 음수 보정 (컨벤션 오류 방어)
    neg_mask = df["runs_scored"] < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        logger.warning("음수 runs_scored %d건 → 0으로 보정", n_neg)
        df.loc[neg_mask, "runs_scored"] = 0

    df["runs_scored"] = df["runs_scored"].fillna(0).astype(int)

    # ── Step 7: after_state WE 계산 ──────────────────────────────────────────
    df["_next_out_count"] = (
        df.groupby(["game_id", "inning", "home_or_away"])["out_count"].shift(-1)
    )
    df["_next_base_state"] = (
        df.groupby(["game_id", "inning", "home_or_away"])["base_state"].shift(-1)
    )

    df["we_after"] = df.apply(_compute_we_after, axis=1)

    # ── Step 8: 데이터 품질 플래그 ───────────────────────────────────────────
    first_pa_mask = (
        ~df.duplicated(subset=["game_id", "inning", "home_or_away"], keep="first")
        & (df["inning"] == 1)
    )
    df["data_quality_flag"] = ""
    df.loc[first_pa_mask & (df["score_diff"] != 0), "data_quality_flag"] = (
        "inning1_nonzero_start"
    )
    high_runs_mask = df["runs_scored"] >= 5
    df.loc[high_runs_mask & (df["data_quality_flag"] == ""), "data_quality_flag"] = (
        "high_runs_scored_artifact"
    )

    # ── 임시 컬럼 정리 (_orig_idx는 유지) ────────────────────────────────────
    tmp_cols = [
        c for c in df.columns
        if c.startswith("_") and c != "_orig_idx"
    ]
    df = df.drop(columns=tmp_cols, errors="ignore")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    df.to_parquet(output_path, index=False)
    logger.info("저장 완료: %s (%d 행)", output_path, len(df))

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  __main__: sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # 프로젝트 루트 기준 상대 경로 유지 (python -m 실행 시 CWD = 프로젝트 루트)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    df_out = build_state_transitions()

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  State Transition sanity check")
    print(SEP)

    # 1. shape 검증
    EXPECTED_ROWS = 11_984
    assert len(df_out) == EXPECTED_ROWS, (
        f"행 수 불일치: {len(df_out)} (기대 {EXPECTED_ROWS})"
    )
    print(f"[OK] shape: {df_out.shape}  (기대 {EXPECTED_ROWS} 행)")

    # 2. 결측 검증
    for col in ("we_before", "we_after", "runs_scored"):
        n_nan = int(df_out[col].isna().sum())
        assert n_nan == 0, f"{col} 결측 {n_nan}건"
        print(f"[OK] {col} 결측 0건")

    # 3. runs_scored 분포
    rs = df_out["runs_scored"]
    neg_cnt = int((rs < 0).sum())
    print(
        f"\n[runs_scored] 평균={rs.mean():.4f}  중앙값={rs.median():.1f}"
        f"  최댓값={int(rs.max())}  음수 보정={neg_cnt}건"
    )

    # 4. inning_ended 카운트
    n_ended = int(df_out["inning_ended"].sum())
    n_groups = df_out.groupby(["game_id", "inning", "home_or_away"]).ngroups
    assert n_ended == n_groups, (
        f"inning_ended 카운트 불일치: ended={n_ended}, groups={n_groups}"
    )
    print(f"[OK] inning_ended=True: {n_ended}건 == 그룹 수 {n_groups}건")

    # 5. WE 범위
    for col in ("we_before", "we_after"):
        in_range = df_out[col].between(0.0, 1.0).all()
        assert in_range, f"{col}이 [0,1] 범위를 벗어나는 값 존재"
        print(f"[OK] {col} 모두 [0, 1] 이내")

    # 6. 상위 5개 PA 샘플
    sample_cols = [
        "game_id", "inning", "half", "score_diff_attacker",
        "base_state", "out_count", "runs_scored",
        "we_before", "we_after", "inning_ended",
    ]
    print(f"\n[상위 5개 PA 샘플]")
    print(df_out[sample_cols].head(5).to_string(index=False))

    # 7. data_quality_flag 카운트
    n_flag = int((df_out["data_quality_flag"] == "inning1_nonzero_start").sum())
    print(f"\n[data_quality_flag] inning1_nonzero_start: {n_flag}건 (참고: 태스크 기대치 ~55건)")

    print(f"\n모든 sanity check 통과!")
    print(SEP)
