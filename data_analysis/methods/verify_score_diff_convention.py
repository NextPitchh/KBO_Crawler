"""
score_diff 컨벤션 검증 스크립트
hsk_pa.parquet의 score_diff 부호 컨벤션을 데이터로 확정한다.
데이터 변환 없음 — 읽기/출력 전용.

실행: uv run python data_analysis/methods/verify_score_diff_convention.py
"""

import sys
import os

import pandas as pd

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data_analysis", "results", "hsk_pa.parquet")

SEP = "─" * 65


def load() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    # 주자상태 문자열 파생 (검증용)
    df["base_state"] = (
        df["is_base1"].astype(str)
        + df["is_base2"].astype(str)
        + df["is_base3"].astype(str)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: 단일 경기 흐름 추적
# ─────────────────────────────────────────────────────────────────────────────

def step1_single_game(df: pd.DataFrame) -> None:
    print(SEP)
    print("[Step 1] 샘플 경기 흐름")
    print(SEP)

    # score_diff 변화가 많은(역동적인) 경기를 골라야 패턴이 잘 보임
    game_id = (
        df.groupby("game_id")["score_diff"]
        .apply(lambda s: s.max() - s.min())
        .idxmax()
    )
    print(f"선택 game_id: {game_id}")

    g = (
        df[df["game_id"] == game_id]
        .sort_values(["inning", "home_or_away", "out_count"])
        .reset_index(drop=True)
    )

    cols = ["inning", "home_or_away", "out_count", "base_state",
            "score_diff", "pa_result"]

    if len(g) > 60:
        display = pd.concat([g[cols].head(30), g[cols].tail(30)])
        print(f"(총 {len(g)}타석 — 앞 30 + 뒤 30 표시)")
    else:
        display = g[cols]

    print(display.to_string(index=True))
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: 가설 4종 위반 카운트
# ─────────────────────────────────────────────────────────────────────────────

def _count_monotone_violations(
    df: pd.DataFrame,
    group_cols: list[str],
    direction: str,   # "inc" | "dec"
) -> int:
    """
    그룹 내 score_diff가 direction 방향의 단조성을 위반하는 그룹 수를 반환.
    - "inc": diff >= 0 이어야 함 (절대 줄지 않음)
    - "dec": diff <= 0 이어야 함 (절대 늘지 않음)
    """
    violations = 0
    for _, g in df.groupby(group_cols, sort=False):
        if len(g) < 2:
            continue
        diffs = g["score_diff"].diff().dropna()
        if direction == "inc" and (diffs < 0).any():
            violations += 1
        elif direction == "dec" and (diffs > 0).any():
            violations += 1
    return violations


def step2_hypothesis_check(df: pd.DataFrame) -> str:
    """
    H1: score_diff = 홈팀 − 원정팀 (절대 기준)
        → home_or_away=1 그룹(홈팀 공격)에서 단조 증가 OK
          home_or_away=0 그룹(원정팀 공격)에서 단조 감소 OK (수비팀=홈팀 득점)
    H2: score_diff = 공격팀 − 수비팀 (공격팀 기준)
        → 모든 그룹에서 단조 증가 OK
    H3: score_diff = 원정팀 − 홈팀 (절대 기준, H1 반전)
        → home_or_away=0 그룹에서 단조 증가 OK
          home_or_away=1 그룹에서 단조 감소 OK
    H4: score_diff = 수비팀 − 공격팀 (H2 반전)
        → 모든 그룹에서 단조 감소 OK
    """
    print(SEP)
    print("[Step 2] 가설 4종 위반 카운트")
    print(SEP)
    print("검증 기준: 같은 (game_id, inning, home_or_away) 그룹 내")
    print("           score_diff가 가설에 따른 단조성을 위반하는 그룹 수")
    print()

    key = ["game_id", "inning", "home_or_away"]

    home_df = df[df["home_or_away"] == 1]
    away_df = df[df["home_or_away"] == 0]

    # H1: 홈팀 공격(=1) → 득점 시 홈−원정 상승, 원정팀 공격(=0) → 득점 시 홈−원정 불변
    #     (1이닝 내 같은팀 공격만, 수비팀=상대가 득점할 이닝아님 → 단조 유지)
    #     실제로는 같은 half-inning 내에서는 득점이 쌓이면 score_diff 변화가 일정해야 함
    h1_v_home = _count_monotone_violations(home_df, key, "inc")  # 홈 공격 → 홈−원 증가
    h1_v_away = _count_monotone_violations(away_df, key, "dec")  # 원정 공격 → 홈−원 감소
    h1_v = h1_v_home + h1_v_away

    # H2: 공격팀 기준 → 모든 그룹 증가
    h2_v = _count_monotone_violations(df, key, "inc")

    # H3: 원정−홈 → 원정 공격 시 증가, 홈 공격 시 감소
    h3_v_away = _count_monotone_violations(away_df, key, "inc")
    h3_v_home = _count_monotone_violations(home_df, key, "dec")
    h3_v = h3_v_away + h3_v_home

    # H4: 수비팀−공격팀 → 모든 그룹 감소
    h4_v = _count_monotone_violations(df, key, "dec")

    total_groups = df.groupby(key).ngroups
    print(f"총 그룹 수 (game_id × inning × half): {total_groups:,}")
    print()

    results = {
        "H1 (홈-원정, 절대 기준)":      h1_v,
        "H2 (공격팀-수비팀, 상대 기준)": h2_v,
        "H3 (원정-홈, 절대 기준)":      h3_v,
        "H4 (수비팀-공격팀, 상대 기준)": h4_v,
    }

    winner = None
    for label, v in results.items():
        flag = "◀ 정답 후보" if v == 0 else ""
        print(f"  {label:36s}: 위반 {v:4d}건  {flag}")
        if v == 0:
            winner = label

    print()
    if winner:
        print(f"  ✓ 위반 0건 가설: {winner}")
    else:
        min_v = min(results.values())
        candidates = [k for k, v in results.items() if v == min_v]
        print(f"  ! 위반 0건 가설 없음 — 최소 위반 {min_v}건: {candidates}")
    print()

    # 위반 사례 샘플 (H2와 반대되는 H4 위반 예시)
    key2 = ["game_id", "inning", "home_or_away"]
    shown = 0
    print("  [H2 위반 그룹 샘플 (최대 3건)]")
    for name, g in df.groupby(key2, sort=False):
        if len(g) < 2:
            continue
        diffs = g["score_diff"].diff().dropna()
        if (diffs < 0).any():
            print(f"    game={name[0]}, inning={name[1]}, ha={name[2]}")
            print(g[["inning", "home_or_away", "score_diff", "pa_result"]]
                  .to_string(index=False))
            shown += 1
            if shown >= 3:
                break
    if shown == 0:
        print("    (위반 없음)")
    print()

    # 가설 코드 추출 (H1~H4 중)
    if winner:
        for hcode in ("H1", "H2", "H3", "H4"):
            if hcode in winner:
                return hcode
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: 1회초·1회말 시작 시 score_diff 분포
# ─────────────────────────────────────────────────────────────────────────────

def step3_first_inning_check(df: pd.DataFrame) -> None:
    print(SEP)
    print("[Step 3] 1회 시작 시 score_diff 분포")
    print(SEP)

    mask = (
        (df["inning"] == 1) &
        (df["out_count"] == 0) &
        (df["is_base1"] == 0) &
        (df["is_base2"] == 0) &
        (df["is_base3"] == 0)
    )
    first_pa = (
        df[mask]
        .sort_values(["game_id", "home_or_away"])
        .groupby(["game_id", "home_or_away"])
        .first()
        .reset_index()
    )

    print(f"1회초·말, 무사, 주자없음 첫 PA: {len(first_pa)}건")
    print()
    print("  score_diff 분포:")
    dist = first_pa["score_diff"].value_counts().sort_index()
    for val, cnt in dist.items():
        bar = "█" * min(cnt, 40)
        print(f"    score_diff = {val:+3d} : {cnt:4d}건  {bar}")
    print()

    nonzero = first_pa[first_pa["score_diff"] != 0]
    if nonzero.empty:
        print("  ✓ 모든 1회 시작 PA의 score_diff = 0 (정상)")
    else:
        print(f"  ! score_diff ≠ 0 인 케이스: {len(nonzero)}건")
        print(nonzero[["game_id", "home_or_away", "score_diff", "pa_result"]]
              .head(10).to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: home_or_away 코드값 확인
# ─────────────────────────────────────────────────────────────────────────────

def step4_home_or_away_check(df: pd.DataFrame) -> None:
    print(SEP)
    print("[Step 4] home_or_away unique 값 분포")
    print(SEP)

    dist = df["home_or_away"].value_counts().sort_index()
    print(f"  dtype : {df['home_or_away'].dtype}")
    print()
    for val, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"  home_or_away = {val} : {cnt:6,}타석  ({pct:.1f}%)")
    print()
    print("  CLAUDE.md 명세: 1=홈팀 공격(말), 0=원정팀 공격(초)")

    # 이닝 내 home_or_away 순서로 공/수 검증
    # 같은 경기, 같은 이닝에서 home_or_away=0이 먼저 나와야 정상 (초→말 순서)
    g = (
        df.groupby(["game_id", "inning"])["home_or_away"]
        .apply(list)
    )
    both_present = g[g.apply(lambda x: 0 in x and 1 in x)]
    order_ok = both_present.apply(lambda x: x.index(0) < x.index(1)).sum()
    order_fail = len(both_present) - order_ok

    print()
    print(f"  이닝 내 공/수 순서 검증 (0→1 순이어야 '초→말' 정상):")
    print(f"    home_or_away=0이 먼저인 이닝: {order_ok:,}건")
    print(f"    home_or_away=1이 먼저인 이닝: {order_fail:,}건")
    if order_fail == 0:
        print("  ✓ 0=초(원정팀 공격), 1=말(홈팀 공격) 컨벤션 확인됨")
    else:
        print("  ! 순서 위반 존재 — 컨벤션 재확인 필요")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("score_diff 컨벤션 검증 스크립트")
    print(f"데이터: {PARQUET_PATH}")
    print(SEP)

    df = load()
    print(f"로드 완료: {len(df):,}타석, {df['game_id'].nunique()}경기\n")

    step1_single_game(df)
    winning_h = step2_hypothesis_check(df)
    step3_first_inning_check(df)
    step4_home_or_away_check(df)

    print(SEP)
    print(f"[최종 결론] score_diff 컨벤션은 {winning_h}로 확정됨")
    print(SEP)


if __name__ == "__main__":
    main()
