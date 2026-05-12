"""
RE/WE 룩업 모듈 — 문형우·우용태·신양우 (2016) KBO 논문 이식

논문: "한국 프로야구 경기에서 기대득점과 기대승리확률의 계산"
      The Korean Journal of Applied Statistics, 29(2), 321-330.
      DOI: 10.5351/KJAS.2016.29.2.321

공개 인터페이스:
    get_re(out_count, base_state)              -> float  (RE 조회)
    get_we(inning, half, score_diff, ...)      -> float  (WE 조회, 보간 포함)
    get_we_with_boundary(inning, half, ...)    -> float  (경계 조건 포함)

주의: WE 함수의 score_diff 는 *공격팀 관점* (공격팀 득점 - 수비팀 득점).
      hsk_pa.parquet 의 score_diff 는 홈팀 기준이므로 Phase 2에서 부호 변환 필요.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  타입 별칭 / 상수
# ─────────────────────────────────────────────────────────────────────────────

Half = Literal["top", "bot"]

SCORE_DIFF_MIN: int = -4
SCORE_DIFF_MAX: int = 4

_INN_ANCHOR_LOW: int = 3     # WE 보간 기준 이닝 (하한)
_INN_ANCHOR_HIGH: int = 7    # WE 보간 기준 이닝 (상한)

BASE_STATES: tuple[str, ...] = ("0", "1", "2", "3", "12", "13", "23", "123")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: RE 테이블 (24-state)  —  논문 Table 2.1 MC 컬럼 (p.325)
# ─────────────────────────────────────────────────────────────────────────────

RE_TABLE: dict[tuple[int, str], float] = {
    # 주자 키: "0"=주자없음 "1"=1루 "2"=2루 "3"=3루
    #          "12"=1·2루  "13"=1·3루  "23"=2·3루  "123"=만루
    # ── 0아웃 ─────────────────────────────────────────────────────────────
    (0, "0"):   0.549,
    (0, "1"):   1.042,
    (0, "2"):   1.258,
    (0, "3"):   1.536,
    (0, "12"):  1.756,
    (0, "13"):  2.053,
    (0, "23"):  2.225,
    (0, "123"): 2.734,
    # ── 1아웃 ─────────────────────────────────────────────────────────────
    (1, "0"):   0.267,
    (1, "1"):   0.556,
    (1, "2"):   0.700,
    (1, "3"):   1.042,
    (1, "12"):  0.972,
    (1, "13"):  1.342,
    (1, "23"):  1.490,
    (1, "123"): 1.662,
    # ── 2아웃 ─────────────────────────────────────────────────────────────
    (2, "0"):   0.095,
    (2, "1"):   0.215,
    (2, "2"):   0.316,
    (2, "3"):   0.360,
    (2, "12"):  0.439,
    (2, "13"):  0.494,
    (2, "23"):  0.581,
    (2, "123"): 0.729,
}


def get_re(out_count: int, base_state: str) -> float:
    """
    (아웃카운트, 주자상태) → 기대득점(RE) 반환.
    3아웃 이상(이닝 종료)은 0.0.
    출처: 논문 Table 2.1 MC 컬럼 (p.325).
    """
    if out_count >= 3:
        return 0.0
    return RE_TABLE[(out_count, base_state)]


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: WE 테이블  —  논문 Table 3.2 (1사 상태, p.327)
#
#  구조: {base_state: [WE at j=-4, -3, -2, -1, 0, +1, +2, +3, +4]}
#  의미: 공격팀(현재 이닝 공격 중인 팀)의 승리확률 P(W_attack)
#  아웃 차원: Table 3.2는 1사만 제공 → Option A (plan2.md §1-2)로 전 아웃에 동일 적용
# ─────────────────────────────────────────────────────────────────────────────

# ── 3회초 1사  (Table 3.2, p.327) ────────────────────────────────────────
_WE_INN3_TOP: dict[str, list[float]] = {
    "0":   [0.078, 0.134, 0.216, 0.313, 0.449, 0.613, 0.707, 0.825, 0.896],
    "1":   [0.100, 0.162, 0.250, 0.353, 0.486, 0.641, 0.732, 0.840, 0.904],
    "2":   [0.108, 0.174, 0.264, 0.372, 0.510, 0.654, 0.749, 0.850, 0.908],
    "3":   [0.127, 0.202, 0.297, 0.418, 0.567, 0.686, 0.790, 0.874, 0.918],
    "12":  [0.133, 0.205, 0.300, 0.407, 0.538, 0.677, 0.764, 0.859, 0.915],
    "13":  [0.155, 0.236, 0.336, 0.457, 0.597, 0.711, 0.807, 0.884, 0.926],
    "23":  [0.169, 0.252, 0.358, 0.481, 0.606, 0.728, 0.813, 0.886, 0.929],
    "123": [0.194, 0.278, 0.378, 0.491, 0.616, 0.731, 0.812, 0.888, 0.931],
}

# ── 3회말 1사  (Table 3.2, p.327) ────────────────────────────────────────
_WE_INN3_BOT: dict[str, list[float]] = {
    "0":   [0.123, 0.164, 0.277, 0.374, 0.518, 0.670, 0.763, 0.807, 0.866],
    "1":   [0.145, 0.196, 0.311, 0.413, 0.551, 0.691, 0.777, 0.823, 0.880],
    "2":   [0.151, 0.213, 0.325, 0.433, 0.573, 0.704, 0.784, 0.831, 0.888],
    "3":   [0.165, 0.251, 0.358, 0.483, 0.625, 0.736, 0.798, 0.851, 0.908],
    "12":  [0.179, 0.245, 0.360, 0.465, 0.594, 0.719, 0.790, 0.844, 0.897],
    "13":  [0.196, 0.287, 0.396, 0.517, 0.649, 0.752, 0.814, 0.866, 0.918],
    "23":  [0.217, 0.301, 0.419, 0.538, 0.658, 0.756, 0.823, 0.873, 0.925],
    "123": [0.240, 0.324, 0.435, 0.542, 0.661, 0.764, 0.830, 0.876, 0.923],
}

# ── 7회초 1사  (Table 3.2, p.327) ────────────────────────────────────────
_WE_INN7_TOP: dict[str, list[float]] = {
    "0":   [0.026, 0.067, 0.119, 0.207, 0.426, 0.682, 0.813, 0.939, 0.972],
    "1":   [0.044, 0.093, 0.162, 0.269, 0.480, 0.714, 0.835, 0.945, 0.975],
    "2":   [0.050, 0.101, 0.175, 0.300, 0.516, 0.732, 0.853, 0.950, 0.976],
    "3":   [0.064, 0.119, 0.205, 0.374, 0.604, 0.777, 0.897, 0.961, 0.979],
    "12":  [0.074, 0.140, 0.228, 0.349, 0.551, 0.753, 0.862, 0.953, 0.978],
    "13":  [0.091, 0.161, 0.263, 0.429, 0.642, 0.801, 0.907, 0.965, 0.982],
    "23":  [0.102, 0.179, 0.301, 0.465, 0.652, 0.816, 0.905, 0.965, 0.983],
    "123": [0.139, 0.224, 0.329, 0.471, 0.653, 0.810, 0.902, 0.965, 0.983],
}

# ── 7회말 1사  (Table 3.2, p.327) ────────────────────────────────────────
_WE_INN7_BOT: dict[str, list[float]] = {
    "0":   [0.046, 0.077, 0.167, 0.266, 0.512, 0.795, 0.891, 0.958, 0.984],
    "1":   [0.066, 0.111, 0.214, 0.332, 0.565, 0.816, 0.904, 0.963, 0.986],
    "2":   [0.071, 0.124, 0.229, 0.366, 0.605, 0.830, 0.914, 0.967, 0.988],
    "3":   [0.081, 0.155, 0.262, 0.450, 0.703, 0.863, 0.937, 0.976, 0.991],
    "12":  [0.101, 0.168, 0.285, 0.415, 0.631, 0.843, 0.920, 0.969, 0.989],
    "13":  [0.114, 0.203, 0.324, 0.505, 0.732, 0.877, 0.943, 0.979, 0.992],
    "23":  [0.134, 0.220, 0.366, 0.543, 0.734, 0.884, 0.944, 0.979, 0.992],
    "123": [0.172, 0.266, 0.390, 0.538, 0.729, 0.881, 0.942, 0.978, 0.992],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: 보간 및 조회 함수
# ─────────────────────────────────────────────────────────────────────────────

def _clip_score_diff(j: int) -> int:
    """점수차를 [SCORE_DIFF_MIN, SCORE_DIFF_MAX] 범위로 clip."""
    return int(np.clip(j, SCORE_DIFF_MIN, SCORE_DIFF_MAX))


def _interpolate_inning(inning: int, half: Half) -> tuple[float, float]:
    """
    (이닝, 초/말) → (w3, w7) 선형 보간 가중치 반환.
    WE = w3 * WE_3회 + w7 * WE_7회

    - 1~3회  : (1.0, 0.0) — 3회 값 외삽 없이 고정
    - 3 < i < 7 : 선형 보간
    - 7~12회 : (0.0, 1.0) — 7회 값 외삽 없이 고정
    """
    if inning <= _INN_ANCHOR_LOW:
        return (1.0, 0.0)
    if inning >= _INN_ANCHOR_HIGH:
        return (0.0, 1.0)
    w7 = (inning - _INN_ANCHOR_LOW) / (_INN_ANCHOR_HIGH - _INN_ANCHOR_LOW)
    return (1.0 - w7, w7)


def get_we(
    inning: int,
    half: Half,
    score_diff: int,
    out_count: int,
    base_state: str,
) -> float:
    """
    공격팀 관점의 기대승리확률(WE) 반환.

    Option A (plan2.md §1-2): 아웃 차원은 1사 값으로 고정.
    이닝은 3회·7회 기준 선형 보간, 점수차는 ±4 clip.

    Parameters
    ----------
    inning     : 현재 이닝 (1~12)
    half       : "top" = 초(원정팀 공격) / "bot" = 말(홈팀 공격)
    score_diff : 공격팀 기준 점수차 (공격팀 득점 − 수비팀 득점)
    out_count  : 아웃카운트 0/1/2 (Option A에서는 무시됨)
    base_state : 주자상태 문자열 ("0" | "1" | ... | "123")

    Returns
    -------
    float : WE in [0, 1]
    """
    j_idx = _clip_score_diff(score_diff) - SCORE_DIFF_MIN  # -4→0, 0→4, +4→8

    w3, w7 = _interpolate_inning(inning, half)
    table3 = _WE_INN3_TOP if half == "top" else _WE_INN3_BOT
    table7 = _WE_INN7_TOP if half == "top" else _WE_INN7_BOT

    we3 = table3[base_state][j_idx]
    we7 = table7[base_state][j_idx]

    return float(w3 * we3 + w7 * we7)


def get_we_with_boundary(
    inning: int,
    half: Half,
    score_diff: int,
    out_count: int,
    base_state: str,
) -> float:
    """
    경계 조건을 포함한 WE 조회.

    - 9회말 끝내기: half='bot', score_diff > 0 → WE = 1.0
    - 12회 이상 종료 (out_count >= 3):
        score_diff > 0 → 1.0 / score_diff < 0 → 0.0 / 동점 → 0.5 (무승부)
    - 그 외: get_we() 위임
    """
    if inning == 9 and half == "bot" and score_diff > 0:
        return 1.0
    if inning >= 12 and out_count >= 3:
        if score_diff > 0:
            return 1.0
        if score_diff < 0:
            return 0.0
        return 0.5
    return get_we(inning, half, score_diff, out_count, base_state)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: 단위 테스트 (sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def _test_re_all_cells() -> None:
    """RE 테이블 24개 셀 전체 조회 및 3아웃 분기 확인."""
    for out in range(3):
        for base in BASE_STATES:
            val = get_re(out, base)
            assert isinstance(val, float), f"RE({out},{base}) is not float"
            assert val >= 0.0, f"RE({out},{base}) = {val} < 0"
    assert get_re(3, "0") == 0.0, "3아웃 RE는 0.0이어야 함"
    assert get_re(4, "123") == 0.0, "4아웃(연장 이닝 종료) RE도 0.0이어야 함"
    print("[PASS] test_re_all_cells — 24개 셀 + 3아웃 분기 정상")


def test_we_monotonic_score() -> None:
    """같은 (이닝, 아웃, 주자)에서 점수차가 클수록 WE 증가 (논문 조건 W1)."""
    for half in ("top", "bot"):
        for base in BASE_STATES:
            prev = get_we(5, half, SCORE_DIFF_MIN, 1, base)
            for j in range(SCORE_DIFF_MIN + 1, SCORE_DIFF_MAX + 1):
                cur = get_we(5, half, j, 1, base)
                assert prev <= cur, (
                    f"단조성 위반: get_we(5,{half},{j-1},1,{base!r})={prev:.4f}"
                    f" > get_we(5,{half},{j},1,{base!r})={cur:.4f}"
                )
                prev = cur
    print("[PASS] test_we_monotonic_score — 전 주자상태 × 초말 단조성 확인")


def test_we_monotonic_inning_when_ahead() -> None:
    """앞서고 있을 때 이닝이 진행될수록 WE가 커야 함 (논문 조건 W3)."""
    for half in ("top", "bot"):
        for base in BASE_STATES:
            we_3rd = get_we(3, half, +1, 1, base)
            we_7th = get_we(7, half, +1, 1, base)
            assert we_3rd <= we_7th, (
                f"이닝 단조성 위반 ({half},{base!r}): "
                f"WE(3회)={we_3rd:.4f} > WE(7회)={we_7th:.4f}"
            )
    print("[PASS] test_we_monotonic_inning_when_ahead — 전 주자상태 × 초말 확인")


def test_we_bounds() -> None:
    """모든 (이닝, 초말, 점수차, 아웃, 주자) 조합에서 WE ∈ [0, 1]."""
    for inning in range(1, 13):
        for half in ("top", "bot"):
            for j in range(SCORE_DIFF_MIN, SCORE_DIFF_MAX + 1):
                for out in range(3):
                    for base in BASE_STATES:
                        val = get_we(inning, half, j, out, base)
                        assert 0.0 <= val <= 1.0, (
                            f"범위 위반: get_we({inning},{half},{j},{out},{base!r})"
                            f" = {val:.4f}"
                        )
    print("[PASS] test_we_bounds — 전 입력 조합 [0,1] 범위 확인")


def _test_home_advantage() -> None:
    """동점 상황에서 말(홈팀 공격) WE > 초(원정팀 공격) WE (홈어드밴티지)."""
    for inning in (3, 5, 7):
        we_top = get_we(inning, "top", 0, 1, "0")
        we_bot = get_we(inning, "bot", 0, 1, "0")
        assert we_bot > we_top, (
            f"홈어드밴티지 위반 ({inning}회): WE_bot={we_bot:.4f} <= WE_top={we_top:.4f}"
        )
    print("[PASS] test_home_advantage — 동점 상황 홈어드밴티지 확인")


def _test_boundary_conditions() -> None:
    """get_we_with_boundary 경계 조건 확인."""
    assert get_we_with_boundary(9, "bot", 1, 1, "0") == 1.0, "9회말 끝내기 WE=1.0"
    assert get_we_with_boundary(12, "bot", 0, 3, "0") == 0.5, "12회 동점 종료 WE=0.5"
    assert get_we_with_boundary(12, "top", 2, 3, "0") == 1.0, "12회 공격팀 +2 WE=1.0"
    assert get_we_with_boundary(12, "bot", -1, 3, "0") == 0.0, "12회 공격팀 -1 WE=0.0"
    print("[PASS] test_boundary_conditions — 경계 조건 4종 확인")


if __name__ == "__main__":
    print("=" * 55)
    print("RE/WE 룩업 모듈 sanity check")
    print("출처: 문형우 외(2016) Table 2.1·3.2")
    print("=" * 55)

    _test_re_all_cells()
    test_we_monotonic_score()
    test_we_monotonic_inning_when_ahead()
    test_we_bounds()
    _test_home_advantage()
    _test_boundary_conditions()

    print("=" * 55)
    print("모든 테스트 통과!")
    print("=" * 55)

    # ── 대표값 출력 (육안 확인용) ──────────────────────────────────────
    print("\n[RE 대표값]")
    for out in range(3):
        print(f"  {out}아웃 주자없음: RE = {get_re(out, '0'):.3f}")

    print("\n[WE 대표값] 주자없음, 각 이닝 초·말, 점수차 0/+1/-1")
    for inn in (3, 5, 7):
        for half in ("top", "bot"):
            label = f"{inn}회{'초' if half == 'top' else '말'}"
            vals = [get_we(inn, half, j, 1, "0") for j in (-1, 0, 1)]
            print(f"  {label}  j=-1:{vals[0]:.3f}  j=0:{vals[1]:.3f}  j=+1:{vals[2]:.3f}")
