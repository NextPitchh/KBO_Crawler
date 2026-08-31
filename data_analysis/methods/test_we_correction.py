"""
data_analysis/methods/test_we_correction.py

get_we_with_out_correction (Option B, RE 차이 기반 아웃 차원 보정) 단위 테스트.

pytest 없이도 실행 가능:
    uv run python -m data_analysis.methods.test_we_correction
pytest 사용 시:
    uv run pytest data_analysis/methods/test_we_correction.py -q
"""

from __future__ import annotations

from .we_re_lookup import (
    BASE_STATES,
    SCORE_DIFF_MAX,
    SCORE_DIFF_MIN,
    get_we_with_boundary,
    get_we_with_out_correction,
)

_INNINGS = tuple(range(1, 13))
_HALVES = ("top", "bot")
_SCORES = tuple(range(SCORE_DIFF_MIN - 2, SCORE_DIFF_MAX + 3))  # clip 경계 밖까지


def test_1_out1_identical_to_boundary() -> None:
    """1) out_count == 1 이면 기존 get_we_with_boundary 와 정확히 일치."""
    mism = []
    for inning in _INNINGS:
        for half in _HALVES:
            for sd in _SCORES:
                for base in BASE_STATES:
                    a = get_we_with_boundary(inning, half, sd, 1, base)
                    b = get_we_with_out_correction(inning, half, sd, 1, base)
                    if a != b:
                        mism.append((inning, half, sd, base, a, b))
    assert not mism, f"1사에서 불일치 {len(mism)}건 (예: {mism[:3]})"
    print("[PASS] test_1 — out_count=1 전 조합에서 Option A와 동일")


def test_2_more_outs_lower_we() -> None:
    """2) 같은 상태에서 아웃이 늘수록 공격팀 WE 감소 (0사 > 1사 > 2사)."""
    bad = []
    for inning in _INNINGS:
        for half in _HALVES:
            for sd in range(SCORE_DIFF_MIN, SCORE_DIFF_MAX + 1):
                for base in BASE_STATES:
                    w0 = get_we_with_out_correction(inning, half, sd, 0, base)
                    w1 = get_we_with_out_correction(inning, half, sd, 1, base)
                    w2 = get_we_with_out_correction(inning, half, sd, 2, base)
                    if not (w0 >= w1 >= w2):
                        bad.append((inning, half, sd, base, w0, w1, w2))
    assert not bad, f"아웃 단조성 위반 {len(bad)}건 (예: {bad[:3]})"
    print("[PASS] test_2 — 0사 >= 1사 >= 2사 (공격팀 WE)")


def test_3_loaded_out_effect_exceeds_empty() -> None:
    """3) 만루에서 아웃 효과가 주자없음보다 크다."""
    checked = 0
    for inning in (3, 5, 7):
        for half in _HALVES:
            for sd in (-2, -1, 0, 1, 2):
                loaded = abs(
                    get_we_with_out_correction(inning, half, sd, 0, "123")
                    - get_we_with_out_correction(inning, half, sd, 2, "123")
                )
                empty = abs(
                    get_we_with_out_correction(inning, half, sd, 0, "0")
                    - get_we_with_out_correction(inning, half, sd, 2, "0")
                )
                assert loaded > empty, (
                    f"({inning},{half},sd={sd}) 만루 아웃효과 {loaded:.4f} "
                    f"<= 주자없음 {empty:.4f}"
                )
                checked += 1
    print(f"[PASS] test_3 — 만루 아웃효과 > 주자없음 ({checked}개 상태 확인)")


def test_4_bounds() -> None:
    """4) 모든 (이닝, 초말, 점수차, 아웃, 주자) 조합에서 WE ∈ [0, 1]."""
    for inning in _INNINGS:
        for half in _HALVES:
            for sd in _SCORES:
                for out in (0, 1, 2):
                    for base in BASE_STATES:
                        v = get_we_with_out_correction(inning, half, sd, out, base)
                        assert 0.0 <= v <= 1.0, (
                            f"범위 위반 ({inning},{half},{sd},{out},{base}) = {v}"
                        )
    print("[PASS] test_4 — 전 입력 조합 [0, 1] 범위")


def test_5_walkoff_boundary_preserved() -> None:
    """5) 9회말 끝내기 경계 조건 유지 (score_diff > 0 → WE = 1.0)."""
    for sd in (1, 2, 5):
        for out in (0, 1, 2):
            for base in BASE_STATES:
                assert get_we_with_out_correction(9, "bot", sd, out, base) == 1.0, (
                    f"9회말 끝내기 위반 (sd={sd}, out={out}, base={base})"
                )
    print("[PASS] test_5 — 9회말 끝내기 WE = 1.0 유지")


def test_6_extra_inning_tie_half() -> None:
    """6) 연장 진입(동점) 종료 시 WE = 0.5 유지, 승/패는 1.0/0.0."""
    for half in _HALVES:
        assert get_we_with_out_correction(12, half, 0, 3, "0") == 0.5
        assert get_we_with_out_correction(12, half, 3, 3, "0") == 1.0
        assert get_we_with_out_correction(12, half, -3, 3, "0") == 0.0
    print("[PASS] test_6 — 연장 종료 동점 0.5 / 승 1.0 / 패 0.0")


_ALL_TESTS = [
    test_1_out1_identical_to_boundary,
    test_2_more_outs_lower_we,
    test_3_loaded_out_effect_exceeds_empty,
    test_4_bounds,
    test_5_walkoff_boundary_preserved,
    test_6_extra_inning_tie_half,
]


if __name__ == "__main__":
    print("=" * 60)
    print("get_we_with_out_correction 단위 테스트 (6항목)")
    print("=" * 60)
    for t in _ALL_TESTS:
        t()
    print("=" * 60)
    print("6항목 모두 통과!")
    print("=" * 60)
