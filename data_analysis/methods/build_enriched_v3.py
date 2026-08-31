"""
data_analysis/methods/build_enriched_v3.py

병렬로 진행된 두 작업 산출물을 통합한다:
  - all_pa_enriched_v2.parquet  (88열, matchup_* 추가 + naver_vshra_raw rename)
  - all_pa_enriched_corrected.parquet (71열, Option B WE 보정 3열 추가)

팀 결정: **Option B(아웃 차원 보정)를 기본 학습 타겟으로 승격.**
  네이버 부호 일치율 79.0% -> 88.2%, Spearman rho 0.684 -> 0.813,
  주자 있는 아웃(OUT/SO/GDP) 68.4% -> 84.2%, 도메인 순서/telescoping 유지.

전략: 다운스트림이 `reward_wpa_computed` / `we_before` / `we_after` 이름을
참조하므로, 이름은 유지한 채 내용을 Option B로 교체하고 Option A는
`*_optionA` 접미사로 보존한다 (삭제 없음).

산출: data_analysis/results/all_pa_enriched_v3.parquet  (558,064 x 91)
원본 3개 파일은 전부 보존 (롤백 지점).

실행:
    uv run python -m data_analysis.methods.build_enriched_v3
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

RESULTS = "data_analysis/results"
V2_PATH = f"{RESULTS}/all_pa_enriched_v2.parquet"
CORR_PATH = f"{RESULTS}/all_pa_enriched_corrected.parquet"
BASELINE_PATH = f"{RESULTS}/handedness_baseline.json"
OUT_PATH = f"{RESULTS}/all_pa_enriched_v3.parquet"

# Task 2: ISO 매치업 컬럼을 대응하는 wOBA/avg 컬럼 바로 뒤에 삽입
ISO_AFTER = {
    "prior_matchup_woba_den": "prior_matchup_iso",
    "handedness_base_avg": "handedness_base_iso",
    "matchup_avg_shrunk": "matchup_iso_shrunk",
}

# Task 4: 매치업 검증 리포트의 handedness_base_iso 기대값 (소수 4자리)
ISO_REPRO_EXPECTED = {"L_L": 0.1009, "L_R": 0.1471, "R_L": 0.1305, "R_R": 0.1354}

JOIN_KEYS = ["game_id", "_orig_idx"]
CORR_COLS = ["we_before_corrected", "we_after_corrected", "reward_wpa_computed_corrected"]

# Task 1-2: Option B 승격 + Option A 보존 (동시 rename, 충돌 없음)
RENAME_MAP = {
    "reward_wpa_computed": "reward_wpa_computed_optionA",
    "reward_wpa_computed_corrected": "reward_wpa_computed",
    "we_before": "we_before_optionA",
    "we_after": "we_after_optionA",
    "we_before_corrected": "we_before",
    "we_after_corrected": "we_after",
}

EXPECTED_ORDER = ["HR", "3B", "2B", "1B", "BB", "SF", "OUT", "SO", "GDP"]


def _final_column_order(cols_after_rename: list[str]) -> list[str]:
    """*_optionA 보존 컬럼을 대응되는 기본 컬럼 바로 뒤에 배치."""
    pairs = {
        "we_before": "we_before_optionA",
        "we_after": "we_after_optionA",
        "reward_wpa_computed": "reward_wpa_computed_optionA",
    }
    movable = set(pairs.values())
    ordered: list[str] = []
    for c in cols_after_rename:
        if c in movable:
            continue  # 아래에서 짝 뒤에 삽입
        ordered.append(c)
        if c in pairs:
            ordered.append(pairs[c])
    assert sorted(ordered) == sorted(cols_after_rename), "컬럼 재배열 중 누락/중복"
    return ordered


# ── ISO 매치업 컬럼 (wOBA 매치업과 동일한 3층 구조: prior / base / shrunk) ──
def add_iso_matchup_columns(v3: pd.DataFrame) -> pd.DataFrame:
    """prior_matchup_iso / handedness_base_iso / matchup_iso_shrunk 추가.

    - 원자료(prior_matchup_2b/3b/hr, prior_matchup_ab)만 사용, ab==0 이면 NaN 유지.
    - handedness_base_iso: handedness_baseline.json 의 버킷별 iso 로 조인.
      UNK_* 버킷은 _GLOBAL iso 사용 (handedness_base_woba 의 base_woba_effective 와 동일 규칙).
    - matchup_iso_shrunk: 기존 matchup_weight(재계산 금지) 재사용.
      w = ab/(ab+10); prior_iso NaN(ab==0) 이면 w=0 이라 base 가 그대로 들어감.
    """
    with open(BASELINE_PATH, encoding="utf-8") as f:
        bl = json.load(f)["baseline"]
    gk_iso = float(bl["_GLOBAL"]["iso"])

    # 1-1) prior_matchup_iso = (2B + 2*3B + 3*HR) / AB
    ab = v3["prior_matchup_ab"].to_numpy(dtype=float)
    xb = (
        v3["prior_matchup_2b"].to_numpy(dtype=float) * 1.0
        + v3["prior_matchup_3b"].to_numpy(dtype=float) * 2.0
        + v3["prior_matchup_hr"].to_numpy(dtype=float) * 3.0
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_iso = np.where(ab > 0, xb / ab, np.nan)
    v3["prior_matchup_iso"] = prior_iso

    # 1-2) handedness_base_iso (UNK_* → _GLOBAL)
    def _base_iso(key: object) -> float:
        if isinstance(key, str) and key.startswith("UNK"):
            return gk_iso
        row = bl.get(key) if isinstance(key, str) else None
        if row is None or row.get("iso") is None:
            return gk_iso
        return float(row["iso"])

    base_iso = v3["handedness_key"].map(_base_iso).astype(float).fillna(gk_iso)
    v3["handedness_base_iso"] = base_iso.to_numpy()

    # 1-3) matchup_iso_shrunk = w * prior_iso + (1-w) * base_iso   (w = matchup_weight)
    w = v3["matchup_weight"].to_numpy(dtype=float)
    prior_iso_contrib = np.where(np.isnan(prior_iso), 0.0, prior_iso)  # ab==0 → w=0 → 기여 0
    v3["matchup_iso_shrunk"] = w * prior_iso_contrib + (1.0 - w) * base_iso.to_numpy()
    return v3


def _iso_column_order(cols: list[str]) -> list[str]:
    """ISO 신규 3열을 대응 컬럼 바로 뒤로 이동, 나머지 순서 불변."""
    movable = set(ISO_AFTER.values())
    ordered: list[str] = []
    for c in cols:
        if c in movable:
            continue
        ordered.append(c)
        if c in ISO_AFTER:
            ordered.append(ISO_AFTER[c])
    assert sorted(ordered) == sorted(cols), "ISO 컬럼 재배열 중 누락/중복"
    return ordered


def validate_iso(v3: pd.DataFrame, v3_91: pd.DataFrame) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    checks.append(("1. 행 수 558,064 유지", len(v3) == 558_064, f"{len(v3):,}"))
    checks.append(("2. 열 수 94", v3.shape[1] == 94, f"{v3.shape[1]}열"))

    try:
        pd.testing.assert_frame_equal(
            v3[list(v3_91.columns)].reset_index(drop=True),
            v3_91.reset_index(drop=True),
            check_dtype=True,
        )
        checks.append(("3. 기존 91개 컬럼 값 전수 일치", True, "assert_frame_equal 통과"))
    except AssertionError as exc:
        checks.append(("3. 기존 91개 컬럼 값 전수 일치", False, str(exc)[:200]))

    m_iso = float(v3["prior_matchup_iso"].isna().mean() * 100)
    m_avg = float(v3["prior_matchup_avg"].isna().mean() * 100)
    checks.append(("4. prior_matchup_iso 결측률 == prior_matchup_avg 결측률",
                   abs(m_iso - m_avg) < 1e-9, f"iso {m_iso:.4f}% / avg {m_avg:.4f}%"))

    n_shrunk_na = int(v3["matchup_iso_shrunk"].isna().sum())
    checks.append(("5. matchup_iso_shrunk 결측 0건", n_shrunk_na == 0, f"{n_shrunk_na}건"))

    pi = v3["prior_matchup_iso"].to_numpy()
    si = v3["matchup_iso_shrunk"].to_numpy()
    pi_bad = int(np.sum(~np.isnan(pi) & ((pi < 0) | (pi > 3))))
    si_bad = int(np.sum(np.isnan(si) | (si < 0) | (si > 3)))
    checks.append(("6. prior_matchup_iso ∈ [0,3] 범위 밖 0건", pi_bad == 0, f"{pi_bad}건"))
    checks.append(("6. matchup_iso_shrunk ∈ [0,3] 범위 밖 0건", si_bad == 0, f"{si_bad}건"))

    seg = v3.loc[v3["prior_matchup_ab"] >= 20, "prior_matchup_iso"]
    med = float(seg.median())
    checks.append(("7. ab>=20 prior_matchup_iso 중앙값 ∈ [0.05,0.25]",
                   0.05 <= med <= 0.25, f"median={med:.4f}  (n={len(seg):,})"))
    return checks


def main() -> None:
    base = pd.read_parquet(V2_PATH)
    corr = pd.read_parquet(CORR_PATH)
    logger.info("v2 로드: %d행 %d열 / corrected 로드: %d행 %d열",
                *base.shape, *corr.shape)

    v2_cols = list(base.columns)
    v2_reward_optionA = base["reward_wpa_computed"].copy()

    # ── Task 1-1: 통합 ────────────────────────────────────────────────────
    for df, name in ((base, "v2"), (corr, "corrected")):
        n_dup = df.duplicated(JOIN_KEYS).sum()
        if n_dup:
            raise ValueError(f"{name}에 (game_id,_orig_idx) 중복 {n_dup}건")

    v3 = base.merge(
        corr[JOIN_KEYS + CORR_COLS],
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    logger.info("병합 완료: %d행 %d열", *v3.shape)

    # ── Task 1-3 검증 (rename 전) ────────────────────────────────────────
    checks: list[tuple[str, bool, str]] = []

    checks.append(("1. 행 수 558,064 유지", len(v3) == 558_064, f"{len(v3):,}"))

    n_join_nan = int(v3[CORR_COLS].isna().any(axis=1).sum())
    checks.append(("3. 조인 실패(NaN) 0건", n_join_nan == 0,
                   f"CORR_COLS NaN 행 {n_join_nan}건"))

    try:
        pd.testing.assert_frame_equal(
            v3[v2_cols].reset_index(drop=True),
            base[v2_cols].reset_index(drop=True),
            check_dtype=True,
        )
        checks.append(("4. v2 88개 컬럼 값 전수 일치", True, "assert_frame_equal 통과"))
    except AssertionError as exc:
        checks.append(("4. v2 88개 컬럼 값 전수 일치", False, str(exc)[:200]))

    n_key_dup = int(v3.duplicated(JOIN_KEYS).sum())
    checks.append(("7. (game_id,_orig_idx) 중복 0건", n_key_dup == 0, f"{n_key_dup}건"))

    # ── Task 1-2: rename ────────────────────────────────────────────────
    v3 = v3.rename(columns=RENAME_MAP)
    v3 = v3[_final_column_order(list(v3.columns))]

    checks.append(("2. 열 수 91", v3.shape[1] == 91, f"{v3.shape[1]}열"))

    eq_b = v3["reward_wpa_computed"].equals(corr["reward_wpa_computed_corrected"])
    checks.append(("5. reward_wpa_computed == corrected 원본", eq_b,
                   "전수 일치" if eq_b else "불일치"))

    eq_a = v3["reward_wpa_computed_optionA"].equals(v2_reward_optionA)
    checks.append(("6. reward_wpa_computed_optionA == v2 원본", eq_a,
                   "전수 일치" if eq_a else "불일치"))

    # WE 컬럼도 확인 (보너스)
    corr_indexed = corr.set_index(JOIN_KEYS)
    v3_indexed = v3.set_index(JOIN_KEYS)
    for new_col, src_col in (("we_before", "we_before_corrected"),
                             ("we_after", "we_after_corrected")):
        ok = v3_indexed[new_col].equals(corr_indexed[src_col])
        checks.append((f"   +{new_col} == corr.{src_col}", ok,
                       "일치" if ok else "불일치"))

    # ── 결과 표 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Task 1-3 검증")
    print("=" * 72)
    all_pass = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
        all_pass &= ok
    if not all_pass:
        raise SystemExit("검증 실패 — 저장 중단")

    # ── Task 1-4: 도메인 재확인 ─────────────────────────────────────────
    d = v3[v3["data_quality_flag"] == ""]
    dom = d.groupby("pa_result").agg(
        n=("reward_wpa_computed", "size"),
        dwe_optionA=("reward_wpa_computed_optionA", "mean"),
        dwe_optionB=("reward_wpa_computed", "mean"),
    ).reindex([r for r in EXPECTED_ORDER if r in d["pa_result"].unique()])
    print("\n" + "=" * 72)
    print("Task 1-4  pa_result별 평균 ΔWE  (Option A vs B, data_quality_flag=='' )")
    print("=" * 72)
    print(dom.round(5).to_string())
    b = dom["dwe_optionB"].to_numpy()
    order_ok = bool((b[:-1] - b[1:] >= -1e-9).all())
    print(f"\n  기대 순서 HR>3B>2B>1B>BB>SF>0>OUT>SO>GDP  —  Option B 단조 감소: {order_ok}")
    print(f"  GDP(B) = {dom.loc['GDP','dwe_optionB']:+.5f}  (리포트 −0.055 대조)")
    print(f"  SF (B) = {dom.loc['SF','dwe_optionB']:+.5f}  (리포트 +0.010 대조)")
    if not order_ok:
        raise SystemExit("도메인 순서 위반 — 저장 중단")

    # ── ISO 매치업 컬럼 추가 (91 → 94열) ───────────────────────────────
    v3_91 = v3.copy()  # rename/재배열 완료된 기존 91열 스냅샷 (Task 3-3 대조 기준)
    v3 = add_iso_matchup_columns(v3)
    v3 = v3[_iso_column_order(list(v3.columns))]

    iso_checks = validate_iso(v3, v3_91)
    print("\n" + "=" * 72)
    print("ISO 매치업 컬럼 검증 (Task 3)")
    print("=" * 72)
    iso_pass = True
    for name, ok, detail in iso_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
        iso_pass &= ok

    # ── Task 4: handedness_base_iso 재현 검증 ─────────────────────────
    print("\n" + "=" * 72)
    print("Task 4  handedness_base_iso 재현 검증 (리포트 수치 대조)")
    print("=" * 72)
    rep = v3.groupby("handedness_key")["handedness_base_iso"].first()
    print("  | key | 기대(리포트) | 실제 | Δ |")
    print("  |-----|-------------:|-----:|--:|")
    repro_ok = True
    for k, exp in ISO_REPRO_EXPECTED.items():
        got = float(rep[k])
        d = got - exp
        ok = abs(d) < 5e-4
        repro_ok &= ok
        print(f"  | {k} | {exp:.4f} | {got:.4f} | {d:+.5f} {'OK' if ok else 'MISMATCH'} |")
    iso_pass &= repro_ok
    if not repro_ok:
        print("  → 불일치: handedness_baseline.json 조인 문제")

    # ── prior_matchup_iso 분포 리포트 ────────────────────────────────
    pi_ser = v3["prior_matchup_iso"]
    seg20 = v3.loc[v3["prior_matchup_ab"] >= 20, "prior_matchup_iso"]
    print("\n" + "=" * 72)
    print("prior_matchup_iso 분포")
    print("=" * 72)
    print(f"  결측률           : {pi_ser.isna().mean()*100:.4f}%  ({int(pi_ser.isna().sum()):,} 행, ab==0)")
    print(f"  전체 중앙값(실측): {pi_ser.median():.5f}")
    print(f"  ab>=20 구간      : n={len(seg20):,}  중앙값={seg20.median():.5f}  "
          f"평균={seg20.mean():.5f}  (리그 ISO≈0.133)")

    if not iso_pass:
        raise SystemExit("ISO 컬럼 검증 실패 — 저장 중단")

    # ── Task 1-5: 저장 ─────────────────────────────────────────────────
    v3.to_parquet(OUT_PATH, index=False)
    logger.info("저장: %s (%d행 %d열)", OUT_PATH, *v3.shape)

    print("\n" + "=" * 72)
    print(f"최종 컬럼 목록 ({v3.shape[1]}개)")
    print("=" * 72)
    for i, c in enumerate(v3.columns, 1):
        print(f"  {i:2d}. {c}")


if __name__ == "__main__":
    main()
