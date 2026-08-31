"""
validate_out_correction.py

아웃 차원 보정(Option B) 검증:
  Task 4-2  pa_result 별 평균 ΔWE (보정 전/후 도메인 순서)
  Task 4-3  Telescoping 재검증 (홈팀 관점 통일 공식)
  Task 4-4  네이버 원본 WPA 대비 부호 일치율 / Spearman (2024~2025)
  Task 5    보정으로 값이 크게 바뀐 케이스 분석

실행:
    uv run python validate_out_correction.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CORR = "data_analysis/results/all_pa_enriched_corrected.parquet"
FSCORES = "data/game_ids/final_scores.csv"

EXPECTED_ORDER = ["HR", "3B", "2B", "1B", "BB", "SF", "OUT", "SO", "GDP"]


def task_4_2(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Task 4-2  pa_result 별 평균 ΔWE  (공격팀 관점)")
    print("=" * 72)
    d = df[df["data_quality_flag"] == ""]
    g = d.groupby("pa_result").agg(
        n=("reward_wpa_computed", "size"),
        dwe_A=("reward_wpa_computed", "mean"),
        dwe_B=("reward_wpa_computed_corrected", "mean"),
    )
    g = g.reindex([r for r in EXPECTED_ORDER if r in g.index])
    g["Δ(B−A)"] = g["dwe_B"] - g["dwe_A"]
    print(g.round(5).to_string())

    def is_sorted_desc(s: pd.Series) -> bool:
        v = s.to_numpy()
        return bool(np.all(np.diff(v) <= 1e-9))

    order_A = is_sorted_desc(g["dwe_A"])
    order_B = is_sorted_desc(g["dwe_B"])
    print(f"\n  기대 순서 HR>3B>2B>1B>BB>SF>OUT>SO>GDP")
    print(f"  보정 전(A) 단조 감소 만족: {order_A}")
    print(f"  보정 후(B) 단조 감소 만족: {order_B}")

    gdp_A, gdp_B = g.loc["GDP", "dwe_A"], g.loc["GDP", "dwe_B"]
    out_A, out_B = g.loc["OUT", "dwe_A"], g.loc["OUT", "dwe_B"]
    so_A, so_B = g.loc["SO", "dwe_A"], g.loc["SO", "dwe_B"]
    print(f"\n  GDP 음수폭:  {gdp_A:+.5f} → {gdp_B:+.5f}  (Δ {gdp_B-gdp_A:+.5f}, 더 음수여야 정상)")
    print(f"  OUT 음수폭:  {out_A:+.5f} → {out_B:+.5f}  (Δ {out_B-out_A:+.5f})")
    print(f"  SO  음수폭:  {so_A:+.5f} → {so_B:+.5f}  (Δ {so_B-so_A:+.5f})")
    hits = g.loc[[r for r in ["HR", "3B", "2B", "1B"] if r in g.index]]
    print(f"  안타 계열 |Δ(B−A)| 최대: {hits['Δ(B−A)'].abs().max():.5f} (작아야 정상)")
    return order_B


def _telescoping_error(df: pd.DataFrame, wb_col: str, wa_col: str) -> dict:
    """build_all_enriched.telescoping_check 와 동일한 홈팀 관점 통일 공식."""
    fs = pd.read_csv(FSCORES, dtype={"game_id": str})
    winner = {
        r["game_id"]: (1.0 if r["home_score"] > r["away_score"]
                       else 0.0 if r["home_score"] < r["away_score"] else 0.5)
        for _, r in fs.iterrows()
    }
    extras: set[str] = set()
    for year in range(2016, 2026):
        files = glob.glob(f"data/pbp_full/{year}/*.csv")
        if not files:
            continue
        raw = pd.concat(
            [pd.read_csv(f, usecols=["game_id", "inning"], low_memory=False) for f in files],
            ignore_index=True,
        )
        extras |= set(raw.loc[raw["inning"] >= 10, "game_id"].unique())

    d = df.sort_values("_orig_idx")
    is_bot = d["half"] == "bot"
    wb_home = np.where(is_bot, d[wb_col], 1 - d[wb_col])
    wa_home = np.where(is_bot, d[wa_col], 1 - d[wa_col])
    delta_home = wa_home - wb_home
    tmp = d[["game_id"]].copy()
    tmp["delta_home"] = delta_home
    tmp["wb_home"] = wb_home

    errs = []
    for gid, gg in tmp.groupby("game_id"):
        initial = gg["wb_home"].iloc[0]
        final_we = 0.5 if gid in extras else winner.get(gid)
        if final_we is None:
            continue
        errs.append(gg["delta_home"].sum() - (final_we - initial))
    errs = np.abs(np.array(errs))
    return {
        "n_games": len(errs),
        "mean_abs_err": float(errs.mean()),
        "max_abs_err": float(errs.max()),
        "n_ge_0.1": int((errs >= 0.1).sum()),
        "n_ge_0.01": int((errs >= 0.01).sum()),
    }


def task_4_3(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("Task 4-3  Telescoping 재검증 (홈팀 관점 통일)")
    print("=" * 72)
    a = _telescoping_error(df, "we_before", "we_after")
    b = _telescoping_error(df, "we_before_corrected", "we_after_corrected")
    print(f"  {'':14s}{'보정 전(A)':>16s}{'보정 후(B)':>16s}")
    for k in ("n_games", "mean_abs_err", "max_abs_err", "n_ge_0.1", "n_ge_0.01"):
        print(f"  {k:14s}{a[k]:>16.6g}{b[k]:>16.6g}")
    if b["n_ge_0.1"] > a["n_ge_0.1"] + 5:
        print("  ⚠️  오차 0.1 이상 게임이 늘었다 — 보정 로직 재검토 필요")
    else:
        print("  ✓ 오차 0.1 이상 게임 증가 없음")


def task_4_4(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("Task 4-4  네이버 원본 WPA 부호 일치율 / Spearman  (reward_wpa 존재 구간)")
    print("=" * 72)
    d = df[df["reward_wpa"].notna() & (df["data_quality_flag"] == "")].copy()
    yrs = sorted(d["game_id"].str[:4].unique())
    print(f"  비교 표본 {len(d):,} PA  (연도 {yrs[0]}~{yrs[-1]})")
    nav = d["reward_wpa"].to_numpy()

    def stats(col):
        comp = d[col].to_numpy()
        nz = (np.sign(nav) != 0) & (np.sign(comp) != 0)
        sign = float((np.sign(nav[nz]) == np.sign(comp[nz])).mean())
        rho = spearmanr(nav, comp).statistic
        return sign, float(rho), int(nz.sum())

    sA, rA, nA = stats("reward_wpa_computed")
    sB, rB, nB = stats("reward_wpa_computed_corrected")
    print(f"  {'':22s}{'부호 일치율':>14s}{'Spearman ρ':>14s}")
    print(f"  보정 전 (Option A)     {sA:>13.2%}{rA:>14.3f}")
    print(f"  보정 후 (Option B)     {sB:>13.2%}{rB:>14.3f}")
    print(f"  변화                   {sB - sA:>+13.2%}{rB - rA:>+14.3f}")
    if sB < sA - 1e-4:
        print("  ⚠️  부호 일치율이 나빠졌다 — 채택 불가 신호")
    else:
        print("  ✓ 부호 일치율 개선(또는 유지)")

    # 아웃 있는 상황만 따로
    print("\n  [주자 있는 아웃성 결과(OUT/SO/GDP, base!='0')만]")
    sub = d[d["pa_result"].isin(["OUT", "SO", "GDP"]) & (d["base_state"] != "0")]
    n = sub["reward_wpa"].to_numpy()
    for label, col in [("A", "reward_wpa_computed"), ("B", "reward_wpa_computed_corrected")]:
        c = sub[col].to_numpy()
        nz = (np.sign(n) != 0) & (np.sign(c) != 0)
        print(f"    {label}: 부호 일치 {(np.sign(n[nz])==np.sign(c[nz])).mean():.2%}  "
              f"ρ={spearmanr(n, c).statistic:.3f}  (n={len(sub):,})")


def task_5(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("Task 5  보정 영향 범위")
    print("=" * 72)
    d = df.copy()
    d["dwe_change"] = (d["reward_wpa_computed_corrected"] - d["reward_wpa_computed"]).abs()
    d["wb_change"] = (d["we_before_corrected"] - d["we_before"]).abs()

    print("\n  [ (out_count, base_state) 조합별 평균 |we_before 변화| ]")
    piv = d.pivot_table("wb_change", "base_state", "out_count", "mean").round(4)
    piv = piv.reindex(["0", "1", "2", "3", "12", "13", "23", "123"])
    print(piv.to_string())

    print("\n  [ 평균 |we_before 변화| 상위: out×base ]")
    top = (d.groupby(["out_count", "base_state"])["wb_change"]
           .agg(["mean", "size"]).sort_values("mean", ascending=False).head(8))
    print(top.round(4).to_string())

    print("\n  [ |ΔWE 변화| 상위 20 PA ]")
    cols = ["game_id", "inning", "half", "score_diff_attacker", "out_count",
            "base_state", "pa_result", "runs_scored", "reward_wpa_computed",
            "reward_wpa_computed_corrected", "dwe_change"]
    print(d.nlargest(20, "dwe_change")[cols].to_string(index=False))

    t20 = d.nlargest(20, "dwe_change")
    print(f"\n  상위 20건 out_count 분포: {t20['out_count'].value_counts().to_dict()}")
    print(f"  상위 20건 base_state 분포: {t20['base_state'].value_counts().to_dict()}")
    print(f"  상위 20건 |score_diff_attacker| 중앙값: {t20['score_diff_attacker'].abs().median()}")


def main() -> None:
    df = pd.read_parquet(CORR)
    order_B = task_4_2(df)
    task_4_3(df)
    task_4_4(df)
    task_5(df)
    print("\n" + "=" * 72)
    print("검증 완료" + ("" if order_B else "  — ⚠️ 도메인 순서 위반 확인 필요"))
    print("=" * 72)


if __name__ == "__main__":
    main()
