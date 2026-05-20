"""
Phase 4 WPA Validation
- 4-1: Self sanity checks on reward_wpa_computed
- 4-2: Comparison against Naver original reward_wpa
- 4-3: 5-criterion PASS/FAIL table
- 4-4: Save wpa_validation_report.md
"""

import warnings
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PARQUET_PATH = Path("data_analysis/results/hsk_pa_with_wpa.parquet")
SCATTER_PATH = Path("data_analysis/results/wpa_validation_scatter.png")
REPORT_PATH = Path("data_analysis/results/wpa_validation_report.md")

PA_ORDER = ["HR", "3B", "2B", "1B", "SF", "BB", "IBB", "HBP", "SO", "GDP", "OUT", "UNK"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _sign_match_rate(naver: pd.Series, computed: pd.Series) -> float:
    mask = (naver != 0) & (computed != 0)
    if mask.sum() == 0:
        return float("nan")
    return (np.sign(naver[mask]) == np.sign(computed[mask])).mean()


def _pa_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pa in PA_ORDER:
        sub = df[df["pa_result"] == pa]["reward_wpa_computed"]
        if len(sub) == 0:
            continue
        rows.append({"pa_result": pa, "n": len(sub), "mean_delta_we": sub.mean()})
    return pd.DataFrame(rows)


# ── 4-1  self sanity ──────────────────────────────────────────────────────────

def validate_self(df: pd.DataFrame) -> dict:
    col = df["reward_wpa_computed"]

    missing = col.isna().sum()
    out_of_range = ((col < -1) | (col > 1)).sum()
    grand_mean = col.mean()

    pa_tbl = _pa_mean_table(df)

    # domain order check: HR > 1B > BB > 0 > OUT
    def _mean(pa):
        sub = df[df["pa_result"] == pa]["reward_wpa_computed"]
        return sub.mean() if len(sub) else float("nan")

    hr_mean = _mean("HR")
    b1_mean = _mean("1B")
    bb_mean = _mean("BB")
    out_mean = _mean("OUT")
    domain_ok = (hr_mean > b1_mean > bb_mean > 0 > out_mean)

    return {
        "missing": int(missing),
        "out_of_range": int(out_of_range),
        "grand_mean": grand_mean,
        "pa_table": pa_tbl,
        "domain_ok": domain_ok,
        "hr_mean": hr_mean,
        "b1_mean": b1_mean,
        "bb_mean": bb_mean,
        "out_mean": out_mean,
    }


# ── 4-2  vs naver ────────────────────────────────────────────────────────────

def validate_vs_naver(df: pd.DataFrame) -> dict:
    cmp = df[df["reward_wpa"].notna()].copy()
    n_cmp = len(cmp)

    # (1) season distribution
    cmp["season"] = cmp["game_id"].str[:4]
    season_dist = cmp.groupby("season").size().rename("n_pa")

    naver_desc = cmp["reward_wpa"].describe()

    # (2) sign match rate
    smr = _sign_match_rate(cmp["reward_wpa"], cmp["reward_wpa_computed"])

    # (3) spearman + pearson
    sp_rho, sp_p = stats.spearmanr(cmp["reward_wpa"], cmp["reward_wpa_computed"])
    pe_r, pe_p = stats.pearsonr(cmp["reward_wpa"], cmp["reward_wpa_computed"])

    # (4) scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cmp["reward_wpa"], cmp["reward_wpa_computed"], alpha=0.3, s=10, color="steelblue")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Naver reward_wpa (non-standard scale)")
    ax.set_ylabel("reward_wpa_computed (WE delta, [-1,+1])")
    ax.set_title(
        f"WPA Validation Scatter  n={n_cmp}\n"
        f"Sign match={smr:.1%}  Spearman rho={sp_rho:.3f}"
    )
    fig.tight_layout()
    SCATTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCATTER_PATH, dpi=150)
    plt.close(fig)

    # (5) top-10 sign mismatches by |naver| * |computed|
    mask_nonzero = (cmp["reward_wpa"] != 0) & (cmp["reward_wpa_computed"] != 0)
    mismatch = cmp[mask_nonzero & (np.sign(cmp["reward_wpa"]) != np.sign(cmp["reward_wpa_computed"]))].copy()
    mismatch["abs_product"] = mismatch["reward_wpa"].abs() * mismatch["reward_wpa_computed"].abs()
    top10_cols = ["game_id", "inning", "half", "base_state", "score_diff_attacker",
                  "pa_result", "runs_scored", "reward_wpa", "reward_wpa_computed"]
    top10 = mismatch.nlargest(10, "abs_product")[top10_cols]

    # (6) data_quality_flag breakdown
    flag_stats = {}
    for flag_val in ["inning1_nonzero_start", "high_runs_scored_artifact"]:
        sub = df[df["data_quality_flag"] == flag_val]["reward_wpa_computed"]
        if len(sub) == 0:
            flag_stats[flag_val] = None
        else:
            flag_stats[flag_val] = {"n": len(sub), "mean": sub.mean(), "std": sub.std()}

    normal_sub = df[df["data_quality_flag"] == ""]["reward_wpa_computed"]
    flag_stats["normal"] = {"n": len(normal_sub), "mean": normal_sub.mean(), "std": normal_sub.std()}

    return {
        "n_cmp": n_cmp,
        "season_dist": season_dist,
        "naver_desc": naver_desc,
        "smr": smr,
        "sp_rho": sp_rho, "sp_p": sp_p,
        "pe_r": pe_r, "pe_p": pe_p,
        "top10": top10,
        "flag_stats": flag_stats,
    }


# ── 4-3  verdict table ───────────────────────────────────────────────────────

def build_verdict(s: dict, v: dict) -> pd.DataFrame:
    rows = [
        {"#": 1, "Criterion": "Missing rate = 0",
         "Value": f"{s['missing']} missing",
         "Threshold": "0",
         "Result": "PASS" if s["missing"] == 0 else "FAIL"},
        {"#": 2, "Criterion": "Range [-1,+1] violations = 0",
         "Value": f"{s['out_of_range']} violations",
         "Threshold": "0",
         "Result": "PASS" if s["out_of_range"] == 0 else "FAIL"},
        {"#": 3, "Criterion": "Sign match rate ≥ 80%",
         "Value": f"{v['smr']:.1%}",
         "Threshold": "80%",
         "Result": "PASS" if v["smr"] >= 0.80 else "FAIL"},
        {"#": 4, "Criterion": "Spearman ρ ≥ 0.6 & p < 0.05",
         "Value": f"ρ={v['sp_rho']:.3f}, p={v['sp_p']:.3e}",
         "Threshold": "ρ≥0.6, p<0.05",
         "Result": "PASS" if (v["sp_rho"] >= 0.6 and v["sp_p"] < 0.05) else "FAIL"},
        {"#": 5, "Criterion": "pa_result mean order (HR>1B>BB>0>OUT)",
         "Value": f"HR={s['hr_mean']:.4f}, 1B={s['b1_mean']:.4f}, BB={s['bb_mean']:.4f}, OUT={s['out_mean']:.4f}",
         "Threshold": "HR>1B>BB>0>OUT",
         "Result": "PASS" if s["domain_ok"] else "FAIL"},
    ]
    return pd.DataFrame(rows)


# ── 4-4  report ──────────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    buf = StringIO()
    col_fmts = {c: "r" for c in df.columns}
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows   = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def save_report(s: dict, v: dict, verdict: pd.DataFrame) -> str:
    n_pass = (verdict["Result"] == "PASS").sum()
    smr = v["smr"]
    sp_rho = v["sp_rho"]

    if n_pass == 5:
        conclusion = "5개 항목 모두 PASS → **데이터 구축 완료. 팀원 핸드오프 가능.**"
    elif smr < 0.50 or (smr < 0.80 and sp_rho < 0.6):
        conclusion = "부호일치율 50% 근처 또는 ρ 미달 → **근본 부호 오류 잔존 의심, 조사 필요.**"
    else:
        conclusion = "부호일치율 또는 ρ가 합격선 약간 미달 → **Phase 5 진행 권장.**"

    # flag breakdown text
    flag_lines = []
    for flag_val in ["inning1_nonzero_start", "high_runs_scored_artifact"]:
        fs = v["flag_stats"][flag_val]
        if fs is None:
            flag_lines.append(f"- `{flag_val}`: **데이터 없음 (0건)**")
        else:
            flag_lines.append(
                f"- `{flag_val}` (n={fs['n']}): mean={fs['mean']:.4f}, std={fs['std']:.4f}"
            )
    ns = v["flag_stats"]["normal"]
    flag_lines.append(f"- `normal` (n={ns['n']}): mean={ns['mean']:.4f}, std={ns['std']:.4f}")

    pa_tbl_md = _df_to_md(
        s["pa_table"].assign(mean_delta_we=s["pa_table"]["mean_delta_we"].map("{:.4f}".format))
    )
    verdict_md = _df_to_md(verdict)
    top10_md = _df_to_md(v["top10"].reset_index(drop=True))
    season_md = _df_to_md(v["season_dist"].reset_index())

    report = f"""# WPA Validation Report
Generated: 2026-05-20
Parquet: `data_analysis/results/hsk_pa_with_wpa.parquet`

---

## 4-1. Self Sanity Check

| Item | Value |
|---|---|
| Missing (reward_wpa_computed) | {s['missing']} |
| Out-of-range [-1,+1] violations | {s['out_of_range']} |
| Grand mean (expected ≈ 0) | {s['grand_mean']:.6f} |
| Domain order OK (HR>1B>BB>0>OUT) | {'YES' if s['domain_ok'] else 'NO'} |

### pa_result mean ΔWE (domain order check)

{pa_tbl_md}

---

## 4-2. Comparison vs Naver Original

### (1) Sample Statistics

- **Comparable PAs**: {v['n_cmp']:,} (both reward_wpa and reward_wpa_computed non-null)
- Naver `reward_wpa` is **non-standard scale** (describe below; values up to ±50 suggest percentage-point or proprietary unit — NOT WE delta)

**Naver reward_wpa describe:**

```
{v['naver_desc'].to_string()}
```

**Season distribution of comparable PAs:**

{season_md}

### (2) Sign Match Rate

- Sign match rate (excluding exact-zero on either side): **{v['smr']:.1%}**
- Threshold: ≥ 80%  →  {'**PASS**' if v['smr'] >= 0.80 else '**FAIL**'}

### (3) Rank Correlation

| Method | Statistic | p-value |
|---|---|---|
| Spearman ρ | {v['sp_rho']:.4f} | {v['sp_p']:.3e} |
| Pearson r (reference) | {v['pe_r']:.4f} | {v['pe_p']:.3e} |

Spearman threshold: ρ ≥ 0.6, p < 0.05  →  {'**PASS**' if (v['sp_rho'] >= 0.6 and v['sp_p'] < 0.05) else '**FAIL**'}

### (4) Scatter Plot

Saved to `data_analysis/results/wpa_validation_scatter.png`

### (5) Top-10 Sign-Mismatch Cases (by |naver| × |computed|)

{top10_md}

### (6) Data Quality Flag Breakdown

{chr(10).join(flag_lines)}

---

## 4-3. 합격 기준 5종 판정

{verdict_md}

**통과: {n_pass}/5**

---

## Phase 5(Option B) 진행 여부

{conclusion}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading parquet …")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  rows={len(df):,}  cols={len(df.columns)}")

    print("\n[4-1] Self sanity …")
    s = validate_self(df)
    print(f"  Missing:        {s['missing']}")
    print(f"  Out-of-range:   {s['out_of_range']}")
    print(f"  Grand mean:     {s['grand_mean']:.6f}")
    print(f"  Domain order:   {'OK' if s['domain_ok'] else 'FAIL'}")
    print()
    print(s["pa_table"].to_string(index=False))

    print("\n[4-2] vs Naver …")
    v = validate_vs_naver(df)
    print(f"  Comparable PAs: {v['n_cmp']:,}")
    print(f"  Sign match rate:{v['smr']:.1%}")
    print(f"  Spearman rho:   {v['sp_rho']:.4f}  p={v['sp_p']:.3e}")
    print(f"  Pearson r:      {v['pe_r']:.4f}  p={v['pe_p']:.3e}")
    print(f"  Scatter saved:  {SCATTER_PATH}")

    print("\n  Season distribution:")
    print(v["season_dist"].to_string())

    print("\n  Top-10 sign mismatches:")
    print(v["top10"].to_string(index=False))

    print("\n  Data quality flag breakdown:")
    for flag_val in ["inning1_nonzero_start", "high_runs_scored_artifact"]:
        fs = v["flag_stats"][flag_val]
        if fs is None:
            print(f"    {flag_val}: not present in data (0 rows)")
        else:
            print(f"    {flag_val} n={fs['n']}: mean={fs['mean']:.4f} std={fs['std']:.4f}")
    ns = v["flag_stats"]["normal"]
    print(f"    normal n={ns['n']}: mean={ns['mean']:.4f} std={ns['std']:.4f}")

    print("\n[4-3] Verdict …")
    verdict = build_verdict(s, v)
    print(verdict.to_string(index=False))

    n_pass = (verdict["Result"] == "PASS").sum()
    print(f"\n  PASS: {n_pass}/5")

    print("\n[4-4] Saving report …")
    save_report(s, v, verdict)
    print(f"  Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
