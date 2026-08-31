"""
추가 검증: XBH 예측에서 base 지표(wOBA vs ISO) 재검토.
X_shrunk 의 개선이 실제 매치업 신호인지, 잘못된 base(wOBA)를 개별 이력이 상쇄한 것인지 구분.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

V2 = "data_analysis/results/all_pa_enriched_v2.parquet"
BASELINE = "data_analysis/results/handedness_baseline.json"
BASE_K = 10
AB_RESULTS = {"1B", "2B", "3B", "HR", "SO", "OUT", "GDP"}
PA_RESULTS = AB_RESULTS | {"BB", "SF"}

bl = json.load(open(BASELINE))["baseline"]
g_iso = bl["_GLOBAL"]["iso"]
g_woba = bl["_GLOBAL"]["woba"]

df = pd.read_parquet(V2)
r = df["pa_result"].astype(str)
df["is_pa"] = r.isin(PA_RESULTS)
df["is_ab"] = r.isin(AB_RESULTS)
df["y_xbh"] = r.isin({"2B", "3B", "HR"}).astype(int)

# handedness base 지표
def base_of(field, glob):
    return df["handedness_key"].map(
        lambda k: (bl.get(k, {}).get(f"base_{field}_effective")
                   if bl.get(k, {}).get(f"base_{field}_effective") is not None
                   else (glob if str(k).startswith("UNK") else bl.get(k, {}).get(field, glob)))
    ).astype(float).fillna(glob).to_numpy()

df["hand_iso"] = base_of("iso", g_iso)          # JSON 에 base_iso_effective 없음 → UNK=global 규칙 적용
df["hand_woba"] = df["handedness_base_woba"].to_numpy()

# 개별 이력 ISO (leakage-free, matchup_history 의 누적 카운트 사용)
ab = df["prior_matchup_ab"].to_numpy(dtype=float)
tb_xbh = (df["prior_matchup_2b"] + 2 * df["prior_matchup_3b"] + 3 * df["prior_matchup_hr"]).to_numpy(dtype=float)
with np.errstate(invalid="ignore", divide="ignore"):
    df["prior_matchup_iso"] = np.where(ab > 0, tb_xbh / ab, np.nan)

w = ab / (ab + BASE_K)
p_iso = df["prior_matchup_iso"].to_numpy()
df["matchup_iso_shrunk"] = w * np.where(np.isnan(p_iso), 0.0, p_iso) + (1 - w) * df["hand_iso"].to_numpy()
# wOBA shrunk 는 이미 파일에 있음: matchup_woba_shrunk

d = df[df["is_pa"]].sort_values("_orig_idx", kind="mergesort")
cut = int(len(d) * 0.70)
tr, te = d.iloc[:cut], d.iloc[cut:]
print(f"train={len(tr):,}  test={len(te):,}  test XBH rate={te.y_xbh.mean():.4f}\n")


def ev(cols):
    X = lambda s: np.column_stack([_col(s, c) for c in cols])
    m = LogisticRegression(max_iter=1000).fit(X(tr), tr.y_xbh.values)
    p = m.predict_proba(X(te))[:, 1]
    coefs = dict(zip(cols, np.round(m.coef_[0], 3)))
    return (roc_auc_score(te.y_xbh, p), log_loss(te.y_xbh, p, labels=[0, 1]),
            brier_score_loss(te.y_xbh, p), coefs)


def _col(s, c):
    if c == "ab":
        return np.log1p(s["prior_matchup_ab"].to_numpy(dtype=float))
    return s[c].to_numpy(dtype=float)


rows = [
    ("wOBA base",        ["hand_woba"]),
    ("wOBA shrunk",      ["matchup_woba_shrunk"]),
    ("wOBA full",        ["matchup_woba_shrunk", "ab"]),
    ("ISO base",         ["hand_iso"]),
    ("ISO shrunk",       ["matchup_iso_shrunk"]),
    ("ISO full",         ["matchup_iso_shrunk", "ab"]),
    ("wOBA+ISO base",    ["hand_woba", "hand_iso"]),
    ("wOBA+ISO base +ab",["hand_woba", "hand_iso", "ab"]),
    ("ISO base + ISO shrunk", ["hand_iso", "matchup_iso_shrunk"]),
    ("ISO base + shrunk + ab", ["hand_iso", "matchup_iso_shrunk", "ab"]),
]
print(f"{'model':28s} {'XBH AUC':>9s} {'log-loss':>10s} {'Brier':>9s}   coef")
print("-" * 78)
res = {}
for name, cols in rows:
    auc, ll, br, cf = ev(cols)
    res[name] = auc
    print(f"{name:28s} {auc:9.4f} {ll:10.5f} {br:9.5f}   {cf}")

print()
# ab 구간별: ISO base vs ISO shrunk
mb = LogisticRegression(max_iter=1000).fit(tr[["hand_iso"]].values, tr.y_xbh.values)
ms = LogisticRegression(max_iter=1000).fit(tr[["matchup_iso_shrunk"]].values, tr.y_xbh.values)
print("ab 구간별 XBH-AUC (ISO base vs ISO shrunk):")
print(f"{'bucket':>8s} {'n':>8s} {'ISO base':>9s} {'ISO shrunk':>11s} {'prior_iso mean':>15s}")
for lo, hi, lab in [(0, 0, "0"), (1, 9, "1-9"), (10, 19, "10-19"), (20, 49, "20-49"), (50, 10**9, ">=50")]:
    seg = te[(te.prior_matchup_ab >= lo) & (te.prior_matchup_ab <= hi)]
    if len(seg) < 50 or seg.y_xbh.nunique() < 2:
        print(f"{lab:>8s} {len(seg):>8,} {'-':>9s} {'-':>11s}"); continue
    ab_ = roc_auc_score(seg.y_xbh, mb.predict_proba(seg[["hand_iso"]].values)[:, 1])
    as_ = roc_auc_score(seg.y_xbh, ms.predict_proba(seg[["matchup_iso_shrunk"]].values)[:, 1])
    pim = seg["prior_matchup_iso"].mean()
    print(f"{lab:>8s} {len(seg):>8,} {ab_:9.4f} {as_:11.4f} {pim:15.4f}")

print()
# raw 상관 (test): 개별 이력 ISO 자체가 XBH 를 예측하는가 (base 제거)
sub = te[te.prior_matchup_ab >= 5]
from scipy.stats import pointbiserialr
print(f"test, ab>=5 (n={len(sub):,}): corr(prior_matchup_iso, y_xbh) = "
      f"{pointbiserialr(sub.y_xbh, sub.prior_matchup_iso.fillna(0))[0]:+.4f}")
sub10 = te[te.prior_matchup_ab >= 10]
print(f"test, ab>=10 (n={len(sub10):,}): corr(prior_matchup_iso, y_xbh) = "
      f"{pointbiserialr(sub10.y_xbh, sub10.prior_matchup_iso.fillna(0))[0]:+.4f}")
