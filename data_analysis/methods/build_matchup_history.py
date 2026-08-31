"""
투수-타자 매치업 이력 구축 (leakage-free)  —  wOBA 정렬 버전

배경: 좌우 도메인 게이트를 raw AVG 로 설계한 것이 오류였다. 좌완 투수의 플래툰
우위는 안타 억제가 아니라 장타·볼넷 억제로 나타나므로(세이버메트릭스 정설),
지표 체계 전체를 wOBA 로 정렬한다. (사유·근거 수치는 리포트에 기록)

3층 구조:
  1) base   : (pitcher_throws, batter_side) 좌우 매치업 유형별 wOBA 기댓값
  2) adjust : 개별 (pitcher_id, batter_id) 누적 상대 wOBA (expanding().shift(1))
  3) signal : 표본 크기(prior_matchup_ab) + shrinkage weight

산출물:
  data_analysis/results/handedness_baseline.json
  data_analysis/results/matchup_history.parquet        (558,064 행)
  data_analysis/results/all_pa_enriched_v2.parquet      (원본 보존, vsHra 컬럼명만 변경)
  data_analysis/results/matchup_validation_report.md
"""
from __future__ import annotations

import json
import tracemalloc
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SRC = "data_analysis/results/all_pa_enriched.parquet"
OUT_BASELINE = "data_analysis/results/handedness_baseline.json"
OUT_MATCHUP = "data_analysis/results/matchup_history.parquet"
OUT_V2 = "data_analysis/results/all_pa_enriched_v2.parquet"
OUT_REPORT = "data_analysis/results/matchup_validation_report.md"

BASE_K = 10  # 도메인 판단: 상대 10타수에서 개별 전적 50% 반영. 검증에서 5/10/20/30 비교하되 기본값 유지.

# ── wOBA 선형 가중치 (2016 wOBA scale, 팀 지정) ─────────────────────────
# HBP·IBB 는 데이터상 BB 로 병합되어 있으므로 BB 가중치에 함께 반영된다.
W_HR, W_3B, W_2B, W_1B, W_BB = 1.70, 1.37, 1.08, 0.77, 0.62
# 분모 = AB + BB + SF   (SO/OUT/GDP/SF 는 분자 기여 0)

# ── Task 1: 타수(AB) / 안타(H) 정의 (pa_result 9-class) ────────────────
AB_RESULTS = {"1B", "2B", "3B", "HR", "SO", "OUT", "GDP"}
HIT_RESULTS = {"1B", "2B", "3B", "HR"}
PA_RESULTS = AB_RESULTS | {"BB", "SF"}          # 유효 PA (UNK 제외)

REPORT: list[str] = []


def log(s: str = "") -> None:
    print(s)
    REPORT.append(s)


# ────────────────────────────────────────────────────────────────────────
def add_result_flags(df: pd.DataFrame) -> pd.DataFrame:
    r = df["pa_result"].astype(str)
    rr = df["pa_result_raw"].astype(str)
    df["is_ab"] = r.isin(AB_RESULTS)
    df["is_hit"] = r.isin(HIT_RESULTS)
    df["is_pa"] = r.isin(PA_RESULTS)
    df["is_1b"] = r.eq("1B")
    df["is_2b"] = r.eq("2B")
    df["is_3b"] = r.eq("3B")
    df["is_hr"] = r.eq("HR")
    df["is_so"] = r.eq("SO")
    df["is_bb"] = r.eq("BB")           # BB(+IBB+HBP 병합)
    df["is_sf"] = r.eq("SF")
    df["is_ibb"] = rr.eq("IBB")        # 원본 보존분에서 IBB 만 분리 (v2 wOBA 대조용)
    df["is_xbh"] = r.isin({"2B", "3B", "HR"})
    # per-PA wOBA 값 (분자 가중치). 유효 PA 기준, 그 외 0.
    lw = {"1B": W_1B, "2B": W_2B, "3B": W_3B, "HR": W_HR, "BB": W_BB}
    df["pa_woba_value"] = r.map(lw).fillna(0.0).where(df["is_pa"], np.nan)
    return df


def woba_from_counts(n1b, n2b, n3b, nhr, nbb, ab, sf, ibb=0, drop_ibb=False):
    """카운트로부터 wOBA. drop_ibb=True 면 IBB 를 분자·분모에서 제거(정식 wOBA 근사)."""
    num = W_1B * n1b + W_2B * n2b + W_3B * n3b + W_HR * nhr + W_BB * (nbb - (ibb if drop_ibb else 0))
    den = ab + (nbb - (ibb if drop_ibb else 0)) + sf
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ── Task 2: 좌우 매치업 기준값 (base) ─────────────────────────────────
def pitcher_key(x) -> str:
    if x in ("R", "L", "U"):
        return x           # U = 언더핸드(정상 값), 독립 그룹 유지
    return "UNK"            # None/NaN → 결측 그룹 (2016~2017 집중)


def build_baseline(df: pd.DataFrame):
    log("## Task 2: 좌우 매치업 기준값 (base) — wOBA 정렬\n")

    # 2-1) 스위치 타자 처리
    n_switch = int((df["batter_hit_type"] == "S").sum())
    n_bat_null = int(df["batter_hit_type"].isna().sum())
    log("### 2-1) 스위치 타자 처리")
    log(f"- batter_hit_type == 'S' : {n_switch:,} 행 ({n_switch/len(df)*100:.2f}%)")
    log(f"- batter_hit_type 결측    : {n_bat_null:,} 행")
    side_cols = [c for c in df.columns
                 if any(t in c.lower() for t in ("stand", "bat_side", "batside", "batterside"))
                 and c not in ("batter_hit_type", "pitcher_throws")]
    log(f"- 실제 타석 좌우 기록 컬럼 탐색: {side_cols if side_cols else '없음'}")
    log("- → 실측 없음. 통상 규칙: **우투수 상대 좌타석 / 좌투수 상대 우타석**.")
    log("  손 미상(U/UNK) 투수 상대 스위치 타석은 변환 불가 → 'S' 유지.")
    log("  변환 행에 `batter_side_inferred = True` 플래그.\n")

    pthrow = df["pitcher_throws"].map(pitcher_key)
    bside = df["batter_hit_type"].where(df["batter_hit_type"].notna(), "U").astype(str)
    is_sw = bside.eq("S")
    to_L = is_sw & pthrow.eq("R")
    to_R = is_sw & pthrow.eq("L")
    bside_eff = bside.copy()
    bside_eff[to_L] = "L"
    bside_eff[to_R] = "R"
    inferred = (to_L | to_R)
    n_unres = int((is_sw & ~inferred).sum())

    df["pitcher_throws_key"] = pthrow.values
    df["batter_side_effective"] = bside_eff.values
    df["batter_side_inferred"] = inferred.values
    df["handedness_key"] = (pthrow + "_" + bside_eff).values
    log(f"- 스위치 변환: 좌타석 {int(to_L.sum()):,} / 우타석 {int(to_R.sum()):,} / 미변환 {n_unres:,}\n")

    # 2-2) 조합별 집계
    d = df[df["is_pa"]]
    grp = d.groupby("handedness_key")
    agg = grp.agg(
        n_pa=("is_pa", "sum"), n_ab=("is_ab", "sum"), n_hits=("is_hit", "sum"),
        n_1b=("is_1b", "sum"), n_2b=("is_2b", "sum"), n_3b=("is_3b", "sum"),
        n_hr=("is_hr", "sum"), n_bb=("is_bb", "sum"), n_so=("is_so", "sum"),
        n_sf=("is_sf", "sum"), n_ibb=("is_ibb", "sum"),
    ).sort_index()

    agg["avg"] = agg["n_hits"] / agg["n_ab"]
    agg["woba"] = woba_from_counts(agg.n_1b, agg.n_2b, agg.n_3b, agg.n_hr, agg.n_bb, agg.n_ab, agg.n_sf)
    agg["woba_no_ibb"] = woba_from_counts(agg.n_1b, agg.n_2b, agg.n_3b, agg.n_hr, agg.n_bb,
                                          agg.n_ab, agg.n_sf, agg.n_ibb, drop_ibb=True)
    den = agg["n_ab"] + agg["n_bb"] + agg["n_sf"]
    agg["obp"] = (agg["n_hits"] + agg["n_bb"]) / den
    agg["so_rate"] = agg["n_so"] / den
    agg["bb_rate"] = agg["n_bb"] / den
    agg["hr_rate"] = agg["n_hr"] / agg["n_ab"]
    agg["iso"] = (agg["n_2b"] + 2 * agg["n_3b"] + 3 * agg["n_hr"]) / agg["n_ab"]  # ISO = SLG - AVG

    # IBB 병합 영향 확인
    max_ibb_gap = float((agg["woba"] - agg["woba_no_ibb"]).abs().max())
    log("### 2-2) 조합별 집계  (key = pitcherThrowsKey_batterSideEffective)\n")
    log("| key | n_pa | n_ab | wOBA | wOBA(no IBB) | AVG | OBP | ISO | HR% | SO% | BB% |")
    log("|-----|-----:|-----:|-----:|-----:|----:|----:|----:|----:|----:|----:|")
    for k, x in agg.iterrows():
        log(f"| {k} | {int(x.n_pa):,} | {int(x.n_ab):,} | {x.woba:.4f} | {x.woba_no_ibb:.4f} "
            f"| {x.avg:.4f} | {x.obp:.4f} | {x.iso:.4f} | {x.hr_rate:.4f} | {x.so_rate:.4f} | {x.bb_rate:.4f} |")
    log("")
    log(f"- IBB 병합 vs 제외 wOBA 최대 차이: **{max_ibb_gap:.5f}** "
        f"({'< 0.005 → 병합 버전 사용' if max_ibb_gap < 0.005 else '>= 0.005 → 검토 필요'})\n")

    # 전역 fallback
    g = df[df["is_pa"]]
    gw = float(woba_from_counts(g.is_1b.sum(), g.is_2b.sum(), g.is_3b.sum(), g.is_hr.sum(),
                                g.is_bb.sum(), g.is_ab.sum(), g.is_sf.sum()))
    global_row = {
        "n_pa": int(g.is_pa.sum()), "n_ab": int(g.is_ab.sum()), "n_hits": int(g.is_hit.sum()),
        "woba": gw, "avg": float(g.is_hit.sum() / g.is_ab.sum()),
        "obp": float((g.is_hit.sum() + g.is_bb.sum()) / (g.is_ab.sum() + g.is_bb.sum() + g.is_sf.sum())),
        "iso": float((g.is_2b.sum() + 2 * g.is_3b.sum() + 3 * g.is_hr.sum()) / g.is_ab.sum()),
        "hr_rate": float(g.is_hr.sum() / g.is_ab.sum()),
    }

    # 2-3) 도메인 검증  ── wOBA 게이트 (raw AVG 는 참고만) ──────────────
    log("### 2-3) 도메인 검증 — wOBA 게이트  (동일 손 매치업에서 타자 불리)\n")
    w = agg["woba"]
    a = agg["avg"]
    ok1 = w.get("L_L") < w.get("L_R")
    ok2 = w.get("R_R") < w.get("R_L")
    log(f"- [{'PASS' if ok1 else 'FAIL'}] 좌투: L_L wOBA({w.get('L_L'):.4f}) < L_R wOBA({w.get('L_R'):.4f})  "
        f"Δ={w.get('L_L')-w.get('L_R'):+.4f}   (참고 AVG Δ={a.get('L_L')-a.get('L_R'):+.4f})")
    log(f"- [{'PASS' if ok2 else 'FAIL'}] 우투: R_R wOBA({w.get('R_R'):.4f}) < R_L wOBA({w.get('R_L'):.4f})  "
        f"Δ={w.get('R_R')-w.get('R_L'):+.4f}   (참고 AVG Δ={a.get('R_R')-a.get('R_L'):+.4f})")
    log(f"- 참고 HR%: L_L {agg.hr_rate.get('L_L'):.4f} vs L_R {agg.hr_rate.get('L_R'):.4f}  |  "
        f"R_R {agg.hr_rate.get('R_R'):.4f} vs R_L {agg.hr_rate.get('R_L'):.4f}")
    log(f"- 참고 ISO: L_L {agg.iso.get('L_L'):.4f} vs L_R {agg.iso.get('L_R'):.4f}  |  "
        f"R_R {agg.iso.get('R_R'):.4f} vs R_L {agg.iso.get('R_L'):.4f}\n")
    if not (ok1 and ok2):
        raise RuntimeError("wOBA 도메인 검증 실패 — 중단. pitcher_throws 파싱 재점검 필요.")
    log("→ **PASS**: wOBA 기준 양방향 모두 동일 손 매치업에서 타자 불리 확인.\n")

    # shrinkage/예측에 쓸 유효 base: UNK_* 는 전체 평균으로 대체 (pitcher_throws 결측 구간)
    baseline = {}
    for k, x in agg.iterrows():
        row = {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in x.items()}
        row["base_woba_effective"] = gw if k.startswith("UNK") else float(x.woba)
        baseline[k] = row
    baseline["_GLOBAL"] = global_row

    meta = {
        "gate": "wOBA (raw AVG 는 참고 지표로만 기록, 게이트 판정 제외)",
        "gate_change_rationale": (
            "좌완 투수의 좌타자 억제는 안타(AVG)가 아니라 장타·볼넷 억제로 나타남(세이버메트릭스 정설). "
            "raw AVG 게이트는 좌투 L_L vs L_R 를 Δ=+.0014(표준오차 ±.002 미만, z≈0.5)로 사실상 구분 못 함. "
            "wOBA 로 보면 좌타 −15.7점, 우타 −9.8점으로 양방향 정상. HR%(L_L .0147 vs L_R .0272) 도 일치."),
        "woba_weights": {"1B": W_1B, "2B": W_2B, "3B": W_3B, "HR": W_HR, "BB(+IBB+HBP)": W_BB,
                         "denominator": "AB + BB + SF", "scale": "2016 wOBA (팀 지정)"},
        "woba_ibb_merge_max_gap": max_ibb_gap,
        "woba_ibb_decision": ("병합(BB=BB+IBB+HBP) 버전 사용 — 4버킷 최대 차이 %.5f < 0.005" % max_ibb_gap
                              if max_ibb_gap < 0.005 else "차이 0.005 이상 — 검토 필요"),
        "BASE_K": BASE_K,
        "BASE_K_rationale": "상대 10타수에서 개별 전적 50% 반영(w=0.5). 검증에서 k=5/10/20/30 비교하되 기본값 유지(팀 결정).",
        "switch_hitter_handling": ("실측 타석 좌우 컬럼 없음. 통상 규칙(vs RHP→좌타석, vs LHP→우타석)으로 변환, "
                                   "batter_side_inferred 플래그. 손 미상 투수 상대 스위치 타석은 'S' 유지."),
        "pitcher_throws_missing": (
            "None/NaN 13.66% 는 무작위 아님 — 2016 시즌 100%, 2017 시즌 34% 결측, 2018+ ~0% "
            "(preview API 미제공 구간과 일치). 이 구간(UNK_*)은 좌우 매치업 base 를 전체 평균으로 설정. "
            "학습 시 해당 구간 제외 여부는 별도 검토 권고."),
        "pitcher_throws_U": "throws=='U' 는 언더핸드(정상 값). 22,701행 / 58명. 독립 그룹 유지.",
        "ab_definition": "AB={1B,2B,3B,HR,SO,OUT,GDP}, H={1B,2B,3B,HR}. BB(=BB+IBB+HBP)·SF 제외. UNK 전부 제외.",
        "known_caveat": "pa_result '1B' 에 ROE 5,622행 병합 → 안타로 집계(전체 안타의 약 4%). 사양의 9-class 정의 준수.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": SRC,
    }
    with open(OUT_BASELINE, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "baseline": baseline}, f, ensure_ascii=False, indent=2)
    log(f"산출: `{OUT_BASELINE}`\n")
    return baseline


# ── Task 3: 개별 상대 전적 (leakage-free) ──────────────────────────────
def build_prior_matchup(df: pd.DataFrame):
    log("## Task 3: 개별 상대 전적 (expanding().shift(1))\n")
    sort_keys = ["pitcher_id", "batter_id", "date", "game_id", "_orig_idx"]
    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    tracemalloc.start()
    g = df.groupby(["pitcher_id", "batter_id"], sort=False)
    df["prior_matchup_pa"] = g.cumcount().astype("int64")
    # expanding().sum().shift(1)  ==  cumsum - 현재값   (leakage-free)
    cum = {}
    for src, dst in [("is_ab", "ab"), ("is_hit", "hits"), ("is_1b", "1b"), ("is_2b", "2b"),
                     ("is_3b", "3b"), ("is_hr", "hr"), ("is_bb", "bb"), ("is_so", "so"),
                     ("is_sf", "sf")]:
        c = g[src].cumsum() - df[src].astype("int64")
        cum[dst] = c
        df[f"prior_matchup_{dst}"] = c.astype("int64")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ab = df["prior_matchup_ab"].to_numpy()
    hits = df["prior_matchup_hits"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["prior_matchup_avg"] = np.where(ab > 0, hits / ab, np.nan)
    df["prior_matchup_woba"] = woba_from_counts(
        cum["1b"].to_numpy(), cum["2b"].to_numpy(), cum["3b"].to_numpy(), cum["hr"].to_numpy(),
        cum["bb"].to_numpy(), ab, cum["sf"].to_numpy())
    df["prior_matchup_woba_den"] = (df["prior_matchup_ab"] + df["prior_matchup_bb"]
                                    + df["prior_matchup_sf"]).astype("int64")

    log(f"- 정렬 키: {sort_keys}")
    log("- 방식: `groupby.cumsum() - 현재값` (= expanding().sum().shift(1), leakage-free)")
    log(f"- 피크 메모리 (matchup 누적 연산 구간, tracemalloc): **{peak/1e6:.1f} MB**")
    n_nan_avg = int(np.isnan(df['prior_matchup_avg']).sum())
    n_nan_woba = int(np.isnan(df['prior_matchup_woba']).sum())
    log(f"- prior_matchup_avg 결측(첫 대결/ab==0): {n_nan_avg:,} ({n_nan_avg/len(df)*100:.1f}%) → NaN 유지")
    log(f"- prior_matchup_woba 결측(den==0): {n_nan_woba:,} ({n_nan_woba/len(df)*100:.1f}%) → NaN 유지\n")
    return df


def verify_no_matchup_leakage(df: pd.DataFrame, n_pairs=15, min_meet=5, seed=42):
    log("### 3-3) leakage 검증\n")
    rng = np.random.default_rng(seed)
    cnt = df.groupby(["pitcher_id", "batter_id"]).size()
    elig = cnt[cnt >= min_meet].index.to_list()
    pick = [elig[i] for i in rng.choice(len(elig), size=min(n_pairs, len(elig)), replace=False)]

    checked_pairs = checked_meet = 0
    for pid, bid in pick:
        sub = df[(df.pitcher_id == pid) & (df.batter_id == bid)].sort_values(
            ["date", "game_id", "_orig_idx"], kind="mergesort")
        c = dict(ab=0, hits=0, n1=0, n2=0, n3=0, hr=0, bb=0, sf=0)
        for _, row in sub.iterrows():
            exp_avg = (c["hits"] / c["ab"]) if c["ab"] > 0 else np.nan
            eden = c["ab"] + c["bb"] + c["sf"]
            exp_woba = ((W_1B*c["n1"] + W_2B*c["n2"] + W_3B*c["n3"] + W_HR*c["hr"] + W_BB*c["bb"]) / eden
                        if eden > 0 else np.nan)
            assert row["prior_matchup_ab"] == c["ab"], f"AB ({pid},{bid})"
            assert row["prior_matchup_hits"] == c["hits"], f"H ({pid},{bid})"
            assert row["prior_matchup_1b"] == c["n1"] and row["prior_matchup_hr"] == c["hr"], f"cnt ({pid},{bid})"
            g_avg, g_wo = row["prior_matchup_avg"], row["prior_matchup_woba"]
            assert (np.isnan(g_avg) if np.isnan(exp_avg) else abs(g_avg - exp_avg) < 1e-9), f"avg ({pid},{bid})"
            assert (np.isnan(g_wo) if np.isnan(exp_woba) else abs(g_wo - exp_woba) < 1e-9), f"woba ({pid},{bid})"
            c["ab"] += int(row.is_ab); c["hits"] += int(row.is_hit); c["n1"] += int(row.is_1b)
            c["n2"] += int(row.is_2b); c["n3"] += int(row.is_3b); c["hr"] += int(row.is_hr)
            c["bb"] += int(row.is_bb); c["sf"] += int(row.is_sf)
            checked_meet += 1
        checked_pairs += 1

    log(f"- 검사 조합 수: **{checked_pairs}** (대결 {min_meet}회 이상)")
    log(f"- 검사 대결(PA) 수: **{checked_meet}**")
    log("- 결과: **PASS** — 모든 prior_matchup_*(ab/hits/1b/hr/avg/wOBA)가 '그 이전 대결만'으로 계산한 값과 정확히 일치\n")


# ── Task 4: Shrinkage (base = wOBA) ──────────────────────────────────
def apply_shrinkage(df: pd.DataFrame, baseline: dict, k: int = BASE_K) -> pd.DataFrame:
    df = df.copy()
    gk = baseline["_GLOBAL"]
    base_woba = df["handedness_key"].map(
        lambda key: (baseline.get(key) or gk).get("base_woba_effective", gk["woba"])).astype(float)
    base_woba = base_woba.fillna(gk["woba"])
    base_avg = df["handedness_key"].map(
        lambda key: (baseline.get(key) or gk).get("avg") or gk["avg"]).astype(float).fillna(gk["avg"])

    ab = df["prior_matchup_ab"].to_numpy(dtype=float)
    w = ab / (ab + k)                                       # ab==0 → w=0 → base 그대로
    p_wo = df["prior_matchup_woba"].to_numpy(dtype=float)
    p_av = df["prior_matchup_avg"].to_numpy(dtype=float)
    p_wo_s = np.where(np.isnan(p_wo), 0.0, p_wo)            # NaN 이면 w=0 이라 기여 0
    p_av_s = np.where(np.isnan(p_av), 0.0, p_av)

    df["handedness_base_woba"] = base_woba.to_numpy()
    df["handedness_base_avg"] = base_avg.to_numpy()
    df["matchup_weight"] = w
    df["matchup_woba_shrunk"] = w * p_wo_s + (1 - w) * base_woba.to_numpy()   # 주 피처
    df["matchup_avg_shrunk"] = w * p_av_s + (1 - w) * base_avg.to_numpy()     # 참고
    return df


# ── Task 5: 표본 분포 ─────────────────────────────────────────────────
def report_sample_dist(df: pd.DataFrame):
    log("## Task 5: 표본 분포\n")
    ab = df["prior_matchup_ab"]
    qs = [0, .25, .5, .75, .90, .95, 1.0]
    qv = ab.quantile(qs)
    log("### prior_matchup_ab 분위수")
    log("| min | 25% | 50% | 75% | 90% | 95% | max |")
    log("|----:|----:|----:|----:|----:|----:|----:|")
    log("| " + " | ".join(f"{qv.loc[q]:.0f}" for q in qs) + " |\n")
    log("### 구간별 비율")
    log(f"- ab == 0 (첫 대결): {(ab==0).mean()*100:.1f}%  ({int((ab==0).sum()):,} 행)")
    for t in (5, 10, 20, 50):
        log(f"- ab >= {t:<2d}        : {(ab>=t).mean()*100:.1f}%  ({int((ab>=t).sum()):,} 행)")
    w = df["matchup_weight"]
    wv = w.quantile(qs)
    log("\n### matchup_weight 분포")
    log("| min | 25% | 50% | 75% | 90% | 95% | max |")
    log("|----:|----:|----:|----:|----:|----:|----:|")
    log("| " + " | ".join(f"{wv.loc[q]:.3f}" for q in qs) + " |")
    log(f"- w >= 0.5 비율: {(w>=0.5).mean()*100:.1f}%")
    med = ab.median()
    log(f"- 중앙값 ab = {med:.0f} → k=10 기준 w = {med/(med+10):.3f}\n")


# ── Task 6: 예측력 검증 ───────────────────────────────────────────────
def _feat(d, cols, baseline, k):
    gk = baseline["_GLOBAL"]
    ab = d["prior_matchup_ab"].to_numpy(dtype=float)
    base_wo = d["handedness_key"].map(
        lambda key: (baseline.get(key) or gk).get("base_woba_effective", gk["woba"])).astype(float)
    base_wo = base_wo.fillna(gk["woba"]).to_numpy()
    w = ab / (ab + k)
    p = d["prior_matchup_woba"].to_numpy(dtype=float)
    shrunk = w * np.where(np.isnan(p), 0.0, p) + (1 - w) * base_wo
    out = []
    for c in cols:
        out.append(base_wo if c == "base" else shrunk if c == "shrunk" else np.log1p(ab))
    return np.column_stack(out)


def predictive_check(df: pd.DataFrame, baseline: dict):
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                                 mean_squared_error, mean_absolute_error, r2_score)
    from scipy.stats import spearmanr

    log("## Task 6: 예측력 검증\n")
    d = df[df["is_pa"]].copy().sort_values("_orig_idx", kind="mergesort")
    d["y_woba"] = d["pa_woba_value"].astype(float)
    d["y_xbh"] = d["is_xbh"].astype(int)
    d["y_hit"] = d["is_hit"].astype(int)
    cut = int(len(d) * 0.70)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    log(f"- 표본(유효 PA): {len(d):,}  | 시간순 70/30  train={len(tr):,}  test={len(te):,}")
    log(f"- test  wOBA 평균={te.y_woba.mean():.4f}  장타율(XBH)={te.y_xbh.mean():.4f}  안타율={te.y_hit.mean():.4f}\n")

    def ev_reg(cols, k):
        m = LinearRegression().fit(_feat(tr, cols, baseline, k), tr.y_woba.values)
        p = m.predict(_feat(te, cols, baseline, k))
        return (mean_squared_error(te.y_woba.values, p) ** 0.5,
                mean_absolute_error(te.y_woba.values, p),
                r2_score(te.y_woba.values, p),
                spearmanr(te.y_woba.values, p).statistic)

    def ev_clf(cols, k, ycol):
        m = LogisticRegression(max_iter=1000).fit(_feat(tr, cols, baseline, k), tr[ycol].values)
        p = m.predict_proba(_feat(te, cols, baseline, k))[:, 1]
        return (roc_auc_score(te[ycol].values, p),
                log_loss(te[ycol].values, p, labels=[0, 1]),
                brier_score_loss(te[ycol].values, p))

    log("### 6-1) base vs shrunk vs full  (k=10)\n")
    log("**타겟 A — per-PA wOBA 값 (연속형, 선형회귀)**\n")
    log("| 모델 | 피처 | RMSE | MAE | R² | Spearman |")
    log("|------|------|-----:|----:|---:|---------:|")
    rows = [("X_base", ["base"]), ("X_shrunk", ["shrunk"]), ("X_full", ["shrunk", "ab"])]
    resA = {}
    for name, cols in rows:
        r = ev_reg(cols, 10); resA[name] = r
        log(f"| {name} | {'+'.join(cols)} | {r[0]:.5f} | {r[1]:.5f} | {r[2]:+.5f} | {r[3]:+.4f} |")
    log("\n**타겟 B — 장타 여부 2B/3B/HR (binary, 로지스틱)**\n")
    log("| 모델 | 피처 | AUC | log-loss | Brier |")
    log("|------|------|----:|---------:|------:|")
    resB = {}
    for name, cols in rows:
        r = ev_clf(cols, 10, "y_xbh"); resB[name] = r
        log(f"| {name} | {'+'.join(cols)} | {r[0]:.4f} | {r[1]:.5f} | {r[2]:.5f} |")
    log("\n**(참고) 타겟 C — 안타 여부 (binary)**\n")
    log("| 모델 | AUC | log-loss | Brier |")
    log("|------|----:|---------:|------:|")
    for name, cols in rows:
        r = ev_clf(cols, 10, "y_hit")
        log(f"| {name} | {r[0]:.4f} | {r[1]:.5f} | {r[2]:.5f} |")
    log("")

    log("### 6-2) k 민감도 (X_full)\n")
    log("| k | wOBA RMSE | wOBA R² | XBH AUC | XBH logloss |")
    log("|--:|----------:|--------:|--------:|------------:|")
    for k in (5, 10, 20, 30):
        ra = ev_reg(["shrunk", "ab"], k)
        rb = ev_clf(["shrunk", "ab"], k, "y_xbh")
        log(f"| {k} | {ra[0]:.5f} | {ra[2]:+.5f} | {rb[0]:.4f} | {rb[1]:.5f} |")
    log("\n**k 기본값 10 유지 (도메인 판단, 변경은 팀 결정).**\n")

    log("### 6-3) 표본 구간별 (test, X_base vs X_shrunk)\n")
    log("| ab 구간 | n(test) | base RMSE | shrunk RMSE | base XBH-AUC | shrunk XBH-AUC |")
    log("|---------|--------:|----------:|------------:|-------------:|---------------:|")
    mb_r = LinearRegression().fit(_feat(tr, ["base"], baseline, 10), tr.y_woba.values)
    ms_r = LinearRegression().fit(_feat(tr, ["shrunk"], baseline, 10), tr.y_woba.values)
    mb_c = LogisticRegression(max_iter=1000).fit(_feat(tr, ["base"], baseline, 10), tr.y_xbh.values)
    ms_c = LogisticRegression(max_iter=1000).fit(_feat(tr, ["shrunk"], baseline, 10), tr.y_xbh.values)
    for lo, hi, lab in [(0, 0, "0"), (1, 9, "1-9"), (10, 19, "10-19"), (20, 49, "20-49"), (50, 10**9, ">=50")]:
        seg = te[(te.prior_matchup_ab >= lo) & (te.prior_matchup_ab <= hi)]
        if len(seg) < 50:
            log(f"| {lab} | {len(seg):,} | - | - | - | - |"); continue
        pbr = mb_r.predict(_feat(seg, ["base"], baseline, 10))
        psr = ms_r.predict(_feat(seg, ["shrunk"], baseline, 10))
        rb = mean_squared_error(seg.y_woba, pbr) ** 0.5
        rs = mean_squared_error(seg.y_woba, psr) ** 0.5
        if seg.y_xbh.nunique() < 2:
            ab_, as_ = float("nan"), float("nan")
        else:
            ab_ = roc_auc_score(seg.y_xbh, mb_c.predict_proba(_feat(seg, ["base"], baseline, 10))[:, 1])
            as_ = roc_auc_score(seg.y_xbh, ms_c.predict_proba(_feat(seg, ["shrunk"], baseline, 10))[:, 1])
        log(f"| {lab} | {len(seg):,} | {rb:.5f} | {rs:.5f} | {ab_:.4f} | {as_:.4f} |")
    log("")
    return resA, resB


# ── Task 7 / 8: 산출물 저장 ───────────────────────────────────────────
def save_outputs(df: pd.DataFrame):
    log("## Task 7: 산출물 저장\n")
    keep = ["pitcher_id", "batter_id", "game_id", "_orig_idx",
            "handedness_key", "batter_side_inferred",
            "prior_matchup_pa", "prior_matchup_ab", "prior_matchup_hits",
            "prior_matchup_1b", "prior_matchup_2b", "prior_matchup_3b", "prior_matchup_hr",
            "prior_matchup_bb", "prior_matchup_so", "prior_matchup_sf",
            "prior_matchup_avg", "prior_matchup_woba", "prior_matchup_woba_den",
            "handedness_base_woba", "handedness_base_avg",
            "matchup_weight", "matchup_woba_shrunk", "matchup_avg_shrunk"]
    mh = df[keep].sort_values("_orig_idx", kind="mergesort").reset_index(drop=True)
    mh.to_parquet(OUT_MATCHUP, index=False)
    assert len(mh) == 558064, f"행 수 {len(mh)}"
    log(f"- `{OUT_MATCHUP}`  →  {mh.shape[0]:,} 행 × {mh.shape[1]} 열")

    base = pd.read_parquet(SRC).rename(columns={"pitcher_vs_batter_avg": "naver_vshra_raw"})
    add = mh.set_index(["pitcher_id", "batter_id", "game_id", "_orig_idx"])
    addcols = [c for c in keep if c not in ("pitcher_id", "batter_id", "game_id", "_orig_idx")]
    base = base.merge(add[addcols], how="left",
                      left_on=["pitcher_id", "batter_id", "game_id", "_orig_idx"], right_index=True)
    assert len(base) == 558064
    base.to_parquet(OUT_V2, index=False)
    log(f"- `{OUT_V2}`  →  {base.shape[0]:,} 행 × {base.shape[1]} 열 "
        f"(pitcher_vs_batter_avg → **naver_vshra_raw** rename, matchup_* 추가)")
    log(f"- 원본 `{SRC}` 미수정\n")


def main():
    t0 = datetime.now()
    log("# 투수-타자 매치업 이력 검증 리포트  (wOBA 정렬)\n")
    log(f"생성: {t0.isoformat()}  |  source: `{SRC}`  |  BASE_K = {BASE_K}\n")
    log("> **게이트 변경**: 좌우 도메인 검증을 raw AVG → **wOBA** 로 교체. "
        "좌완 플래툰 우위가 장타·볼넷 억제로 나타나 raw AVG 로는 좌투 스플릿이 노이즈 수준(Δ=+.0014, z≈0.5)이기 때문. "
        "wOBA 기준 양방향 정상(좌타 −15.7점, 우타 −9.8점). 상세는 Task 2-3.\n")

    df = pd.read_parquet(SRC)
    log(f"입력: {df.shape[0]:,} 행 × {df.shape[1]} 열\n")
    df = add_result_flags(df)

    log("## Task 1: 타수(AB) / 안타(H) 정의\n")
    log(f"- AB 포함: {sorted(AB_RESULTS)}   안타: {sorted(HIT_RESULTS)}")
    log("- AB 제외: BB(=BB+IBB+HBP 병합), SF   |   UNK 전부 제외")
    log(f"- 유효 PA: {int(df.is_pa.sum()):,}  AB: {int(df.is_ab.sum()):,}  H: {int(df.is_hit.sum()):,}  "
        f"전체 AVG: {df.is_hit.sum()/df.is_ab.sum():.4f}")
    n_roe = int((df.pa_result_raw == 'ROE').sum())
    log(f"- caveat: pa_result '1B' 에 ROE {n_roe:,}행 병합 → 안타 집계(전체 안타의 {n_roe/df.is_hit.sum()*100:.1f}%). 9-class 정의 준수.\n")

    baseline = build_baseline(df)
    df = build_prior_matchup(df)
    verify_no_matchup_leakage(df)
    df = apply_shrinkage(df, baseline, BASE_K)
    report_sample_dist(df)
    resA, resB = predictive_check(df, baseline)
    save_outputs(df)

    log("## 결론\n")
    bA, sA, fA = resA["X_base"], resA["X_shrunk"], resA["X_full"]
    bB, sB, fB = resB["X_base"], resB["X_shrunk"], resB["X_full"]
    log(f"- 타겟 A(wOBA):  base R²={bA[2]:+.5f} → shrunk {sA[2]:+.5f} → full {fA[2]:+.5f}  "
        f"(RMSE {bA[0]:.5f}→{sA[0]:.5f}→{fA[0]:.5f})")
    log(f"- 타겟 B(XBH):   base AUC={bB[0]:.4f} → shrunk {sB[0]:.4f} → full {fB[0]:.4f}  "
        f"(logloss {bB[1]:.5f}→{sB[1]:.5f}→{fB[1]:.5f})")
    log("")
    with open(OUT_REPORT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(REPORT) + "\n")
    print(f"\n리포트 저장: {OUT_REPORT}\n소요: {datetime.now()-t0}")


if __name__ == "__main__":
    main()
