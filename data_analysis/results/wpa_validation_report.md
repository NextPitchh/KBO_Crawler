# WPA Validation Report
Generated: 2026-05-20
Parquet: `data_analysis/results/hsk_pa_with_wpa.parquet`

---

## 4-1. Self Sanity Check

| Item | Value |
|---|---|
| Missing (reward_wpa_computed) | 0 |
| Out-of-range [-1,+1] violations | 0 |
| Grand mean (expected ≈ 0) | 0.013693 |
| Domain order OK (HR>1B>BB>0>OUT) | YES |

### pa_result mean ΔWE (domain order check)

| pa_result | n | mean_delta_we |
| --- | --- | --- |
| HR | 309 | 0.1099 |
| 3B | 23 | 0.0929 |
| 2B | 452 | 0.0719 |
| 1B | 1723 | 0.0448 |
| SF | 86 | 0.0290 |
| BB | 1121 | 0.0341 |
| SO | 2116 | -0.0081 |
| GDP | 216 | -0.0302 |
| OUT | 5384 | -0.0031 |
| UNK | 554 | 0.0329 |

---

## 4-2. Comparison vs Naver Original

### (1) Sample Statistics

- **Comparable PAs**: 1,263 (both reward_wpa and reward_wpa_computed non-null)
- Naver `reward_wpa` is **non-standard scale** (describe below; values up to ±50 suggest percentage-point or proprietary unit — NOT WE delta)

**Naver reward_wpa describe:**

```
count    1263.000000
mean       -0.180285
std         5.451105
min       -21.000000
25%        -2.600000
50%        -1.200000
75%         1.500000
max        59.000000
```

**Season distribution of comparable PAs:**

| season | n_pa |
| --- | --- |
| 2024 | 1263 |

### (2) Sign Match Rate

- Sign match rate (excluding exact-zero on either side): **78.5%**
- Threshold: ≥ 80%  →  **FAIL**

### (3) Rank Correlation

| Method | Statistic | p-value |
|---|---|---|
| Spearman ρ | 0.6764 | 1.077e-169 |
| Pearson r (reference) | 0.7573 | 1.510e-235 |

Spearman threshold: ρ ≥ 0.6, p < 0.05  →  **PASS**

### (4) Scatter Plot

Saved to `data_analysis/results/wpa_validation_scatter.png`

### (5) Top-10 Sign-Mismatch Cases (by |naver| × |computed|)

| game_id | inning | half | base_state | score_diff_attacker | pa_result | runs_scored | reward_wpa | reward_wpa_computed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20240502SKHH02024 | 2 | bot | 12 | -2 | OUT | 0 | -11.2 | 0.059 |
| 20240524HHSK02024 | 6 | bot | 12 | -1 | OUT | 0 | -4.8 | 0.11424999999999996 |
| 20240525HHSK02024 | 7 | bot | 2 | -1 | OUT | 0 | -7.2 | 0.04899999999999999 |
| 20240328HHSK02024 | 2 | bot | 12 | 0 | GDP | 0 | -9.9 | 0.031000000000000028 |
| 20240525HHSK02024 | 10 | top | 1 | 0 | SO | 0 | -4.3 | 0.07100000000000006 |
| 20240816HHSK02024 | 9 | bot | 0 | -1 | OUT | 0 | -5.7 | 0.051999999999999935 |
| 20240910HHSK02024 | 2 | top | 12 | 0 | GDP | 0 | -10.1 | 0.028999999999999915 |
| 20240328HHSK02024 | 5 | bot | 23 | -7 | 1B | 2 | 3.9 | -0.06999999999999999 |
| 20240525HHSK02024 | 8 | bot | 2 | -1 | OUT | 0 | -3.2 | 0.08400000000000002 |
| 20240525HHSK02024 | 9 | bot | 0 | 0 | OUT | 0 | -4.3 | 0.062000000000000055 |

### (6) Data Quality Flag Breakdown

- `inning1_nonzero_start` (n=47): mean=0.0137, std=0.0191
- `high_runs_scored_artifact`: **데이터 없음 (0건)**
- `normal` (n=11937): mean=0.0137, std=0.0509

---

## 4-3. 합격 기준 5종 판정

| # | Criterion | Value | Threshold | Result |
| --- | --- | --- | --- | --- |
| 1 | Missing rate = 0 | 0 missing | 0 | PASS |
| 2 | Range [-1,+1] violations = 0 | 0 violations | 0 | PASS |
| 3 | Sign match rate ≥ 80% | 78.5% | 80% | FAIL |
| 4 | Spearman ρ ≥ 0.6 & p < 0.05 | ρ=0.676, p=1.077e-169 | ρ≥0.6, p<0.05 | PASS |
| 5 | pa_result mean order (HR>1B>BB>0>OUT) | HR=0.1099, 1B=0.0448, BB=0.0341, OUT=-0.0031 | HR>1B>BB>0>OUT | PASS |

**통과: 4/5**

---

## Phase 5(Option B) 진행 여부

부호일치율 또는 ρ가 합격선 약간 미달 → **Phase 5 진행 권장.**
