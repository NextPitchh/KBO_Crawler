# 데이터셋 안내

## 파일 목록

| 파일 | 용도 |
|------|------|
| **`all_pa_enriched_v3.parquet`** | **← 사용할 것 (558,064 × 94)** |
| `all_pa_enriched_v2.parquet` | 중간 산출물 (matchup까지) |
| `all_pa_enriched_corrected.parquet` | 중간 산출물 (WE 보정만) |
| `all_pa_enriched.parquet` | 기준선 (Option A, matchup 없음) |
| `hsk_pa_enriched_STALE.parquet` | **사용 금지** |
| `matchup_history.parquet` | 매치업 이력 (조인용) |
| `all_pitcher_history.parquet` | `prior_*` 원본 |
| `handedness_baseline.json` | 좌우 매치업 기준값 |
| `all_league_baseline.json` | 리그 평균 |

원본 3개 파일(`_v2`, `_corrected`, `all_pa_enriched`)은 롤백 지점으로 전부 보존한다.

## `all_pa_enriched_v3.parquet` 컬럼 그룹 (94열)

- **기본 68열** — 게임 상황, 투수·타자, 타석 결과, `prior_*`, 불펜 자원
  (단, `pitcher_vs_batter_avg`는 `naver_vshra_raw`로 rename됨)
- **`matchup_*` / `prior_matchup_*` 23열** — 시점 안전 투수×타자 매치업 피처
  (`matchup_woba_shrunk`, `prior_matchup_ab` 등).
  wOBA(컨택트·출루)와 ISO(파워) 두 차원을 함께 제공한다.
  장타 예측에는 ISO가, 종합 매치업에는 wOBA가 적합하다.
  (`prior_matchup_iso`, `handedness_base_iso`, `matchup_iso_shrunk`)
- **Option A 보존 컬럼 3열** — `we_before_optionA`, `we_after_optionA`,
  `reward_wpa_computed_optionA`

## WE / 보상 컬럼 (중요)

**`v3`의 `we_before` / `we_after` / `reward_wpa_computed`는 Option B
(아웃 차원 보정) 값이다.** 다운스트림 코드가 참조하는 이름을 유지한 채
내용만 Option B로 교체했다.

| 컬럼 | 의미 |
|------|------|
| `reward_wpa_computed` | **기본 학습 타겟.** 공격팀 관점 ΔWE, Option B |
| `we_before` / `we_after` | Option B WE (아웃 차원 보정 적용) |
| `reward_wpa_computed_optionA` | 구 기본값 (Option B 승격 전), 비교용 보존 |
| `we_before_optionA` / `we_after_optionA` | Option A WE, 비교용 보존 |
| `reward_wpa` | 네이버 원본. 결측 80%, **사용 금지** (검증 참고용) |

### Option B 채택 근거 (요약)

- 네이버 부호 일치율 79.0% → 88.2% (+9.2%p), Spearman ρ 0.684 → 0.813
- 주자 있는 아웃(OUT/SO/GDP): 68.4% → 84.2%
- pa_result 도메인 순서·telescoping 항등식 유지, 안타 계열 왜곡 없음(|Δ| ≤ 0.002)
- GDP 평균 ΔWE −0.032 → −0.055 (무사 만루 병살 등 아웃 페널티 현실화)

상세: `out_correction_report.md`. 컬럼 전체 명세는 `HANDOFF_FULL.md` 갱신 시 추가.

## 생성 스크립트

```bash
uv run python -m data_analysis.methods.build_enriched_v3
```
