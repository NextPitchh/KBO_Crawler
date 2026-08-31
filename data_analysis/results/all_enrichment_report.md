# 전 구단 10년(2016-2025) 최종 통합 리포트

이 문서는 파일럿 검증(Stage 0) → 이닝 완전성/concurrency 실측(Stage 1-2) →
전 구단 10년 크롤링 + UNK 정규식 보강 + 터미널 PA 보정 + 연장전 정책까지
완료된 뒤, 최종 병합·enrichment 단계의 산출물과 검증 결과를 기록한다.

## 최종 산출물

| 파일 | 내용 | shape |
|---|---|---|
| **`all_pa_enriched.parquet`** | **최종 산출물.** PA+WPA+등판이력+불펜상태+투수손 결합 | 558,064행 × 68열 |
| `all_pa_with_wpa.parquet` | 10개년 병합(enrichment 이전) | 558,064행 × 31열 |
| `all_pitcher_appearances.parquet` | 등판 단위 집계 | 64,571행 × 22열 |
| `all_pitcher_history.parquet` | 시점 안전 누적 투수 이력(prior_*) | 64,571행 × 28열(신규) |
| `all_league_baseline.json` | 리그 베이스라인(v2 방식, 투수 단위 sd) | - |
| `all_pa_bullpen_state.parquet` | PA별 불펜 소모 상태 | 558,064행 × 14열 |
| `all_game_lineup.parquet` | 선발투수+시즌스탯 (10년) | 12,466행 |
| `all_game_bullpen.parquet` | 불펜 명단 (10년) | 155,911행 |

`staging/pa_states_{2016..2025}.parquet` 10개 파일은 롤백 지점으로 그대로 보존.
`hsk_pa_with_wpa.parquet`(원본 153경기)은 이번 라운드 내내 미수정 확인됨(git 상태 clean).

## 검증 8항목

| # | 항목 | 결과 | 상세 |
|---|---|---|---|
| 1 | 행 수 == Task2 결과와 동일 | PASS | 558,064 → 558,064 |
| 2 | 기존 31컬럼 값 전수 일치 | PASS | `assert_frame_equal` 통과 |
| 3 | verify_no_leakage(30명 이상) | PASS | 투수 30명, 등판 2,226건, prior_wpa_std 정확 일치 |
| 4 | prior_n_apps==0 → prior_wpa_std NaN | PASS | 위반 0건 (829명 중 첫 등판) |
| 5 | n_pitchers_used 단조 증가 | PASS | 위반 그룹 0개 (14,394개 game×half 그룹 전수) |
| 6 | 비율 컬럼 [0,1] 또는 NaN | PASS | prior_bb/so/hr_rate, bullpen_available_ratio 전부 정상 |
| 7 | pa_result 도메인 순서(Tier 1) | PASS | 경고 0건(Tier 2 포함) |
| 8 | Telescoping 검증(홈팀 관점 통일) | PASS | 7,197경기, 평균절대오차 **0.000005**, 오차≥0.1 게임 **0건** |

## 연도별 PA 분포

| 연도 | PA |
|---|---|
| 2016 | 57,033 |
| 2017 | 56,107 |
| 2018 | 56,262 |
| 2019 | 54,826 |
| 2020 | 55,926 |
| 2021 | 55,785 |
| 2022 | 55,057 |
| 2023 | 55,286 |
| 2024 | 56,515 |
| 2025 | 55,267 |
| **합계** | **558,064** |

## bullpen_source 분포

| source | PA 수 | 비중 |
|---|---|---|
| preview | 481,495 | 86.3% |
| estimated | 76,569 | 13.7% |

기대치 검증: estimated 대상은 2016년 전체(720경기) + 2017-05-30 이전(244경기) =
964/7,197경기(13.4%, 게임 수 기준) — PA 가중 실측치(13.7%)와 근접해 정합성 확인.
`pitcher_throws` 결측률(13.66%)도 이와 정확히 일치.

## prior_n_apps 분포 — 153경기 대비 개선

| | 153경기(구버전) | 전 구단 10년(신버전) | 개선 배율 |
|---|---|---|---|
| 중앙값 | 5 | 52 | **10.4배** |
| 평균 | 8.08 | 80.34 | **9.9배** |
| 최댓값 | 64 | 604 | **9.4배** |
| n_established_pitchers(≥10등판) | 41명 | **626명** | **15.3배** |

표본 부족 문제가 근본적으로 해소됨 — CatBoost 학습 시 prior_* 피처의 신뢰도가
크게 개선될 것으로 기대.

## league_baseline: v2(153경기) vs 전 구단(10년)

| 지표 | 153경기 v2 | 전 구단 10년 |
|---|---|---|
| league_wpa_std | 0.1399 | 0.1851 |
| league_bb_rate | 0.0907 (sd 0.0451) | 0.1011 (sd 0.0432) |
| league_so_rate | 0.1755 (sd 0.0486) | 0.1654 (sd 0.0415) |
| league_hr_rate | 0.0255 (sd 0.0143) | 0.0228 (sd 0.0106) |
| n_established_pitchers | 41 | 626 |
| n_regular_season_appearances | 1,330 | 64,571 |

v2 조건(정규시즌·prior_n_apps≥10·투수 단위 sd 계산) 그대로 유지했으며, 세 비율
지표 모두 `sd < mean` 정상 범위 유지. 표본이 48배 커지면서 sd가 전반적으로
안정화(bb_rate/so_rate/hr_rate sd 전부 감소).

## 메모리 피크

`all_pitcher_history.parquet` 생성(64,571건, 829명 투수별 `expanding().shift(1)`)
피크 RSS: **206.3 MB** — 4GB 기준 대비 여유 충분, 청크 처리 불필요.

## 전체 컬럼 목록 (68열)

### 기존 PA/WPA 레벨 (31열, all_pa_with_wpa와 동일)
`game_id, inning, home_or_away, pitcher_id, batter_id, batter_hit_type,
pitcher_vs_batter_avg, batter_recent_avg, score_diff, out_count, is_base1,
is_base2, is_base3, total_pitch_count, inning_pitch_count, pa_result,
pa_result_raw, reward_wpa, pitches_per_pa, pa_avg_pitch_speed, _orig_idx,
base_state, half, score_diff_attacker, we_before, runs_scored, inning_ended,
we_after, data_quality_flag, reward_wpa_computed, is_extra_innings`

- `reward_wpa`: 네이버 원본 wpaByPlate, 결측 79.97%(2016-2023 전무, 2024-2025만 존재 — 기존 문서화된 특성)
- `pa_result_raw`: 이번 라운드 신규 — HBP/IBB/ROE 등 병합 전 세분 라벨
- `is_extra_innings`: 이번 라운드 신규 — 필터링 후 전량 False(10회 이상 PA는 이미 제외됨. 컬럼 자체는 감사 추적용으로 보존)

### 등판 이력(pitcher_history, 신규 28열)
`date, app_wpa, n_pa, n_bb, n_so, n_hr, n_out, n_1b, n_2b, n_3b, n_gdp, n_sf,
outs_recorded, innings_pitched, total_pitches, start_inning, end_inning,
appearance_order, is_starter, prior_n_apps, prior_wpa_mean, prior_wpa_std,
prior_bb_rate, prior_so_rate, prior_hr_rate, prior_avg_pa_per_app,
prior_n_pa, prior_innings`

### 불펜 상태(bullpen_state, 신규 8열, `_orig_idx` 중복)
`n_pitchers_used, current_pitcher_pa_in_app, is_pitcher_change,
bullpen_listed, bullpen_used, bullpen_available, bullpen_available_ratio,
bullpen_source`

### 투수 손(신규 1열)
`pitcher_throws` (L/R/U, 결측 13.66% — bullpen_source=estimated 구간과 일치)

## 알려진 이슈 및 주의사항

1. **`_orig_idx` 재채번**: `staging/pa_states_{year}.parquet`를 단순 concat하면
   연도별로 0부터 재시작하는 `_orig_idx`가 충돌한다(10개 파일 모두 0-based).
   `all_pa_with_wpa.parquet` 생성 시 `(game_id, _orig_idx)` 정렬 후 전역
   재채번했다 — 향후 연도를 추가할 때도 동일하게 재채번 필요.
2. **`reward_wpa`(네이버 원본) 결측 79.97%**: 2016-2023년은 API 자체에
   해당 필드가 없음(0%), 2024-2025년만 100% 존재. 학습 타깃으로 쓰지 말 것
   (CLAUDE.md 기존 방침 유지) — `reward_wpa_computed`를 사용할 것.
3. **`is_extra_innings` 컬럼은 최종 데이터에서 전량 False**: 연장전 PA
   자체가 이미 제외됐기 때문. 컬럼을 남겨둔 이유는 향후 재현/감사 시
   "이 정책이 적용됐다"는 표식으로 쓰기 위함.
