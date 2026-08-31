# 팀 핸드오프 패키지 — MANIFEST

KBO 투수 교체 의사결정 지원 시스템(K-Moneyball) 졸업 프로젝트.
Phase 1–5 완료 산출물 중 **다운스트림(CatBoost / Monte Carlo / DQN)에서
실제로 쓰는 파일만** 추린 패키지다.

- 생성일: 2026-08-31
- 총 용량: **약 49.1 MB** (18개 파일) → 100 MB 미만, git 커밋 가능
- 원본은 `data_analysis/results/` 및 `data/game_ids/` 에 그대로 있음. 이 폴더는 **복사본**이다.

---

## 읽는 순서

1. **`docs/HANDOFF.md`** — 파이프라인 전체 구조, 상태 변수 정의, WE 조회 예시, 모델 파이프라인 개요
   ⚠ 이 문서는 Phase 1–4(153경기 HSK) 시점 작성본이다. Phase 5(전 구단 10년) 변경점은 아래 3·4번으로 보완할 것.
2. **`docs/README_datasets.md`** — `all_pa_enriched_v3.parquet` 의 94개 컬럼 그룹·WE 컬럼 의미·주의사항
3. **`docs/HANDOFF_matchup_addendum.md`** — 매치업 피처(`matchup_woba_shrunk`, `prior_matchup_ab`, ISO 계열) 사용 지침
4. **`reports/`** — 각 보정·검증 단계의 근거 수치 (참고용, 필독 아님)
5. **`data/`** — 실제 데이터. 메인은 `all_pa_enriched_v3.parquet`

> ⚠ **요청된 `HANDOFF_FULL.md` · `TEAM_SETUP.md` · `MEETING_leverage_proposal.md` 는 리포지토리에 존재하지 않는다.**
> `docs/HANDOFF.md` + `docs/HANDOFF_matchup_addendum.md` 를 현재 있는 가장 가까운 대체본으로 넣었다.
> 세팅 절차는 프로젝트 루트 `README.md` 와 `CLAUDE.md` 참고.

---

## 파일 목록

### `handoff/data/` — 데이터 (필수)

| 파일 | 크기 | shape | 설명 |
|------|-----:|:-----:|------|
| `all_pa_enriched_v3.parquet` | 33.0 MB | 558,064 × 94 | **메인 학습 테이블.** PA별 게임 상황 + 투수·타자 + 결과(`pa_result`) + WPA 보상(`reward_wpa_computed`, Option B) + 매치업 피처(wOBA·ISO shrunk) + 불펜 자원. `we_before/we_after` = Option B(아웃 차원 보정). Option A 값은 `*_optionA` 로 보존 |
| `matchup_history.parquet` | 10.4 MB | 558,064 × 24 | 투수×타자 매치업 이력(leakage-free, `expanding().shift(1)`). `prior_matchup_*` 원자료 + `matchup_woba_shrunk` 등. v3와 `(game_id, _orig_idx)` 로 정렬 정합. v3에 이미 병합돼 있어 별도 조인은 선택 |
| `all_pitcher_history.parquet` | 3.7 MB | 64,571 × 31 | 등판별 투수 `prior_*` 누적 스탯(BB/SO/HR rate, WPA mean/std, PA/inning 등). 시점 안전 |
| `all_pitcher_appearances.parquet` | 850 KB | 64,571 × 22 | 등판 단위 집계(outs_recorded, innings_pitched, total_pitches, n_pa, app_wpa, start/end_inning, is_starter …) |
| `all_game_bullpen.parquet` | 393 KB | 155,911 × 7 | 경기별 불펜 명단·가용 상태(listed / used / available) |
| `all_game_lineup.parquet` | 215 KB | 12,466 × 15 | 경기별 선발 타순(1–9번 + 포지션) |
| `game_index.csv` | 547 KB | 7,841 × 10 | 경기 메타: `game_id, date, year, away/home_code, away/home_name, is_regular_season, game_status, status_info` |
| `handedness_baseline.json` | 10 KB | 14 버킷 + `_GLOBAL` | 좌우 매치업 유형(`throws_side`)별 기준값: wOBA / AVG / OBP / ISO / HR% / SO% / BB%. `UNK_*`(투수 손 미상 13.66%)는 `_GLOBAL` 사용 |
| `all_league_baseline.json` | 629 B | 12 키 | 리그 평균 rate + 표준편차(WPA std, BB/SO/HR rate). `prior_*` 축소 추정용. 정규시즌·최소 10등판 필터 적용 |

### `handoff/docs/` — 문서

| 파일 | 크기 | 설명 |
|------|-----:|------|
| `HANDOFF.md` | 9.0 KB | **(대체본)** Phase 1–4 핸드오프. 상태 변수, WE 조회 예시, CatBoost feature 권장 설정, MC/DQN 파이프라인 개요. ⚠ Phase 5 이전 작성 |
| `README_datasets.md` | 2.7 KB | v3 컬럼 그룹(기본 68 / 매치업 23 / Option A 보존 3), WE·보상 컬럼 의미, 생성 스크립트 |
| `HANDOFF_matchup_addendum.md` | 4.7 KB | 매치업 피처 사용 지침 추가분(2026-08-27). CatBoost 수치 피처로 `matchup_woba_shrunk` + `prior_matchup_ab` 동시 투입 권장 |

### `handoff/code/` — 다운스트림 import 모듈

| 파일 | 크기 | 주요 심볼 | 설명 |
|------|-----:|-----------|------|
| `we_re_lookup.py` | 21 KB | `get_we_with_out_correction`, `get_we_with_boundary`, `get_alpha`, `ALPHA_KAPPA` | RE/WE 룩업. LI 계산·Monte Carlo·DQN 보상에서 import. `get_we_with_out_correction` = Option B(아웃 차원 보정, κ=0.44) |
| `state_transition.py` | 19 KB | `to_attacker_score_diff`, `base_str`, `half_of`, `parse_runs_from_relay`, `build_state_transitions` | 상태 파생 유틸. ⚠ 요청서의 `apply_pa_result` 는 이 파일에 없음 — 득점 추정은 `_estimate_runs_from_pa_result`(내부 함수), 상태 전이는 `build_state_transitions` |

### `handoff/reports/` — 검증 리포트 (참고용)

| 파일 | 크기 | 설명 |
|------|-----:|------|
| `out_correction_report.md` | 11 KB | WE 아웃 차원 보정(Option B) 결과·검증. 네이버 부호 일치 79.0→88.2%, Spearman ρ 0.684→0.813 |
| `matchup_validation_report.md` | 7.5 KB | 매치업 이력 검증. wOBA 게이트, shrinkage(BASE_K=10), base/shrunk/full 예측력 비교 |
| `all_enrichment_report.md` | 6.7 KB | 전 구단 10년(2016–2025) 최종 통합 리포트. 크롤~enrichment 전 단계 산출물·검증 |
| `unk_audit.md` | 5.7 KB | `pa_result` UNK 정규식 커버리지 감사(2017–2025, 연도×패턴 매트릭스) |

---

## 트랙별 필요 파일

- **A · 변희민** — CatBoost (PA 결과 분류 / WPA 보상 회귀)
- **B · 윤서현** — Monte Carlo (PA 시퀀스·경기 시뮬레이션)
- **C · 박규영** — DQN (투수 교체 정책 + Leverage Index)

| 파일 | A·CatBoost | B·MonteCarlo | C·DQN |
|------|:---:|:---:|:---:|
| `data/all_pa_enriched_v3.parquet` | ● | ● | ● |
| `data/matchup_history.parquet` | ○ | ● | ○ |
| `data/all_pitcher_history.parquet` | ○ | ● | ● |
| `data/all_pitcher_appearances.parquet` | – | ● | ● |
| `data/all_game_bullpen.parquet` | – | ● | ● |
| `data/all_game_lineup.parquet` | – | ● | ○ |
| `data/game_index.csv` | ○ | ● | ● |
| `data/handedness_baseline.json` | ○ | ● | ○ |
| `data/all_league_baseline.json` | ○ | ● | ○ |
| `code/we_re_lookup.py` | – | ● | ● |
| `code/state_transition.py` | – | ● | ● |
| `docs/*` | ● | ● | ● |
| `reports/*` | ○ | ○ | ○ |

범례: ● 필수 · ○ 참고/선택 · – 불필요

> **A(CatBoost)** 는 매치업·투수 prior·핸디드니스 base가 `v3` 에 이미 병합돼 있어
> `all_pa_enriched_v3.parquet` + `docs/` 만으로 학습 가능. 나머지는 조인 검증·피처 해석용 참고.
> **B(Monte Carlo)** 는 라인업·불펜·투수 이력·baseline 전부가 시뮬레이션 입력이라 거의 모든 파일이 필수.
> **C(DQN)** 는 상태(게임 상황+불펜 가용) + 보상(`reward_wpa_computed`, `we_re_lookup`) + 행동 공간(불펜)이 핵심.

---

## 제외된 것 (의도적)

구버전·중간 산출물은 넣지 않았다:

- 153경기 HSK 구버전: `hsk_pa*.parquet`, `game_bullpen.parquet`, `game_lineup.parquet`,
  `pitcher_history.parquet`, `pitcher_appearances.parquet`, `pa_bullpen_state.parquet`,
  `league_baseline.json`, `league_baseline_v1.json`, `wpa_validation_report.md`, `wpa_validation_scatter.png`
- STALE: `hsk_pa_enriched_STALE.parquet`, `enrichment_report_STALE.md`, `STALE_FILES_README.md`
- 중간 산출물: `all_pa_enriched.parquet`(기준선), `all_pa_enriched_v2.parquet`,
  `all_pa_enriched_corrected.parquet`, `all_pa_with_wpa.parquet`, `all_pa_bullpen_state.parquet`
- 중간 단계 디렉터리: `pilot/`, `staging/`, `pbp/`
- 기타: `20160317SKHH_pa_wpa.csv`

> `all_` 접두어라도 중간 산출물인 것(`all_pa_enriched`, `_v2`, `_corrected`, `all_pa_with_wpa`,
> `all_pa_bullpen_state`)은 제외. 필요한 `all_*` 는 위 목록 9개뿐.

---

## 무결성 검증 (2026-08-31)

| 항목 | 결과 |
|------|------|
| `data/` parquet 8종 + json 2종 + csv 1종 md5 원본 대조 | **전부 일치** |
| `all_pa_enriched_v3.parquet` shape | **558,064 × 94** ✓ |
| 제외 목록 22개 파일 + `pilot/ staging/ pbp/` 디렉터리 | handoff/ 내 **0건** ✓ |
| 총 용량 | **49.1 MB** (< 100 MB) |

---

## 전달 방식

**49.1 MB < 100 MB → git 저장소에 그대로 커밋 가능.**

- 개별 파일 최대 33 MB(`all_pa_enriched_v3.parquet`) — GitHub 50 MB 경고선 미만, LFS 불필요.
- `handoff/` 를 통째로 커밋 후 팀원은 `git pull` 로 수령.
- `.gitignore` 수정 불필요(현재 `handoff/` 는 무시 대상 아님). 별도 조정이 필요하면 담당자에게 문의.
