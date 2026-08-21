# K-Moneyball ⚾

> **KBO WPA 파이프라인 + CatBoost / Monte Carlo / DQN 학습용 데이터셋**

네이버 스포츠 문자중계 API를 비동기로 크롤링하여 타석 단위(PA-level) Parquet을 생성하고,
문형우 외(2016) 마르코프연쇄 WE 테이블 기반의 `reward_wpa_computed` (ΔWE)를 계산합니다.
Phase 1–4(WPA 산출) + Phase 5(불펜 자원 상태·투수 이력·정규시즌 필터) 완료.
최종 학습용 산출물 `hsk_pa_enriched.parquet` (11,334 PA rows, 정규시즌만·66컬럼)은
CatBoost / Monte Carlo / DQN 학습에 바로 사용 가능합니다.

---

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [Getting Started](#getting-started)
- [전체 파이프라인](#전체-파이프라인)
- [학습용 데이터 사용법](#학습용-데이터-사용법)
- [데이터 명세서](#데이터-명세서)
- [알려진 한계](#알려진-한계)
- [주의사항 및 참고](#주의사항-및-참고)

---

## 프로젝트 구조

```
kbo-catboost/
├── main.py                          # 단건 크롤링 진입점 (GAME_IDS 설정 후 실행)
├── kbo_crawler/
│   ├── fetcher.py                   # NaverSportsAPIFetcher (비동기 API 크롤러, pitch-level)
│   ├── preview_fetcher.py           # PreviewAPIFetcher (비동기, 선발/불펜 명단 · 2017-05-30+)
│   ├── parser.py                    # PitchDataParser (JSON → Row 변환 + 파생변수)
│   ├── pipeline.py                  # KBODataPipeline (오케스트레이터)
│   └── pa_aggregator.py             # PA 집계 모듈 (pitch → PA-level)
├── config/
│   └── season_opening_dates.json    # 연도별 정규시즌 개막일 (하드코딩 금지, 여기만 수정)
├── data_analysis/
│   ├── methods/
│   │   ├── run_hsk.py               # HSK 153경기 병렬 크롤링 + PA 집계 진입점
│   │   ├── we_re_lookup.py          # RE/WE 룩업 (Monte Carlo / DQN에서 import)
│   │   ├── state_transition.py      # before/after 상태 추출 + runs_scored
│   │   ├── inject_wpa.py            # reward_wpa_computed 계산 및 주입
│   │   ├── validate_wpa.py          # Phase 4 검증 (Naver WPA 비교)
│   │   ├── preview_parser.py        # 선발/불펜 JSON → game_lineup / game_bullpen
│   │   ├── season_filter.py         # 정규시즌/시범경기 판별 + 개막일↔불펜인원 교차검증
│   │   ├── appearance_aggregator.py # PA → 등판(pitcher×game×half) 단위 집계
│   │   ├── pitcher_history.py       # 시점 안전(leakage-free) 투수 이력 + 리그 베이스라인
│   │   ├── bullpen_state.py         # PA 시점별 불펜 소모 상황(수비팀 기준)
│   │   └── build_enriched_dataset.py# 최종 병합 + 7항목 검증 + 리포트
│   └── results/
│       ├── hsk_pa_enriched.parquet     # 최종 학습용 메인 데이터 (11,334 PA, 66컬럼) ★
│       ├── hsk_pa_with_wpa.parquet     # Phase 1-4 산출물 (11,984 PA, 원본·불변)
│       ├── hsk_pa.parquet              # WPA 추가 전 PA 집계
│       ├── hsk_pa_with_states.parquet
│       ├── game_lineup.parquet         # 경기×팀 선발투수 + 시즌 누적 스탯 (128/153경기)
│       ├── game_bullpen.parquet        # 경기×팀×투수 불펜 등록 명단
│       ├── pitcher_appearances.parquet # 등판 단위 집계 (정규시즌만)
│       ├── pitcher_history.parquet     # leakage-free 투수 이력 (prior_* 컬럼)
│       ├── pa_bullpen_state.parquet    # PA별 불펜 소모 상황
│       ├── league_baseline.json        # 리그 평균/표준편차 (z-score 정규화용, v2)
│       ├── league_baseline_v1.json     # v1 백업 (등판 단위 sd — 왜곡됨, 참고용)
│       ├── enrichment_report.md        # Phase 5 검증·필터·베이스라인 비교 리포트
│       ├── wpa_validation_report.md    # Phase 4 검증 결과
│       └── wpa_validation_scatter.png
├── data/
│   ├── pbp/                         # 개별 게임 pitch-level CSV (로컬, git 미추적)
│   └── preview/                     # 개별 경기 선발/불펜 명단 JSON (로컬, git 미추적)
├── hsk_game_ids_2016_2024.txt       # HSK 153경기 ID 목록
├── pyproject.toml
└── uv.lock
```

---

## Getting Started

### 1. `uv` 설치 (최초 1회)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 재시작하거나 `source ~/.bashrc`를 실행하세요.

### 2. 레포지토리 클론 및 의존성 설치

```bash
git clone https://github.com/your-org/kbo-catboost.git
cd kbo-catboost
uv sync
```

---

## 전체 파이프라인

```
[Phase 1] 크롤링 + PA 집계
  uv run python data_analysis/methods/run_hsk.py
  → data_analysis/results/hsk_pa.parquet

[Phase 2] 상태 변수 추가 (we_before 등)
  uv run python -m data_analysis.methods.state_transition
  → data_analysis/results/hsk_pa_with_states.parquet

[Phase 3] WPA 계산 주입
  uv run python -m data_analysis.methods.inject_wpa
  → data_analysis/results/hsk_pa_with_wpa.parquet  ★

[Phase 4] 검증
  uv run python -m data_analysis.methods.validate_wpa
  → data_analysis/results/wpa_validation_report.md

[Phase 5] 불펜 자원 상태 + 투수 이력 + 정규시즌 필터 (실행 순서 고정)
  uv run python -m kbo_crawler.preview_fetcher              # data/preview/{game_id}.json
  uv run python -m data_analysis.methods.preview_parser      # → game_lineup / game_bullpen
  uv run python -m data_analysis.methods.bullpen_state       # → pa_bullpen_state.parquet
  uv run python -m data_analysis.methods.season_filter       # 정규시즌 판별 리포트 (선택, 확인용)
  uv run python -m data_analysis.methods.appearance_aggregator  # → pitcher_appearances.parquet ★ 정규시즌만
  uv run python -m data_analysis.methods.pitcher_history      # → pitcher_history.parquet + league_baseline.json
  uv run python -m data_analysis.methods.build_enriched_dataset  # → hsk_pa_enriched.parquet  ★
```

> **학습용 데이터는 이미 생성되어 있습니다.** 크롤링부터 재실행할 필요 없이
> `hsk_pa_enriched.parquet`를 바로 사용하면 됩니다.
>
> **Phase 5 재실행 순서 주의**: `appearance_aggregator` → `pitcher_history` →
> `build_enriched_dataset` 순서를 반드시 지켜야 합니다. PA 데이터(정규시즌
> 필터 등)가 바뀌면 `pitcher_history`의 `prior_*` 컬럼(leakage-free 누적
> 통계)도 반드시 재계산해야 하며, 생략하면 실제 등판 시퀀스와 어긋난
> 값이 남습니다. `hsk_pa_with_wpa.parquet`는 Phase 5 어떤 단계에서도
> 덮어쓰지 않습니다(항상 읽기 전용).

### 백그라운드 실행

```bash
nohup uv run python data_analysis/methods/run_hsk.py > hsk.log 2>&1 &
```

---

## 학습용 데이터 사용법

```python
import pandas as pd

df = pd.read_parquet("data_analysis/results/hsk_pa_enriched.parquet")
df_clean = df[df["data_quality_flag"] == ""]  # 원본 데이터 품질 플래그, 계속 유효

# 기본 특성 / 타겟 분리 (Phase 1-4)
X = df_clean[[
    "inning", "half", "score_diff_attacker", "base_state",
    "out_count", "total_pitch_count", "inning_pitch_count",
    "batter_hit_type", "pitcher_vs_batter_avg", "batter_recent_avg",
    "pitcher_id", "batter_id",
]]
y_pa_result = df_clean["pa_result"]           # 다클래스 분류
y_reward    = df_clean["reward_wpa_computed"]  # 회귀 (ΔWE)

# CatBoost 권장 설정
cat_features = ["half", "base_state", "batter_hit_type", "pitcher_id", "batter_id"]
# pitcher_vs_batter_avg, batter_recent_avg 결측 → 0.250 대체 권장

# Phase 5 추가 특성 — 투수 변동성/이력 + 불펜 자원 상태
# prior_* 는 leakage-free(현재 등판 이전 시점까지만 누적). 표본 부족(prior_n_apps<10)
# 투수는 배제하지 말고 league_baseline.json의 평균/표준편차로 shrinkage 처리 권장:
#   shrunk = w * prior_rate + (1-w) * league_rate,  w = prior_n_pa / (prior_n_pa + 15)
X_phase5 = df_clean[[
    "prior_n_apps", "prior_wpa_mean", "prior_wpa_std",
    "prior_bb_rate", "prior_so_rate", "prior_hr_rate", "prior_innings",
    "n_pitchers_used", "current_pitcher_pa_in_app", "is_pitcher_change",
    "bullpen_available_ratio", "bullpen_source",  # source: "preview"/"estimated" — 이원화 인지 필수
    "pitcher_throws",  # L/R/U, 결측 16% (preview 미커버리지 25경기)
]]
```

> `hsk_pa_enriched.parquet`는 정규시즌 PA만 포함합니다(11,984 → 11,334, 시범경기
> 9경기/650 PA 제외). 컬럼 전체 명세·검증 결과·베이스라인 변경 이력은
> `data_analysis/results/enrichment_report.md` 참고.

### Monte Carlo / DQN에서 WE 조회

```python
from data_analysis.methods.we_re_lookup import get_we_with_boundary

we = get_we_with_boundary(
    inning=7, half="bot", score_diff=2, out_count=1, base_state="13"
)
# we: [0, 1] 범위의 공격팀 승리확률

# ΔWE 직접 계산 예시
we_before = get_we_with_boundary(inning=7, half="bot", score_diff=0, out_count=0, base_state="0")
we_after  = get_we_with_boundary(inning=7, half="bot", score_diff=3, out_count=0, base_state="1")
delta_we  = we_after - we_before  # 공격팀 관점 ΔWE
```

### 다음 단계 아키텍처

```
CatBoost
  → pa_result 다클래스 분류 (또는 reward_wpa_computed 직접 회귀)
  → P(pa_result | state) 확률 벡터 출력

Monte Carlo
  → CatBoost 확률로 PA 시퀀스 시뮬레이션
  → 경기 종료까지 ΔWE 누적 → 분포 산출

DQN
  → E[ΔWE]와 Var[ΔWE]로 마코위츠 효용 최적화
  → U = E[ΔWE] - λ · Var[ΔWE]
  → 타석 전략 선택
```

---

## 데이터 명세서

### PA-level 학습 데이터 (`hsk_pa_with_wpa.parquet`, 11,984 rows, Phase 1-4)

#### 학습 입력 특성 (X)

| 컬럼 | 타입 | 범위/값 | 결측 | 설명 |
|------|------|---------|------|------|
| `inning` | int | 1–15 | 없음 | 이닝 번호 |
| `half` | str | "top"/"bot" | 없음 | top=초(원정팀 공격), bot=말(홈팀 공격) |
| `score_diff_attacker` | int | 음수–양수 | 없음 | 공격팀 관점 점수차 (양수=리드). 부호 변환 완료 |
| `base_state` | str | "0","1","2","3","12","13","23","123" | 없음 | 점유 루 문자열 |
| `out_count` | int | 0–2 | 없음 | 아웃 수 |
| `total_pitch_count` | int | 0+ | 없음 | 투수 누적 투구 수 (피로도 지표) |
| `inning_pitch_count` | int | 0+ | 없음 | 해당 이닝 투수 투구 수 |
| `pitcher_id` | int | — | 없음 | 투수 식별자 (카테고리형) |
| `batter_id` | int | — | 없음 | 타자 식별자 (카테고리형) |
| `batter_hit_type` | str | "L"/"R"/"S" | 없음 | 타자 타석 방향 (카테고리형) |
| `pitcher_vs_batter_avg` | float | 0.0–1.0 | 일부 | 투수 vs 타자 상대 타율. 결측 → 0.250 대체 권장 |
| `batter_recent_avg` | float | 0.0–1.0 | 일부 | 타자 최근 타율. 결측 → 0.250 대체 권장 |

#### 학습 타겟 (y)

| 컬럼 | 타입 | 범위/값 | 결측 | 설명 |
|------|------|---------|------|------|
| `pa_result` | str | HR/3B/2B/1B/SF/BB/IBB/HBP/SO/GDP/OUT/UNK | 없음 | PA 결과 (다클래스 분류 타겟) |
| `reward_wpa_computed` | float | [−1, +1] | 없음 | ΔWE = we_after − we_before, 공격팀 관점 (회귀 타겟) |
| `runs_scored` | int | 0–4 | 없음 | 해당 PA 실제 득점 |

#### 상태 변수 (Monte Carlo / DQN)

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `we_before` | float | 없음 | 타석 시작 전 공격팀 승리확률 [0,1] |
| `we_after` | float | 없음 | 타석 종료 후 공격팀 승리확률 [0,1] |
| `inning_ended` | bool | 없음 | 해당 PA로 이닝이 종료됐는지 |

#### 메타 / 참고용

| 컬럼 | 설명 |
|------|------|
| `reward_wpa` | 네이버 원본 WPA. **89.46% 결측, 비표준 스케일(−21~+59). 학습 사용 금지** |
| `data_quality_flag` | `""` 정상 11,937건 / `"inning1_nonzero_start"` 47건 (학습 제외 권장) |
| `game_id` | `YYYYMMDD{Away}{Home}{N}{YYYY}` 형식 |
| `home_or_away` | 0=원정/초, 1=홈/말 |
| `score_diff` | home − away (H1 컨벤션, 학습 시 score_diff_attacker 사용 권장) |

### Phase 5 추가 컬럼 (`hsk_pa_enriched.parquet`, 11,334 rows = 정규시즌만)

위 Phase 1-4 컬럼 전체(29개, 값 무변경) + 신규 37개 컬럼. 아래는 그중 핵심만 정리한
것이며, 전체 목록·결측률·검증 결과는 `data_analysis/results/enrichment_report.md` 참고.
(별도 목록에 없는 나머지 컬럼은 `date`, `app_wpa`, `n_pa`, `n_bb`/`n_so`/`n_hr`/...,
`is_starter` 등 — **등판 단위 집계값**으로, 해당 등판이 끝난 시점 기준 값이라 등판
도중 PA에 피처로 쓰면 미래 정보가 섞인다. 등판 단위 사후분석에만 쓸 것.)

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `prior_n_apps` | int | 없음 | 이 등판 이전 누적 등판 횟수 (leakage-free) |
| `prior_wpa_mean` / `prior_wpa_std` | float | 첫/둘째 등판 NaN | 이전 등판 app_wpa의 평균/표준편차 — 투수 변동성 직접 측정 |
| `prior_bb_rate` / `prior_so_rate` / `prior_hr_rate` | float [0,1] | 첫 등판 NaN | 누적 사건 수 / 누적 타석 (현재 등판 제외) |
| `prior_n_pa` / `prior_innings` / `prior_avg_pa_per_app` | int/float | 없음(첫 등판=0) | 누적 타석·이닝·등판당 평균 타석 |
| `n_pitchers_used` | int | 없음 | 그 시점까지 등판한 투수 수(현재 투수 포함), (game_id, half) 내 단조증가 |
| `current_pitcher_pa_in_app` | int | 없음 | 현재 투수가 이번 등판에서 소화한 타석 수 |
| `is_pitcher_change` | bool | 없음 | 직전 PA 대비 투수 교체 여부 |
| `bullpen_listed` / `bullpen_used` / `bullpen_available` | int | 없음 | 명단 등재/사용/잔여 불펜 투수 수 |
| `bullpen_available_ratio` | float [0,1] | 없음 | 잔여 불펜 비율 |
| `bullpen_source` | str | 없음 | `"preview"`(실측, 83.6%) / `"estimated"`(같은 시즌·팀 중앙값 대체, 16.4%) — 학습 시 이원화 인지 필수 |
| `pitcher_throws` | str | 16.0% | L/R/U. 불펜명단 우선, 없으면 선발 정보, preview 없는 25경기는 NaN(임의 채움 안 함) |

> `prior_*` 는 각 등판 시점 **이전까지의 데이터만**으로 계산됩니다(`expanding().shift(1)`).
> `pitcher_history.py`의 `verify_no_leakage()`가 매 재계산마다 이를 재검증합니다.
> 표본 부족(`prior_n_apps < 10`) 투수를 학습 데이터에서 제외하지 마세요 — 153경기
> 맞대결 데이터 특성상 상대 선발 대부분이 여기 해당하며, 제외 시 선발 등판의
> 70%가 사라집니다. `league_baseline.json`으로 shrinkage 처리할 것.

### Pitch-level CSV (`data_analysis/results/pbp/{game_id}.csv`)

행(Row) 단위: **투구 1구 = 1행**

| 컬럼 | 설명 |
|------|------|
| `game_id`, `inning`, `home_or_away`, `score_diff`, `out_count` | 상황 변수 |
| `ball_count_B`, `ball_count_S`, `is_base1/2/3` | 볼카운트 / 루상황 |
| `pitcher_id`, `batter_id`, `batter_hit_type` | 선수 식별 |
| `pitcher_vs_batter_avg`, `batter_recent_avg` | 타율 통계 |
| `pitch_speed`, `pitch_type` | 구속(km/h), 구종 |
| `total_pitch_count`, `recent_5_pitch_speed_avg`, `inning_pitch_count` | 피로도 파생 변수 |
| `pitch_result`, `relay_text`, `reward_wpa` | 투구 결과 / 타석 결과 텍스트 / Naver WPA |

---

## 알려진 한계

| 항목 | 내용 |
|------|------|
| Option A 단순화 | WE 테이블이 1사(1 out) 기준이어서 0/2아웃도 동일 값 사용. 부호 일치율 78.5%, Spearman ρ = 0.676 |
| 이닝 보간 | 1·2·4·5·6·8·9회 WE는 3회/7회 값으로 선형 보간. 9회말 끝내기는 별도 처리로 정확 |
| 득점 1스텝 지연 | 네이버 API 특성상 득점이 다음 PA의 score_diff에 반영. runs_scored로 역산 보완 |
| 데이터 품질 47건 | `inning1_nonzero_start`: 1회 첫 PA 시작 시 score_diff ≠ 0 (크롤링 누락 추정) |
| Naver WPA 사용 금지 | 89.46% 결측 + 비표준 스케일. reward_wpa_computed를 학습 타겟으로 사용 |
| preview API 커버리지 83.7% | 2017-05-30 이전 경기(25경기)는 불펜 명단이 없어 `bullpen_source="estimated"`로 추정 대체. `pitcher_throws`도 이 구간은 NaN(16.0%) |
| 시범경기 판별은 "3월"이 아님 | 개막일(`config/season_opening_dates.json`) 기준 9경기/650PA(5.4%)만 시범경기. 달력상 3월 경기는 14건이지만 5건(2018-03-30/31, 2024-03-26/27/28)은 그 해 조기 개막으로 정규시즌 |
| `league_baseline.json` v1→v2 | v1은 등판 단위로 sd를 계산해 표본 적은 투수의 극단값(1타석 1볼넷=rate 1.0)에 오염되어 sd가 평균의 1.65~2.7배로 왜곡. v2는 투수 단위 집계로 수정(v1은 `league_baseline_v1.json`으로 보존) |

---

## 디버깅 이력

데이터 구축 과정의 주요 발견 사항 (발표 자료 참고용):

| 발견 | 원인 | 해결 |
|------|------|------|
| 네이버 API가 반이닝을 시간 역순으로 반환 | API 설계 | `parser.py`에서 `reversed()` 정렬 보정 |
| `home_or_away` 코드가 직관과 반대 | Naver API 명세 확인 | 선발 라인업·박스스코어 외부 진실로 확정 |
| `score_diff` 1스텝 지연 | API 특성 | 설계로 인정, `runs_scored`로 역산 보완 |
| 동일 방향 다중 부호 오류가 단위 테스트를 통과 | 일관된 부호 반전은 테스트로 탐지 불가 | 도메인 외부 진실(경기 결과) 대조 필수 |
| Option A 1아웃 단순화로 부호 일치율 78.5% | WE 테이블이 1아웃 한정 | 한계로 명시, Option B는 보류 |
| preview 실제 API 경로가 문서화되지 않음 | 공개된 URL은 라인업 웹페이지뿐, JSON API는 별도 | 실측으로 `.../games/{game_id}/preview` → `result.previewData` 확인 |
| "시범경기 14건" 배경 수치가 실제로는 "3월 전체" | 개막일 대조 없이 달력 월로만 분류 | 개막일 설정 파일 + 불펜 인원(>20명) 교차검증(127경기 100% 일치)으로 9건/650PA로 정정 |
| `league_baseline` sd가 평균의 최대 2.7배 | 등판 단위로 sd 계산 시 표본 적은 투수의 극단값이 지배 | 투수 단위(투수당 1행)로 재집계, v1은 별도 보존 |

---

## 주의사항 및 참고

### 데이터 컨벤션 (재검토 금지)

이하 컨벤션은 선발 라인업 + 박스스코어 외부 진실로 확정됨. 코드 내부에서 재추론하지 말 것.

| 컨벤션 | 값 |
|--------|-----|
| `home_or_away = 0` | 원정팀 공격 / 초(top) |
| `home_or_away = 1` | 홈팀 공격 / 말(bot) |
| `score_diff` | home_score − away_score (H1) |
| `score_diff_attacker` | 공격팀 관점 변환 완료. 추가 부호 변환 불필요 |
| 득점 반영 시점 | 1 PA 지연 (N번째 PA 득점 → N+1번째 PA의 score_diff에 반영) |
| `reward_wpa_computed` | we_after − we_before, **공격팀 관점** ΔWE |
| 투수(수비팀) 관점 보상 | `-reward_wpa_computed` |

> **경고**: 데이터셋 내부에 여러 부호가 일관되게 뒤집힌 구간이 존재하여
> 내부 추론만으로는 컨벤션을 잘못 판단할 수 있음. 외부 진실 대조가 유일한 신뢰 기준.

### 수집 속도 및 Rate Limit

크롤러는 각 API 요청 사이에 **0.5~1.5초 랜덤 딜레이**를 적용합니다.
단일 경기 수집: 평균 10~20초. `run_hsk.py`는 `asyncio.Semaphore(5)`로 동시 5경기 병렬 처리.

### 경기 ID 형식

```
YYYYMMDD {Away} {Home} {0|1|2} {Season}
예) 20160317SKHH02016 → Away=SK, Home=HH, 단일경기, 2016시즌
```

| 팀코드 | 구단 | 팀코드 | 구단 |
|--------|------|--------|------|
| `LG` | LG 트윈스 | `HH` | 한화 이글스 |
| `KT` | KT 위즈 | `HT` | KIA 타이거즈 |
| `NC` | NC 다이노스 | `OB` | 두산 베어스 |
| `SS` | 삼성 라이온즈 | `SK` | SSG 랜더스 |
| `LT` | 롯데 자이언츠 | `WO` | 키움 히어로즈 |
