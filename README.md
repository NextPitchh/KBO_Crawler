# K-Moneyball ⚾

> **KBO WPA 파이프라인 + CatBoost / Monte Carlo / DQN 학습용 데이터셋**

네이버 스포츠 문자중계 API를 비동기로 크롤링하여 타석 단위(PA-level) Parquet을 생성하고,
문형우 외(2016) 마르코프연쇄 WE 테이블 기반의 `reward_wpa_computed` (ΔWE)를 계산합니다.
Phase 1–4 완료. 산출물 `hsk_pa_with_wpa.parquet` (11,984 PA rows)은 CatBoost / Monte Carlo / DQN 학습에 바로 사용 가능합니다.

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
│   ├── fetcher.py                   # NaverSportsAPIFetcher (비동기 API 크롤러)
│   ├── parser.py                    # PitchDataParser (JSON → Row 변환 + 파생변수)
│   ├── pipeline.py                  # KBODataPipeline (오케스트레이터)
│   └── pa_aggregator.py             # PA 집계 모듈 (pitch → PA-level)
├── data_analysis/
│   ├── methods/
│   │   ├── run_hsk.py               # HSK 153경기 병렬 크롤링 + PA 집계 진입점
│   │   ├── we_re_lookup.py          # RE/WE 룩업 (Monte Carlo / DQN에서 import)
│   │   ├── state_transition.py      # before/after 상태 추출 + runs_scored
│   │   ├── inject_wpa.py            # reward_wpa_computed 계산 및 주입
│   │   └── validate_wpa.py          # Phase 4 검증 (Naver WPA 비교)
│   └── results/
│       ├── hsk_pa_with_wpa.parquet  # 학습용 메인 데이터 (11,984 PA rows) ★
│       ├── hsk_pa.parquet           # WPA 추가 전 PA 집계
│       ├── hsk_pa_with_states.parquet
│       ├── wpa_validation_report.md # Phase 4 검증 결과
│       └── wpa_validation_scatter.png
├── data/
│   └── pbp/                         # 개별 게임 pitch-level CSV (로컬, git 미추적)
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
```

> **학습용 데이터는 이미 생성되어 있습니다.** 크롤링부터 재실행할 필요 없이
> `hsk_pa_with_wpa.parquet`를 바로 사용하면 됩니다.

### 백그라운드 실행

```bash
nohup uv run python data_analysis/methods/run_hsk.py > hsk.log 2>&1 &
```

---

## 학습용 데이터 사용법

```python
import pandas as pd

df = pd.read_parquet("data_analysis/results/hsk_pa_with_wpa.parquet")
df_clean = df[df["data_quality_flag"] == ""]  # 47건 제외 → 11,937 rows

# 특성 / 타겟 분리
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
```

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

### PA-level 학습 데이터 (`hsk_pa_with_wpa.parquet`, 11,984 rows)

#### 학습 입력 특성 (X)

| 컬럼 | 타입 | 범위/값 | 설명 |
|------|------|---------|------|
| `inning` | int | 1–15 | 이닝 번호 |
| `half` | str | "top"/"bot" | top=초(원정팀 공격), bot=말(홈팀 공격) |
| `score_diff_attacker` | int | 음수–양수 | 공격팀 관점 점수차 (양수=리드). 부호 변환 완료 |
| `base_state` | str | "0","1","2","3","12","13","23","123" | 점유 루 문자열 |
| `out_count` | int | 0–2 | 아웃 수 |
| `total_pitch_count` | int | 0+ | 투수 누적 투구 수 (피로도 지표) |
| `inning_pitch_count` | int | 0+ | 해당 이닝 투수 투구 수 |
| `pitcher_id` | int | — | 투수 식별자 (카테고리형) |
| `batter_id` | int | — | 타자 식별자 (카테고리형) |
| `batter_hit_type` | str | "L"/"R"/"S" | 타자 타석 방향 (카테고리형) |
| `pitcher_vs_batter_avg` | float | 0.0–1.0 | 투수 vs 타자 상대 타율. 결측 → 0.250 대체 권장 |
| `batter_recent_avg` | float | 0.0–1.0 | 타자 최근 타율. 결측 → 0.250 대체 권장 |

#### 학습 타겟 (y)

| 컬럼 | 타입 | 범위/값 | 설명 |
|------|------|---------|------|
| `pa_result` | str | HR/3B/2B/1B/SF/BB/IBB/HBP/SO/GDP/OUT/UNK | PA 결과 (다클래스 분류 타겟) |
| `reward_wpa_computed` | float | [−1, +1] | ΔWE = we_after − we_before, 공격팀 관점 (회귀 타겟) |
| `runs_scored` | int | 0–4 | 해당 PA 실제 득점 |

#### 상태 변수 (Monte Carlo / DQN)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `we_before` | float | 타석 시작 전 공격팀 승리확률 [0,1] |
| `we_after` | float | 타석 종료 후 공격팀 승리확률 [0,1] |
| `inning_ended` | bool | 해당 PA로 이닝이 종료됐는지 |

#### 메타 / 참고용

| 컬럼 | 설명 |
|------|------|
| `reward_wpa` | 네이버 원본 WPA. **89.46% 결측, 비표준 스케일(−21~+59). 학습 사용 금지** |
| `data_quality_flag` | `""` 정상 11,937건 / `"inning1_nonzero_start"` 47건 (학습 제외 권장) |
| `game_id` | `YYYYMMDD{Away}{Home}{N}{YYYY}` 형식 |
| `home_or_away` | 0=원정/초, 1=홈/말 |
| `score_diff` | home − away (H1 컨벤션, 학습 시 score_diff_attacker 사용 권장) |

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
