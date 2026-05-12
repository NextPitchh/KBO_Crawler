# K-Moneyball ⚾

> **투수 교체 의사결정 DQN 모델 학습을 위한 KBO PBP(Pitch-by-Pitch) 데이터 수집기**

네이버 스포츠 문자중계 API를 비동기로 크롤링하여, CatBoost 기반 강화학습(DQN) 모델이
바로 ingestion할 수 있는 투구 단위(Pitch-by-Pitch) CSV를 생성합니다.

---

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [Getting Started](#getting-started)
- [실행 방법](#실행-방법)
- [데이터 명세서](#데이터-명세서)
- [주의사항 및 참고](#주의사항-및-참고)

---

## 프로젝트 구조

```
kbo-catboost/
├── main.py                  # 진입점 — GAME_IDS, OUTPUT_DIR 설정 후 실행
├── kbo_crawler/
│   ├── fetcher.py           # NaverSportsAPIFetcher  (비동기 API 크롤러)
│   ├── parser.py            # PitchDataParser        (JSON → Row 변환 + 파생변수)
│   └── pipeline.py         # KBODataPipeline        (오케스트레이터 + 파티셔닝 저장)
├── data/
│   └── pbp/                 # 경기별 개별 CSV 저장 위치 (자동 생성)
│       └── {game_id}.csv
├── pyproject.toml
└── uv.lock
```

---

## Getting Started

### 1. `uv` 설치 (최초 1회)

`uv`는 Rust 기반 Python 패키지 매니저로 `pip` 대비 10~100배 빠른 의존성 설치를 제공합니다.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 재시작하거나 `source ~/.bashrc` (또는 `~/.zshrc`)를 실행하세요.

### 2. 레포지토리 클론 및 의존성 설치

```bash
git clone https://github.com/your-org/kbo-catboost.git
cd kbo-catboost

# 가상환경 생성 + 의존성 한 번에 설치
uv sync
```

> `uv sync`는 `uv.lock`을 기반으로 재현 가능한 환경을 보장합니다.
> lock 파일 없이 최신 버전으로 새로 설치하려면 `uv add aiohttp pandas tqdm`을 사용하세요.

### 3. 수집 대상 경기 ID 설정

`main.py` 상단의 `GAME_IDS` 리스트에 원하는 경기 ID를 추가합니다.

```python
# main.py
GAME_IDS: list[str] = [
    "20240323LGKT02024",   # 2024-03-23 LG vs KT
    "20240323NCOB02024",   # 2024-03-23 NC vs OB
    # 필요한 만큼 추가 ...
]
OUTPUT_DIR = "data/pbp/"   # 경기별 CSV가 저장될 디렉토리
```

경기 ID 형식: `YYYYMMDD{홈팀코드}{원정팀코드}0{시즌연도}`

---

## 실행 방법

```bash
# 가상환경 활성화 없이 바로 실행 (권장)
uv run python main.py

# 또는 가상환경을 직접 활성화 후 실행
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python main.py
```

실행 시 터미널에 진행 상황이 출력되며, 수집 완료된 경기는 `data/pbp/{game_id}.csv`로 저장됩니다.

```
13:42:01 [INFO] kbo_crawler.pipeline: game=20240323LGKT02024 → 247 투구 저장: data/pbp/20240323LGKT02024.csv
13:42:19 [INFO] kbo_crawler.pipeline: game=20240323NCOB02024: 이미 수집된 경기입니다 → 스킵
...
```

> **Resume 기능**: 크롤링 도중 중단되어도 재실행 시 이미 저장된 경기는 자동으로 스킵하고
> 중단된 지점부터 이어서 수집합니다.

---

## 데이터 명세서

출력 파일 위치: `data/pbp/{game_id}.csv`
행(Row) 단위: **투구 1구 = 1행**

### 🏟️ 상황 변수 (State Features)

| 컬럼명 | 설명 | 타입 | 예시 |
|--------|------|------|------|
| `game_id` | 경기 고유 ID | str | `20240323LGKT02024` |
| `inning` | 현재 이닝 | int | `7` |
| `home_or_away` | 공격 팀 (`0`=원정팀 공격, `1`=홈팀 공격) | str | `"1"` |
| `score_diff` | 점수차 (홈팀 득점 − 원정팀 득점) | int | `-2` (홈팀이 2점 뒤짐) |
| `out_count` | 현재 아웃 카운트 (0~2) | int | `1` |
| `ball_count_B` | 볼 카운트 (0~3) | int | `2` |
| `ball_count_S` | 스트라이크 카운트 (0~2) | int | `1` |
| `is_base1` | 1루 주자 존재 여부 (`0`/`1`) | int | `1` |
| `is_base2` | 2루 주자 존재 여부 (`0`/`1`) | int | `0` |
| `is_base3` | 3루 주자 존재 여부 (`0`/`1`) | int | `0` |

### 👤 프로필 변수 (Profile Features)

| 컬럼명 | 설명 | 타입 | 예시 |
|--------|------|------|------|
| `pitcher_id` | 투수 선수 ID | str | `"67890"` |
| `batter_id` | 타자 선수 ID | str | `"12345"` |
| `pitcher_vs_batter_avg` | 해당 투수 vs 해당 타자 통산 타율 | float | `0.25` |
| `batter_recent_avg` | 타자 당일 타율 (없으면 시즌 타율 fallback) | float | `0.333` |
| `batter_hit_type` | 타자 타석 방향 (`L`=좌타, `R`=우타, `S`=양타) | str \| None | `"L"` |

> `batter_hit_type`은 투수-타자 좌우 상성(Platoon Split) 피처로, 좌투수 vs 좌타자 등
> 모델이 구종 선택 패턴 차이를 학습하는 데 활용합니다.

### ⚾ 투구 및 피로도 변수 (Pitching Features)

| 컬럼명 | 설명 | 타입 | 예시 |
|--------|------|------|------|
| `pitch_speed` | 해당 구의 구속 (km/h) | float \| None | `148.0` |
| `pitch_type` | 구종 | str | `"직구"`, `"슬라이더"`, `"체인지업"` |
| `total_pitch_count` | 해당 투수의 경기 누적 투구 수 | int | `87` |
| `recent_5_pitch_speed_avg` | **[파생]** 해당 투수의 최근 5구 구속 평균 (km/h) | float \| None | `146.80` |
| `inning_pitch_count` | **[파생]** 해당 투수의 현재 이닝 내 투구 수 | int | `12` |

> **파생 변수 계산 원칙**
> - `recent_5_pitch_speed_avg`: 구속 데이터가 없는 투구(`pitch_speed=None`)는 캐시에 포함하지 않으며,
>   해당 Row의 값도 `None`으로 처리합니다. 5구 미만인 경우 현재까지의 평균을 사용합니다.
> - `inning_pitch_count`: 이닝이 전환될 때마다 1로 리셋되며, 투수가 교체되어도 새 이닝에서 새로 카운팅합니다.

### 🎯 모델 타겟 및 보상 (Target & Reward)

| 컬럼명 | 설명 | 타입 | 예시 |
|--------|------|------|------|
| `pitch_result` | 투구 결과 코드 | str | `"S"`(헛스윙), `"B"`(볼), `"F"`(파울), `"H"`(타격), `"W"`(번트파울)|
| `reward_wpa` | 해당 타석의 WPA(Win Probability Added) 변동량 | float \| None | `0.032` |

> `reward_wpa`는 DQN 모델의 **보상 신호(Reward Signal)**로 사용됩니다.
> WPA가 양수이면 승리 확률을 높인 타석, 음수이면 낮춘 타석을 의미합니다.

---

## 주의사항 및 참고

### ⏱️ 수집 속도 및 Rate Limit

크롤러는 네이버 서버의 IP 차단을 방지하기 위해 **각 API 요청 사이에 0.5초~1.5초의 랜덤 딜레이**를 적용합니다.
단일 경기(약 12이닝)를 수집하는 데 평균 **10~20초**가 소요됩니다.
대량 수집(시즌 전체 720경기+)은 수 시간이 걸릴 수 있으므로 백그라운드 실행을 권장합니다.

```bash
# 백그라운드 실행 및 로그 파일 저장
nohup uv run python main.py > crawl.log 2>&1 &
```

딜레이 값은 `NaverSportsAPIFetcher(min_delay=0.5, max_delay=1.5)` 생성자 인자로 조정 가능합니다.

### 🔢 결측치(NaN) 처리 원칙

| 변수 | 결측 발생 조건 | 권장 처리 |
|------|---------------|-----------|
| `pitch_speed` | API 미제공 투구 (번트, 몸에 맞는 공 등) | CatBoost의 내장 결측치 처리 활용 또는 투수 평균 구속으로 imputation |
| `recent_5_pitch_speed_avg` | `pitch_speed`가 None인 경우 동일하게 None | 위와 동일 |
| `batter_hit_type` | API에 타석 방향 정보 미등록 선수 | `"U"` (Unknown) 로 fillna 또는 원-핫 인코딩 시 별도 처리 |
| `reward_wpa` | 해당 이닝 WPA 데이터 미제공 경기 | 해당 Row를 보상 학습에서 제외 (`dropna(subset=['reward_wpa'])`) |

### 🔑 경기 ID 형식 참고

```
20240323 LG KT 0 2024
────┬─── ─┬ ─┬ ─ ──┬─
    │      │  │     └ 시즌 연도
    │      │  └─ 원정팀 코드
    │      └─ 홈팀 코드
    └─ 경기 날짜 (YYYYMMDD)
```

가운데 숫자 자리 의미:

| 값 | 의미 |
|----|------|
| `0` | 해당 날짜 단일 경기 |
| `1` | 더블헤더 첫 번째 경기 |
| `2` | 더블헤더 두 번째 경기 |

팀 코드 목록 (네이버 API 기준):

| 코드 | 구단 | 비고 |
|------|------|------|
| `LG` | LG 트윈스 | |
| `KT` | KT 위즈 | |
| `NC` | NC 다이노스 | |
| `SS` | 삼성 라이온즈 | Samsung |
| `HH` | 한화 이글스 | Hanwha |
| `HT` | KIA 타이거즈 | 과거 해태(Haitai) 유래 |
| `OB` | 두산 베어스 | 과거 OB 베어스 유래 |
| `SK` | SSG 랜더스 | 과거 SK 와이번스 유래 |
| `LT` | 롯데 자이언츠 | Lotte |
| `WO` | 키움 히어로즈 | 과거 우리(Woori) 히어로즈 유래 |
