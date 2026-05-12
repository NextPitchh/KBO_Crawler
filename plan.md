# Plan: Pitch-by-Pitch → Plate Appearance (PA) 집계 파이프라인 구축

## 개요

기존 투구 단위(pitch-level) 크롤러를 확장하여 HSK 맞대결 10년 치 데이터를
**타석 단위(PA-level)** 데이터셋으로 변환하는 3단계 파이프라인을 구현한다.

```
[현재]  크롤링 → pitch-level CSV (1구 = 1행)
[목표]  크롤링 → pitch-level CSV → PA-level Parquet (1타석 = 1행)
```

---

## Phase 1 — parser.py 수정: `relay_text` 컬럼 추가

### 목적

`type=13` textOption에서 타석 결과 텍스트(`relay_text`)를 추출한다.
이 텍스트는 Phase 2의 regex 기반 `pa_result` 생성의 유일한 소스다.

### 구현 위치

**`kbo_crawler/parser.py` → `_parse_relay()` 메서드**

### 변경 내용

```python
# _parse_relay() 내부에서 type=1 투구 루프 직전에 추가
relay_text: str = ""
for opt in text_options:
    if opt.get("type") in (13, 23):          # 타석 결과 텍스트
        relay_text = opt.get("text", "") or ""
        break
```

추출한 `relay_text`를 해당 relay의 **모든 투구 Row에 동일하게 삽입**한다.
(GroupBy 후 `.last()`로 꺼낼 수 있어야 하므로 pitch 단위 행에 중복 보관)

```python
row["relay_text"] = relay_text   # _parse_pitch_option() 반환 dict에 추가
```

### `ORDERED_COLS` 업데이트 (pipeline.py)

`relay_text`를 `reward_wpa` 바로 앞에 삽입한다.

---

## Phase 2 — PA 집계 모듈 신설: `kbo_crawler/pa_aggregator.py`

### 설계 원칙

- **단일 책임**: 집계만 담당. 크롤링·저장 로직은 건드리지 않는다.
- **순수 함수**: `aggregate_pa(df: pd.DataFrame) -> pd.DataFrame` 하나로 완결.
- **통계적 정합성**: GroupBy key의 유일성 조건을 사전 검증한다.

### 2-1. 정규표현식 기반 `pa_result` 생성

우선순위 순서(위에서 먼저 매칭되는 규칙이 이긴다)로 적용한다.

| 우선순위 | 패턴 (한국어) | `pa_result` | 이유 |
|----------|--------------|-------------|------|
| 1 | `홈런` | `HR` | 가장 구체적·희귀 |
| 2 | `3루타` | `3B` | |
| 3 | `2루타` | `2B` | |
| 4 | `1루타` | `1B` | |
| 5 | `고의사구` | `IBB` | 볼넷보다 먼저 체크 |
| 6 | `볼넷` | `BB` | |
| 7 | `몸에 맞는 공\|사구` | `HBP` | |
| 8 | `삼진` | `SO` | |
| 9 | `병살타` | `GDP` | OUT 세분화 |
| 10 | `희생플라이\|희생타` | `SF` | OUT 세분화 |
| 11 | `땅볼\|뜬공\|파울플라이\|내야플라이\|파울 아웃` | `OUT` | |
| 99 | 미매칭 | `UNK` | 로그 기록 후 검토 |

```python
import re

PA_RESULT_PATTERNS = [
    (r"홈런",                      "HR"),
    (r"3루타",                     "3B"),
    (r"2루타",                     "2B"),
    (r"1루타",                     "1B"),
    (r"고의사구",                   "IBB"),
    (r"볼넷",                      "BB"),
    (r"몸에 맞는 공|사구",           "HBP"),
    (r"삼진",                      "SO"),
    (r"병살타",                     "GDP"),
    (r"희생플라이|희생타",           "SF"),
    (r"땅볼|뜬공|파울플라이|내야플라이|파울 아웃", "OUT"),
]

def _classify_pa_result(text: str) -> str:
    if not isinstance(text, str):
        return "UNK"
    for pattern, label in PA_RESULT_PATTERNS:
        if re.search(pattern, text):
            return label
    return "UNK"
```

### 2-2. GroupBy 키 설계 및 유일성 주의사항

**GroupBy Key: `[game_id, inning, pitcher_id, batter_id]`**

> **통계적 주의**: 동일 이닝에서 같은 투수-타자 조합이 두 번 나오면
> (타순이 한 바퀴 도는 경우, 매우 드묾) 두 타석이 **하나로 합쳐지는 오류**가 발생한다.
> 이를 방지하기 위해 집계 전에 **연속 그룹 인덱스(PA 시퀀스 번호)를 부여**한다.

```python
# 연속 그룹 변화 감지로 PA 시퀀스 번호 생성
key_cols = ["game_id", "inning", "pitcher_id", "batter_id"]
df["_pa_seq"] = (
    df[key_cols].ne(df[key_cols].shift()).any(axis=1).cumsum()
)
group_key = ["game_id", "_pa_seq"]
```

### 2-3. GroupBy 집계 로직

```python
def aggregate_pa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # pa_result 생성 (relay_text는 타석 내 모든 행에 동일)
    df["pa_result"] = df["relay_text"].apply(_classify_pa_result)

    # PA 시퀀스 번호 부여 (중복 타석 방지)
    key_cols = ["game_id", "inning", "pitcher_id", "batter_id"]
    df["_pa_seq"] = (
        df[key_cols].ne(df[key_cols].shift()).any(axis=1).cumsum()
    )

    # 상태 변수: 타석 첫 투구 기준
    first_cols = [
        "game_id", "inning", "home_or_away",
        "pitcher_id", "batter_id", "batter_hit_type",
        "pitcher_vs_batter_avg", "batter_recent_avg",
        "score_diff", "out_count",
        "is_base1", "is_base2", "is_base3",
        "total_pitch_count", "inning_pitch_count",
    ]
    pa_first = df.groupby("_pa_seq")[first_cols].first()

    # 타겟/보상: 타석 마지막 투구 기준
    last_cols = ["pa_result", "reward_wpa"]
    pa_last = df.groupby("_pa_seq")[last_cols].last()

    # 집계 피처
    pa_agg = df.groupby("_pa_seq").agg(
        pitches_per_pa=("pitch_result", "count"),
        pa_avg_pitch_speed=("pitch_speed", "mean"),   # NaN 자동 무시
    )

    pa_df = pd.concat([pa_first, pa_last, pa_agg], axis=1).reset_index(drop=True)

    # UNK 비율 로깅 (QA)
    unk_rate = (pa_df["pa_result"] == "UNK").mean()
    if unk_rate > 0.05:
        logger.warning("pa_result UNK 비율 %.1f%% — regex 패턴 보강 필요", unk_rate * 100)

    return pa_df
```

### 출력 스키마 (PA-level)

| 컬럼 | 소스 | 설명 |
|------|------|------|
| `game_id` | first | 경기 ID |
| `inning` | first | 이닝 |
| `home_or_away` | first | 공격팀 |
| `pitcher_id` | first | 투수 ID |
| `batter_id` | first | 타자 ID |
| `batter_hit_type` | first | L/R/S |
| `pitcher_vs_batter_avg` | first | 투-타 통산 타율 |
| `batter_recent_avg` | first | 타자 당일/시즌 타율 |
| `score_diff` | first | 타석 시작 시 점수차 |
| `out_count` | first | 타석 시작 시 아웃카운트 |
| `is_base1/2/3` | first | 타석 시작 시 루상황 |
| `total_pitch_count` | first | 타석 시작 시 투수 누적 투구수 |
| `inning_pitch_count` | first | 타석 시작 시 이닝 내 투구수 |
| `pa_result` | last | 타석 결과 (HR/1B/2B/3B/BB/IBB/HBP/SO/GDP/SF/OUT/UNK) |
| `reward_wpa` | last | 타석 WPA 보상 |
| `pitches_per_pa` | agg | 타석 내 투구 수 |
| `pa_avg_pitch_speed` | agg | 타석 평균 구속 (km/h) |

---

## Phase 3 — 병렬 처리 오케스트레이터: `run_hsk.py`

### 아키텍처 선택

| 방식 | 장점 | 단점 | 결론 |
|------|------|------|------|
| `asyncio.Semaphore` | 단순, 단일 프로세스, Rate Limit 제어 용이 | GIL 영향(파싱은 CPU) | **채택** |
| `multiprocessing` | 진짜 병렬 CPU | 경기당 별도 세션 필요, 복잡 | 과잉설계 |
| `concurrent.futures` | 쉬운 API | thread-safe 아닌 aiohttp와 혼용 복잡 | 기각 |

**근거**: 병목은 CPU(파싱)가 아닌 I/O(API 대기)다. asyncio Semaphore로
동시 요청 수를 제한하면 Rate Limit도 안전하게 관리된다.

### 병렬 처리 설계

```
run_hsk.py
├── load_game_ids("hsk_game_ids_2016_2024.txt")
├── asyncio.run(crawl_all(game_ids, concurrency=5))
│     └─ Semaphore(5) — 최대 5경기 동시 크롤링
│          └─ per game: fetch → parse → save CSV
└── aggregate_all_csvs("data/pbp/", "data/hsk_pa.parquet")
      └─ glob CSVs → pd.concat → aggregate_pa() → Parquet
```

```python
# run_hsk.py 핵심 구조
import asyncio, glob
import pandas as pd
from kbo_crawler.fetcher import NaverSportsAPIFetcher
from kbo_crawler.parser import PitchDataParser
from kbo_crawler.pipeline import KBODataPipeline
from kbo_crawler.pa_aggregator import aggregate_pa

CONCURRENCY = 5          # 동시 요청 수 (Rate Limit 안전선)
GAME_ID_FILE = "hsk_game_ids_2016_2024.txt"
OUTPUT_DIR   = "data/pbp/"
PA_OUTPUT    = "data/hsk_pa.parquet"

async def crawl_all(game_ids: list[str]) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    fetcher = NaverSportsAPIFetcher()

    async def process_one(session, game_id: str):
        csv_path = f"{OUTPUT_DIR}{game_id}.csv"
        if os.path.isfile(csv_path):
            return                                # Resume 스킵
        async with sem:
            parser = PitchDataParser()           # 경기당 독립 인스턴스
            innings_data = await fetcher.fetch_game(session, game_id)
            rows = parser.parse_game(game_id, innings_data)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    async with fetcher.create_session() as session:
        tasks = [process_one(session, gid) for gid in game_ids]
        await asyncio.gather(*tasks)

def aggregate_all_csvs() -> pd.DataFrame:
    files = glob.glob(f"{OUTPUT_DIR}*.csv")
    dfs = [pd.read_csv(f) for f in files]
    pitch_df = pd.concat(dfs, ignore_index=True)
    pa_df = aggregate_pa(pitch_df)
    pa_df.to_parquet(PA_OUTPUT, index=False)
    return pa_df
```

> **Rate Limit 안전장치**: `NaverSportsAPIFetcher`가 이미 이닝당 0.5~1.5초 딜레이를
> 내장하므로, Semaphore(5)와 조합하면 초당 최대 ~10 요청으로 제한된다.

---

## 작업 순서 및 파일 변경 목록

```
Step 1  kbo_crawler/parser.py      — relay_text 추출 및 row 삽입
Step 1  kbo_crawler/pipeline.py    — ORDERED_COLS에 relay_text 추가
Step 2  kbo_crawler/pa_aggregator.py  — 신규 파일 (PA 집계 모듈)
Step 3  run_hsk.py                 — 신규 파일 (병렬 실행 진입점)
```

---

## 품질 검증 체크리스트

- [ ] `pa_result == "UNK"` 비율 < 5% (전체 타석 대비)
- [ ] `pitches_per_pa` 분포: 평균 3~5구, 최대 15구 이하 (이상값 확인)
- [ ] `reward_wpa` 결측률 기록 (경기별 편차 확인)
- [ ] 동일 `game_id` 내 총 PA 수 ≈ 총 투구 수 / `pitches_per_pa` 평균 (sanity check)
- [ ] `pa_result` 분포: `OUT` > `1B` > `BB` > `SO` > `HR` 순 (KBO 통계 상식 기준)
