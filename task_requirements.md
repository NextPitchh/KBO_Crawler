# KBO Data Pipeline Task Requirements: Pitch to Plate Appearance (PA) Aggregation

## 1. 프로젝트 목표

기존 KBO Pitch-by-Pitch(투구 단위) 크롤러를 수정하여 텍스트 중계 데이터를 추가로 수집하고, 수집된 투구 단위 데이터를 타석 단위(Plate Appearance, PA)로 압축 및 라벨링하는 전처리 파이프라인을 구축합니다. 이 작업은 10년 치 HSK(한화 vs SK/SSG) 맞대결 데이터에 대해 Subagent를 활용한 병렬 처리로 수행되어야 합니다.

## 2. 세부 요구사항 (Step-by-Step)

### Step 1. 크롤링 파서(Parser) 보완: 텍스트 중계 수집

타석 단위 결과를 도출하기 위해 기존 `parser.py`에 텍스트 파싱 로직을 추가합니다.

* **수정 대상:** JSON 응답에서 텍스트 중계 필드 추출
* **추가 컬럼:** `relay_text`
* **데이터 예시:** `"최정 : 좌익수 앞 1루타"`, `"김광현 : 헛스윙 삼진 아웃"`

### Step 2. 전처리 (Pandas): 타석 단위(PA) 압축 및 라벨링

수집된 수십만 줄의 투구 단위 CSV를 모델(CatBoost/DQN) 학습에 적합한 타석 단위 데이터로 변환합니다.

#### 2.1. 정규표현식(Regex) 기반 타석 결과 추출

* `relay_text` 컬럼의 텍스트를 분석하여 핵심 키워드를 기반으로 `pa_result` (타석 최종 결과) 컬럼을 생성합니다.
* **Mapping Rule 예시:**
  * `1루타` -> `1B`
  * `2루타` -> `2B`
  * `3루타` -> `3B`
  * `홈런` -> `HR`
  * `삼진` -> `SO`
  * `볼넷` -> `BB`
  * `땅볼`, `뜬공`, `희생타`, `파울플라이`, `병살타` 등 아웃 처리 타구 -> `OUT`

#### 2.2. 타석 단위 GroupBy 압축

단일 타석 내의 여러 투구 데이터를 하나의 행(Row)으로 집계합니다.

* **Grouping Key:** `['game_id', 'inning', 'pitcher_id', 'batter_id']`
* **State (시작 상태):** 타석의 첫 공 기준 (`.first()`)으로 데이터를 추출합니다.
  * 대상 컬럼: `score_diff`, `out_count`, `is_base1`, `is_base2`, `is_base3`, `total_pitch_count`, `inning_pitch_count` 등
* **Target (최종 결과):** 타석의 마지막 공 기준 (`.last()`)으로 데이터를 추출합니다.
  * 대상 컬럼: 새로 생성한 `pa_result`, 기존에 존재하는 `reward_wpa`
* **Aggregate (집계 피처 생성):**
  * `pitches_per_pa`: `.count()` (해당 그룹의 투구 수 총합)
  * `pa_avg_pitch_speed`: `.mean()` (해당 타석 투구들의 평균 구속, 결측치 무시)

### Step 3. 병렬 처리 및 실행 오케스트레이션

* **대상 파일:** `hsk_game_ids_2016_2024.txt` (약 150경기 수준)
* **실행 방식:** * Subagent 아키텍처 또는 비동기/멀티프로세싱 프레임워크를 활용하여 위 1~2단계 파이프라인을 병렬로 처리합니다.
  * 속도 최적화와 함께 API Rate Limit(서버 차단)을 고려한 딜레이(지연) 처리 로직이 반드시 포함되어야 합니다.
  * 개별 경기 단위로 처리된 데이터를 최종적으로 하나의 통합된 PA 단위 DataFrame(또는 CSV/Parquet)으로 병합해야 합니다.
