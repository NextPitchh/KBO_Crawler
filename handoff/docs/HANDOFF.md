# KBO WPA Pipeline — Handoff Document

Phase 1–4 complete. Handing off to team for CatBoost / Monte Carlo / DQN.

---

## 1-1. 산출물 위치

| 파일 | 설명 |
|---|---|
| `data_analysis/results/hsk_pa_with_wpa.parquet` | **학습용 메인 데이터** (11,984 PA rows) |
| `data_analysis/methods/we_re_lookup.py` | Monte Carlo/DQN에서 WE 조회 시 import |
| `data_analysis/results/wpa_validation_report.md` | Phase 4 검증 결과 전문 |
| `data_analysis/results/wpa_validation_scatter.png` | Naver vs computed WPA 스캐터 |
| `data_analysis/results/pbp/` | 153경기 투구별 CSV (원본) |
| `data_analysis/results/hsk_pa.parquet` | PA 집계 (WE/WPA 추가 전) |
| `data_analysis/results/hsk_pa_with_states.parquet` | 상태 변수 추가 후 중간 산출물 |

---

## 1-2. 핵심 컬럼 정의 (`hsk_pa_with_wpa.parquet`, 11,984 rows)

### 학습 입력 후보 (X)

| 컬럼 | 타입 | 범위/값 | 결측 | 설명 |
|---|---|---|---|---|
| `inning` | int | 1–15 | 없음 | 이닝 번호 |
| `half` | str | "top" / "bot" | 없음 | top=초(원정팀 공격), bot=말(홈팀 공격) |
| `score_diff_attacker` | int | 음수–양수 | 없음 | 공격팀 관점 점수차 (양수=리드, 음수=뒤짐). 이미 컨벤션 변환됨 |
| `base_state` | str | "0","1","2","3","12","13","23","123" | 없음 | 점유 루 문자열 (비어 있으면 "0") |
| `out_count` | int | 0–2 | 없음 | 아웃 수 |
| `total_pitch_count` | int | 0+ | 없음 | 투수 누적 투구 수 (피로도 지표) |
| `inning_pitch_count` | int | 0+ | 없음 | 해당 이닝 투수 투구 수 (피로도 지표) |
| `pitcher_id` | int | — | 없음 | 투수 식별자 (카테고리형) |
| `batter_id` | int | — | 없음 | 타자 식별자 (카테고리형) |
| `batter_hit_type` | str | "L" / "R" / "S" | 없음 | 타자 타석 방향 (카테고리형) |
| `pitcher_vs_batter_avg` | float | 0.0–1.0 | 일부 | 투수 vs 타자 상대 타율. 결측 → 0.250으로 대체 권장 |
| `batter_recent_avg` | float | 0.0–1.0 | 일부 | 타자 최근 타율. 결측 → 0.250으로 대체 권장 |

### 학습 타겟 후보 (y)

| 컬럼 | 타입 | 범위/값 | 결측 | 설명 |
|---|---|---|---|---|
| `pa_result` | str | HR/3B/2B/1B/SF/BB/IBB/HBP/SO/GDP/OUT/UNK | 없음 | PA 결과 (다클래스 분류 타겟) |
| `reward_wpa_computed` | float | [-1, +1] | 없음 | ΔWE = we_after − we_before, 공격팀 관점 (회귀 타겟) |
| `runs_scored` | int | 0–4 | 없음 | 해당 PA 실제 득점 (회귀 타겟) |

### 상태 변수 (Monte Carlo / DQN)

| 컬럼 | 타입 | 범위 | 결측 | 설명 |
|---|---|---|---|---|
| `we_before` | float | [0, 1] | 없음 | 타석 시작 전 공격팀 승리확률 |
| `we_after` | float | [0, 1] | 없음 | 타석 종료 후 공격팀 승리확률 |
| `inning_ended` | bool | True/False | 없음 | 해당 PA로 이닝이 종료됐는지 |

### 메타 / 참고용

| 컬럼 | 타입 | 비고 |
|---|---|---|
| `reward_wpa` | float | 네이버 원본 WPA. **89.46% 결측, 비표준 스케일(−21~+59). 사용 비권장, 검증 참고용만** |
| `data_quality_flag` | str | `""` (정상 11,937건) / `"inning1_nonzero_start"` (47건, 학습 제외 권장) |
| `game_id` | str | `YYYYMMDD{Away}{Home}{N}{YYYY}` 형식 |
| `home_or_away` | int | 0=원정/초, 1=홈/말 (하단 컨벤션 섹션 필독) |
| `score_diff` | int | home − away (H1 컨벤션, 학습 시 score_diff_attacker 사용 권장) |

---

## 1-3. 컨벤션 (필독 — 재검토 금지)

이하 사실은 **선발 라인업 + 박스스코어 외부 진실**로 확정됨. 코드 내부에서 재추론하지 말 것.

| 컨벤션 | 값 |
|---|---|
| `home_or_away = 0` | 원정팀 공격 / 초(top) |
| `home_or_away = 1` | 홈팀 공격 / 말(bot) |
| `score_diff` | home_score − away_score (H1) |
| `score_diff_attacker` | 공격팀 관점 변환 완료. 추가 부호 변환 불필요 |
| `half` | "top"=초, "bot"=말 |
| 득점 반영 시점 | **1 PA 지연** (N번째 PA에서 발생한 득점은 N+1번째 PA의 score_diff에 반영) |
| `reward_wpa_computed` | we_after − we_before, **공격팀 관점** ΔWE |
| 투수(수비팀) 관점 보상 | `-reward_wpa_computed` |

> **경고**: 데이터셋 내부에 여러 부호가 일관되게 뒤집힌 구간이 존재하여 내부 추론만으로는 컨벤션을 잘못 판단할 수 있음. 외부 진실 대조가 유일한 신뢰 기준.

---

## 1-4. 알려진 한계 (발표 시 명시 필수)

1. **Option A 단순화 (부호 일치율 78.5%)**
   - 논문 Table 3.2가 1사(1 out) 한정이어서 0/1/2아웃을 모두 1사 값으로 처리함.
   - 주자 있는 OUT/GDP/SO에서 ΔWE가 약하게 평가됨.
   - Spearman ρ = 0.676으로 순위 일관성은 강함. 절대 스케일 오차는 인지하고 사용.

2. **이닝 보간**
   - 1·2·4·5·6·8·9회 WE는 3회/7회 값에서 선형 보간됨.
   - 9회말 끝내기는 별도 로직으로 정확하게 처리됨.

3. **득점 1스텝 지연 설계**
   - 네이버 API 특성상 득점이 다음 PA부터 score_diff에 반영됨.
   - `runs_scored`는 이를 역산해 정확히 계산됨.
   - 학습 시 `score_diff_attacker`는 "타석 시작 시점" 값임을 인지할 것.

4. **데이터 품질 플래그 47건**
   - `inning1_nonzero_start`: 1회 첫 PA 시작 시 score_diff ≠ 0 (크롤링 누락 추정).
   - 학습 시 `df[df["data_quality_flag"] == ""]`로 제외 권장.

5. **네이버 reward_wpa 사용 금지**
   - 89.46% 결측 + 비표준 스케일(min=−21, max=+59, 단위 불명).
   - `reward_wpa_computed` 검증 reference로만 보존됨.

---

## 1-5. 재현 명령

```bash
# 1. 환경 설정
uv sync

# 2. 데이터 재생성 (필요 시, 순서 중요)
uv run python data_analysis/methods/run_hsk.py          # 크롤링 + PA 집계 (~30분)
uv run python -m data_analysis.methods.state_transition # 상태 변수(we_before 등) 추가
uv run python -m data_analysis.methods.inject_wpa       # reward_wpa_computed 계산
uv run python -m data_analysis.methods.validate_wpa     # Phase 4 검증 리포트 재생성

# 3. 학습용 데이터 로드 예시
import pandas as pd

df = pd.read_parquet("data_analysis/results/hsk_pa_with_wpa.parquet")
df_clean = df[df["data_quality_flag"] == ""]  # 47건 inning1_nonzero_start 제외

X = df_clean[[
    "inning", "half", "score_diff_attacker", "base_state",
    "out_count", "total_pitch_count", "inning_pitch_count",
    "batter_hit_type", "pitcher_vs_batter_avg", "batter_recent_avg",
]]
y_pa_result = df_clean["pa_result"]          # 다클래스 분류
y_reward    = df_clean["reward_wpa_computed"] # 회귀
```

---

## 1-6. WE 조회 예시 (Monte Carlo / DQN)

```python
from data_analysis.methods.we_re_lookup import get_we_with_boundary

# 시뮬레이션 중 상태 조회 예시
we = get_we_with_boundary(
    inning=7, half="bot", score_diff=2, out_count=1, base_state="13"
)
# we: [0, 1] 범위의 공격팀 승리확률

# reward_wpa_computed 직접 계산 예시
we_before = get_we_with_boundary(inning=7, half="bot", score_diff=0, out_count=0, base_state="0")
we_after  = get_we_with_boundary(inning=7, half="bot", score_diff=3, out_count=0, base_state="1")
delta_we  = we_after - we_before  # 공격팀 관점 ΔWE
```

---

## 1-7. 다음 단계 (팀원 우선순위)

원래 3단계 아키텍처:

```
CatBoost
  → pa_result 다클래스 분류 (또는 reward_wpa_computed 직접 회귀)
  → PA 결과별 확률 벡터 P(pa_result | state) 출력

Monte Carlo
  → CatBoost 확률로 PA 시퀀스 시뮬레이션
  → 경기 종료까지 ΔWE 누적 → 분포 산출

DQN
  → E[ΔWE]와 Var[ΔWE]로 마코위츠 효용 최적화
  → U = E[ΔWE] - λ · Var[ΔWE]
  → 타석 전략(번트/적극공격 등) 선택
```

### CatBoost feature 권장 설정

```python
cat_features = ["half", "base_state", "batter_hit_type", "pitcher_id", "batter_id"]
num_features = [
    "inning", "score_diff_attacker", "out_count",
    "total_pitch_count", "inning_pitch_count",
    "pitcher_vs_batter_avg", "batter_recent_avg",
]
# 결측 대체: pitcher_vs_batter_avg, batter_recent_avg → 0.250 (리그 평균)
# 나머지 수치형 결측 → 0
```

---

## 1-8. 디버깅 이력 (발표 자료용)

이번 데이터 구축 과정의 주요 발견 사항:

| 발견 | 원인 | 해결 |
|---|---|---|
| 네이버 API가 반이닝을 시간 역순으로 반환 | API 설계 | `parser.py`에서 `reversed()` 정렬 보정 |
| `home_or_away` 코드가 직관과 반대 | Naver API 명세 확인 | 선발 라인업·박스스코어 외부 진실로 확정 |
| `score_diff` 1스텝 지연 | API 특성 | 설계로 인정, `runs_scored`로 역산 보완 |
| 동일 방향 다중 부호 오류가 단위 테스트를 통과 | 일관된 부호 반전은 테스트로 탐지 불가 | 도메인 외부 진실(경기 결과) 대조 필수 |
| Option A 1아웃 단순화로 부호 일치율 78.5% | WE 테이블이 1아웃 한정 | 한계로 명시, Phase 5(Option B) 보류 |
