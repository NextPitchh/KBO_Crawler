# Plan 2: WE/RE 기반 `reward_wpa` 계산 파이프라인 구축

## 개요

`hsk_pa.parquet`의 `reward_wpa` 결측치(89.46%)를
문형우 외(2016) 논문(asset/한국프로야구경기에서기대득점과기대승리확률의계산.pdf)의 마르코프연쇄 기반 RE/WE 테이블을 이용해 직접 계산한
ΔWE 값으로 채워 넣는다.

```
[현재]   hsk_pa.parquet (reward_wpa 결측 89%)
[목표]   hsk_pa.parquet + reward_wpa_computed (결측 0%)
         + 검증 리포트 (네이버 reward_wpa와의 상관관계)
```

> **본 plan의 범위**: 데이터 구축까지만. CatBoost/Monte Carlo/DQN 학습은 팀원
> 담당 영역이므로 본 문서에서 다루지 않는다.

---

## 이론적 근거 및 설계 결정

### 왜 ΔWE인가 (RE24가 아니라)

- **마코위츠 효용 U = E[WPA] − λ·Var[WPA]**의 WPA는 정의상 *승리 확률* 변동량이어야 함
- RE24(ΔRE + runs)는 단위가 "점수"이므로 다른 지표
- 따라서 보상 = ΔWE = `WE(after_state) − WE(before_state)` (공격팀 관점)

### 왜 논문의 테이블을 차용하는가

- 153경기로 24-state RE / 약 3,672-state WE를 자체 추정 시 표본 부족
- 논문은 KBO 2007–2012 6년치(약 4,000경기)로 추정한 값을 제공 → 신뢰 가능한 prior
- 졸업 프로젝트 셀링포인트로도 "공식 학술 논문 기반 보상 함수" 서사가 강함

### 네이버 `reward_wpa`는 어떻게 처리하는가

- 그대로 신뢰하지 않는다 (스케일 [-17.5, +23.8]은 표준 WPA(±1) 대비 비표준)
- 단, **검증(validation)용 reference**로 사용
- 결과 컬럼 두 개를 모두 보존:
  - `reward_wpa` (원본, 네이버 제공값, 결측 89%)
  - `reward_wpa_computed` (논문 기반 계산값, 결측 0%)
- 다운스트림(CatBoost/DQN)에서 사용할 컬럼은 `reward_wpa_computed`

---

## Phase 1 — RE/WE 룩업 모듈: `data_analysis/methods/we_re_lookup.py`

### 1-1. RE 테이블 (24-state)

논문 Table 2.1의 **MC 컬럼**을 사용. 상태키는 `(out_count, base_state)`.

```python
# 주자상태 표기: 8가지 (논문 §2.1과 동일)
# '0' = 주자 없음, '1' = 1루, '12' = 1·2루, '123' = 만루, 등

RE_TABLE: dict[tuple[int, str], float] = {
    # out=0
    (0, "0"):   0.549,
    (0, "1"):   1.042,
    (0, "2"):   1.258,
    (0, "3"):   1.536,
    (0, "12"):  1.756,
    (0, "13"):  2.053,
    (0, "23"):  2.225,
    (0, "123"): 2.734,
    # out=1
    (1, "0"):   0.267,
    (1, "1"):   0.556,
    (1, "2"):   0.700,
    (1, "3"):   1.042,
    (1, "12"):  0.972,
    (1, "13"):  1.342,
    (1, "23"):  1.490,
    (1, "123"): 1.662,
    # out=2
    (2, "0"):   0.095,
    (2, "1"):   0.215,
    (2, "2"):   0.316,
    (2, "3"):   0.360,
    (2, "12"):  0.439,
    (2, "13"):  0.494,
    (2, "23"):  0.581,
    (2, "123"): 0.729,
}

# 이닝 종료 상태 (3아웃) → RE = 0
def get_re(out_count: int, base_state: str) -> float:
    if out_count >= 3:
        return 0.0
    return RE_TABLE[(out_count, base_state)]
```

### 1-2. WE 테이블 (이닝 × 점수차 × 아웃 × 주자)

논문 Table 3.2는 **3회/7회의 무사·1사·2사 × 8주자 × 점수차(-4~+4)**만 명시.
따라서 이닝 차원의 **결측 처리**가 필수.

**제공되는 셀:**

- 이닝: 3회초, 3회말, 7회초, 7회말 (각 1사 한정 — Table 3.2)
- 추가로 Table 3.1에서 3회말/7회말 무사·1사·2사 (주자 없음, 점수차 -2 ~ +2) 일부

**미제공 셀의 보간 전략:**

| 차원   | 범위  | 미제공 처리                                                                            |
| ------ | ----- | -------------------------------------------------------------------------------------- |
| 이닝   | 1~9회 | 3회·7회 값으로부터**선형 보간**; 1·2회 = 3회 값 외삽, 8·9회 = 7회 + 추가 외삽 |
| 점수차 | 정수  | 절댓값이 5 이상이면 ±4 값으로**clip** (논문 조건 (1)에 의해 monotonic 보장)     |
| 아웃   | 0/1/2 | Table 3.2는 1사만 제공. 0사·2사는 별도 처리 (아래)                                    |
| 주자   | 8가지 | 모두 제공                                                                              |
| 초/말  | 2가지 | 모두 제공                                                                              |

**아웃 차원 처리 (중요):**

논문 Table 3.2는 **1사 상태만** 모든 점수차/주자에 대해 제공한다. 0사·2사를 위해서는 두 가지 옵션:

- **Option A (단순)**: 1사 값을 0사·2사에도 사용 (아웃 차원 영향 무시)
- **Option B (정합)**: 무사 데이터(Table 3.1 부분 + 외삽)와 ΔRE 차이를 이용해 0사·2사 값을 보정
  - 예: `WE(0사, s, j) ≈ WE(1사, s, j) + α · (RE(0사, s) − RE(1사, s))`
  - α는 "득점 1점이 승률에 미치는 영향" 비례 상수 (점수차·이닝별 다름)

**채택: Option A로 시작, Phase 4 검증에서 네이버 WPA와 상관계수가 낮으면 Option B로 보완.**

### 1-3. WE 테이블 구현 골격

```python
import numpy as np
from typing import Literal

Half = Literal["top", "bot"]  # 초/말

# 논문 Table 3.2 원본 (1사, 3회/7회 × 초/말 × 점수차 -4~+4 × 8주자)
# shape: (이닝=2, 초말=2, 점수차=9, 주자=8)
WE_RAW_1OUT_INN3_TOP = {  # 3회초 1사
    "0":   [0.078, 0.134, 0.216, 0.313, 0.449, 0.613, 0.707, 0.825, 0.896],
    "1":   [0.100, 0.162, 0.250, 0.353, 0.486, 0.641, 0.732, 0.840, 0.904],
    "2":   [0.108, 0.174, 0.264, 0.372, 0.510, 0.654, 0.749, 0.850, 0.908],
    "3":   [0.127, 0.202, 0.297, 0.418, 0.567, 0.686, 0.790, 0.874, 0.918],
    "12":  [0.133, 0.205, 0.300, 0.407, 0.538, 0.677, 0.764, 0.859, 0.915],
    "13":  [0.155, 0.236, 0.336, 0.457, 0.597, 0.711, 0.807, 0.884, 0.926],
    "23":  [0.169, 0.252, 0.358, 0.481, 0.606, 0.728, 0.813, 0.886, 0.929],
    "123": [0.194, 0.278, 0.378, 0.491, 0.616, 0.731, 0.812, 0.888, 0.931],
}
# WE_RAW_1OUT_INN3_BOT (3회말), WE_RAW_1OUT_INN7_TOP (7회초),
# WE_RAW_1OUT_INN7_BOT (7회말) — 동일 구조로 plan2.md 작성 시 입력

SCORE_DIFF_RANGE = np.arange(-4, 5)  # -4, -3, ..., +4

def _clip_score_diff(j: int) -> int:
    return int(np.clip(j, -4, 4))

def _interpolate_inning(inning: int, half: Half) -> tuple[float, float]:
    """
    주어진 (이닝, 초/말)에 대해 3회와 7회 사이의 선형 보간 가중치 반환.
    반환: (w3, w7) — WE = w3 * WE_inn3 + w7 * WE_inn7
    """
    # 1~3회: w3=1, w7=0 (외삽 대신 3회값 사용)
    # 3~7회: 선형 보간
    # 7~9회: w3=0, w7=1 (외삽 대신 7회값 사용; 9회는 조정 필요할 수 있음)
    if inning <= 3:
        return (1.0, 0.0)
    if inning >= 7:
        return (0.0, 1.0)
    # 3 < inning < 7
    w7 = (inning - 3) / 4.0
    return (1.0 - w7, w7)

def get_we(
    inning: int,
    half: Half,           # "top" (초, 원정공격) or "bot" (말, 홈공격)
    score_diff: int,      # 공격팀 관점 점수차
    out_count: int,
    base_state: str,
) -> float:
    """
    공격팀이 (이닝, 초말, 점수차, 아웃, 주자) 상태에서 경기를 계속할 때
    승리할 확률 P(W_attack).
    """
    # 이닝 종료 후의 처리는 별도 (Phase 2에서 다룸)
    j = _clip_score_diff(score_diff)
    j_idx = j + 4  # -4 → 0, ..., +4 → 8

    w3, w7 = _interpolate_inning(inning, half)
    table3 = _WE_INN3_TOP if half == "top" else _WE_INN3_BOT
    table7 = _WE_INN7_TOP if half == "top" else _WE_INN7_BOT

    we3 = table3[base_state][j_idx]
    we7 = table7[base_state][j_idx]
    we = w3 * we3 + w7 * we7

    # Option A: 아웃 차원 무시 (모든 아웃카운트에 1사 값 사용)
    return float(we)
```

### 1-4. 9회말 끝내기 / 정규이닝 종료 경계 처리

- **9회말 끝내기 (Walk-off)**: 홈팀이 9회말 도중 점수차 > 0이 되면 WE = 1.0 즉시 적용
- **9회 종료 동점**: 연장으로 처리. 본 plan에서는 연장 데이터는 무시 (HSK 데이터 내 발생 빈도 낮음)
- **무승부 (KBO 12회 제한)**: 동점 종료 시 WE = 0.5 적용

```python
def get_we_with_boundary(
    inning: int, half: Half, score_diff: int,
    out_count: int, base_state: str,
) -> float:
    # 9회말 끝내기
    if inning == 9 and half == "bot" and score_diff > 0:
        return 1.0
    # 12회 (연장 마지막) 종료 시
    if inning >= 12 and out_count >= 3 and base_state == "0":
        if score_diff > 0:  return 1.0
        if score_diff < 0:  return 0.0
        return 0.5
    return get_we(inning, half, score_diff, out_count, base_state)
```

### 1-5. 단위 테스트 (sanity check)

```python
def test_we_monotonic_score():
    """점수차가 클수록 WE가 커야 함 (논문 조건 W1)"""
    we_minus = get_we(5, "bot", -2, 1, "0")
    we_zero  = get_we(5, "bot",  0, 1, "0")
    we_plus  = get_we(5, "bot", +2, 1, "0")
    assert we_minus < we_zero < we_plus

def test_we_monotonic_inning_when_ahead():
    """앞서고 있을 때 이닝이 진행될수록 WE가 커야 함 (논문 조건 W3)"""
    we_3rd = get_we(3, "bot", +1, 1, "0")
    we_7th = get_we(7, "bot", +1, 1, "0")
    assert we_3rd <= we_7th

def test_we_bounds():
    """WE는 [0, 1] 범위여야 함"""
    assert 0.0 <= get_we(5, "bot", 0, 1, "0") <= 1.0
```

---

## Phase 2 — `before_state` / `after_state` 추출: `data_analysis/methods/state_transition.py`

### 2-1. 핵심 도전 과제

**`hsk_pa.parquet`의 각 행은 PA 시작 시점의 상태만 기록한다.** ΔWE 계산을 위한
`after_state`는 **다음 PA의 시작 상태**에서 가져와야 한다.

```
타석 N의 after_state = 타석 N+1의 before_state    (같은 이닝 내)
타석 N의 after_state = INNING_END                  (이닝 마지막 타석)
```

### 2-2. 이닝 마지막 타석 처리

이닝 마지막 타석은 다음 PA가 다른 이닝 또는 다른 경기에 속한다. 이 경우의
`after_state` 결정 규칙:

| 이닝 마지막 타석의 결과 | after_state                                   |
| ----------------------- | --------------------------------------------- |
| 3아웃으로 이닝 종료     | `out=3, base="0", score_diff=before + 득점` |
| 끝내기로 경기 종료      | `WE = 1.0` 직접 부여 (특수 처리)            |

이닝 종료 상태의 WE는 **다음 이닝 시작 상태**로 환산해야 한다.

- 다음 이닝의 공수가 바뀐다 → 점수차 부호 반전
- 공격팀 입장에서: `WE_attack(end_of_inning_top) = 1 - WE_defense_next(start_of_inning_bot)`

```python
def compute_after_state_we(
    pa_row: pd.Series,
    next_pa_row: pd.Series | None,
    inning_ended: bool,
    runs_scored_in_pa: int,
) -> float:
    """
    PA 종료 시점의 WE를 공격팀 관점에서 계산.
    """
    cur_inning = pa_row["inning"]
    cur_half = "top" if pa_row["home_or_away"] == 0 else "bot"
    cur_score_diff_after = pa_row["score_diff"] + runs_scored_in_pa

    if not inning_ended:
        # 다음 타석의 before_state 사용
        next_out = next_pa_row["out_count"]
        next_base = _base_str(next_pa_row)
        return get_we(cur_inning, cur_half, cur_score_diff_after,
                      next_out, next_base)

    # 이닝 종료 → 다음 이닝/공수 시작 시점의 WE
    next_inning = cur_inning + (1 if cur_half == "bot" else 0)
    next_half = "bot" if cur_half == "top" else "top"
    # 공수 전환: 다음 이닝의 공격팀 관점 점수차 = -현재 공격팀 관점 점수차
    next_score_diff = -cur_score_diff_after

    we_next_attack = get_we_with_boundary(
        next_inning, next_half, next_score_diff,
        out_count=0, base_state="0",
    )
    # 현재 공격팀 관점으로 변환
    return 1.0 - we_next_attack
```

### 2-3. 득점 수 계산 (`runs_scored_in_pa`)

현재 `hsk_pa.parquet`에는 타석별 득점 수가 명시되지 않는다. 두 가지 접근:

**Option 1: `score_diff`의 차이로 역산**

```python
df["runs_scored"] = df.groupby("game_id")["score_diff"].diff().shift(-1)
# 단, 공수 전환 시 부호 반전 처리 필요
```

문제: 공수 전환 시점에서 점수차의 의미가 바뀌므로 단순 diff는 부정확.

**Option 2: `relay_text` 정규식으로 명시 파싱**

```python
RUN_PATTERN = re.compile(r"(\d+)점")  # "3점 홈런", "1점 적시타"
# 홈런은 base 상태 + 1점 (만루 홈런 = 4점)
```

문제: relay_text 표현이 다양해 정규식 커버리지 불완전 가능.

**채택: Option 1 + 검증 — 공수 전환 경계는 별도 처리, 같은 (game_id, inning, half) 내에서만 diff 적용.**

```python
def compute_runs_scored(df: pd.DataFrame) -> pd.Series:
    """
    각 PA에서 득점한 점수 수.
    같은 (game_id, inning, home_or_away) 그룹 내에서만 score_diff diff.
    그룹 마지막 PA는 NaN → relay_text로 후처리 또는 0 가정.
    """
    df = df.sort_values(["game_id", "inning", "home_or_away"]).copy()
    df["_next_score_diff"] = df.groupby(
        ["game_id", "inning", "home_or_away"]
    )["score_diff"].shift(-1)
    df["runs_scored"] = (df["_next_score_diff"] - df["score_diff"]).fillna(0)
    # 음수 발생 시 (점수차 부호 변화) 0으로 보정 + 로깅
    neg_mask = df["runs_scored"] < 0
    if neg_mask.any():
        logger.warning("음수 runs_scored %d건 → 0으로 보정", neg_mask.sum())
        df.loc[neg_mask, "runs_scored"] = 0
    return df["runs_scored"].astype(int)
```

### 2-4. base_state 인코딩

`hsk_pa.parquet`의 `is_base1/2/3` (0/1 플래그) → 논문 표기 문자열:

```python
def base_str(is_b1: int, is_b2: int, is_b3: int) -> str:
    parts = []
    if is_b1: parts.append("1")
    if is_b2: parts.append("2")
    if is_b3: parts.append("3")
    return "".join(parts) if parts else "0"
```

### 2-5. inning_ended 판별

```python
def detect_inning_end(df: pd.DataFrame) -> pd.Series:
    """
    같은 (game_id, inning, home_or_away) 그룹의 마지막 PA = inning_ended.
    또는 PA 결과로 out_count + (0/1/2) >= 3 인 타석 = inning_ended.
    """
    df = df.sort_values(["game_id", "inning", "home_or_away"])
    is_last_in_group = ~df.duplicated(
        subset=["game_id", "inning", "home_or_away"], keep="last"
    )
    return is_last_in_group
```

---

## Phase 3 — WPA 계산 및 주입: `data_analysis/methods/inject_wpa.py`

### 3-1. 메인 파이프라인

```python
def inject_computed_wpa(
    input_path: str = "data_analysis/results/hsk_pa.parquet",
    output_path: str = "data_analysis/results/hsk_pa_with_wpa.parquet",
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)

    # Step 1: base_state 문자열 생성
    df["base_state"] = df.apply(
        lambda r: base_str(r["is_base1"], r["is_base2"], r["is_base3"]),
        axis=1,
    )

    # Step 2: half (top/bot) 결정
    df["half"] = df["home_or_away"].map({0: "top", 1: "bot"})

    # Step 3: before_state WE 계산
    df["we_before"] = df.apply(
        lambda r: get_we_with_boundary(
            r["inning"], r["half"], r["score_diff"],
            r["out_count"], r["base_state"]
        ),
        axis=1,
    )

    # Step 4: runs_scored 계산
    df["runs_scored"] = compute_runs_scored(df)

    # Step 5: inning_ended 판별
    df["inning_ended"] = detect_inning_end(df)

    # Step 6: after_state WE 계산 (다음 PA 참조)
    df = df.sort_values(["game_id", "inning", "home_or_away"]).reset_index(drop=True)
    df["next_out_count"] = df.groupby(
        ["game_id", "inning", "home_or_away"]
    )["out_count"].shift(-1)
    df["next_base_state"] = df.groupby(
        ["game_id", "inning", "home_or_away"]
    )["base_state"].shift(-1)

    df["we_after"] = df.apply(_compute_we_after_row, axis=1)

    # Step 7: ΔWE = reward_wpa_computed
    df["reward_wpa_computed"] = df["we_after"] - df["we_before"]

    # 임시 컬럼 정리
    df = df.drop(columns=["next_out_count", "next_base_state", "_next_score_diff"],
                 errors="ignore")

    df.to_parquet(output_path, index=False)
    return df
```

### 3-2. 출력 스키마 추가 컬럼

| 컬럼                    | 타입  | 설명                                         |
| ----------------------- | ----- | -------------------------------------------- |
| `base_state`          | str   | 주자상태 문자열 ("0", "1", "12", ..., "123") |
| `half`                | str   | "top" / "bot"                                |
| `we_before`           | float | 타석 시작 시점의 공격팀 WE                   |
| `we_after`            | float | 타석 종료 시점의 공격팀 WE                   |
| `runs_scored`         | int   | 타석에서 득점한 점수                         |
| `inning_ended`        | bool  | 이 타석으로 이닝이 종료되었는지              |
| `reward_wpa_computed` | float | ΔWE = we_after − we_before                 |

기존 `reward_wpa` (네이버 원본) 컬럼은 **그대로 보존**.

---

## Phase 4 — 검증: `data_analysis/methods/validate_wpa.py`

### 4-1. 자체 sanity check

```python
def validate_self():
    df = pd.read_parquet("data_analysis/results/hsk_pa_with_wpa.parquet")

    # 1. reward_wpa_computed 결측률 = 0%
    assert df["reward_wpa_computed"].isna().sum() == 0

    # 2. 범위: 단일 PA의 ΔWE는 [-1, +1] 이내
    assert df["reward_wpa_computed"].between(-1.0, 1.0).all()

    # 3. 평균은 0 근처 (게임 전체로 보면 승자 +0.5, 패자 -0.5 의 합)
    print("ΔWE 평균:", df["reward_wpa_computed"].mean())  # 약 0

    # 4. 타석 결과별 평균 ΔWE 부호 일관성
    by_result = df.groupby("pa_result")["reward_wpa_computed"].mean()
    print(by_result)
    # 기대: HR > 3B > 2B > 1B > BB > OUT > SO
```

### 4-2. 네이버 원본과 비교 (결측이 아닌 ~11%만)

```python
def validate_vs_naver():
    df = pd.read_parquet("data_analysis/results/hsk_pa_with_wpa.parquet")
    naver = df[df["reward_wpa"].notna()].copy()
    print(f"비교 가능 표본: {len(naver)}타석")

    # 부호 일치율
    sign_match = (
        np.sign(naver["reward_wpa"]) == np.sign(naver["reward_wpa_computed"])
    ).mean()
    print(f"부호 일치율: {sign_match:.2%}")

    # Spearman 상관계수 (스케일 무시, 순위 일관성만 비교)
    rho, p = scipy.stats.spearmanr(
        naver["reward_wpa"], naver["reward_wpa_computed"]
    )
    print(f"Spearman ρ = {rho:.3f} (p = {p:.4f})")

    # 스캐터 플롯 저장
    plt.scatter(naver["reward_wpa"], naver["reward_wpa_computed"], alpha=0.3)
    plt.xlabel("네이버 reward_wpa (원본)")
    plt.ylabel("논문 기반 reward_wpa_computed")
    plt.savefig("data_analysis/results/wpa_validation_scatter.png")
```

### 4-3. 합격 기준

- [ ] **결측률**: `reward_wpa_computed` 결측 = 0
- [ ] **범위**: 모든 ΔWE 값이 [-1, +1] 이내
- [ ] **부호 일치율**: 네이버 원본 대비 ≥ 80%
- [ ] **Spearman ρ**: ≥ 0.6 (스케일은 달라도 순위는 일치해야)
- [ ] **타석 결과별 평균 부호 순서**: HR > 1B > BB > OUT 만족

위 5가지가 모두 통과하면 Phase A(Option A) 완료. 부호 일치율이 80% 미만이거나
ρ < 0.6이면 **Phase 5로 진행**(Option B: 아웃 차원 보정).

---

## Phase 5 — (조건부) 정밀화: Option B 아웃 차원 보정

Phase 4에서 네이버 WPA와의 일치도가 낮으면, 1사 값만 사용한 단순화가 원인일 가능성이 높다. 이 경우:

```python
# WE(out, s, j, i) ≈ WE(1사, s, j, i) + α * (RE(out, s) − RE(1사, s))
# α는 점수차/이닝별 조정 계수
def get_we_with_out_correction(
    inning, half, score_diff, out_count, base_state
):
    base_we = get_we(inning, half, score_diff, out_count, base_state)  # 1사값
    re_delta = (RE_TABLE[(out_count, base_state)]
                - RE_TABLE[(1, base_state)])
    alpha = _get_alpha(inning, score_diff)  # 휴리스틱 또는 회귀로 추정
    return np.clip(base_we + alpha * re_delta, 0.0, 1.0)
```

α 추정은 별도 데이터 분석이 필요하므로 본 plan에서는 골격만 정의한다.

---

## 작업 순서 및 파일 변경 목록

```
Step 1  data_analysis/methods/we_re_lookup.py       (신규) — RE/WE 룩업
Step 2  data_analysis/methods/state_transition.py   (신규) — before/after 추출
Step 3  data_analysis/methods/inject_wpa.py         (신규) — WPA 계산·주입 메인
Step 4  data_analysis/methods/validate_wpa.py       (신규) — 검증 스크립트
Step 5  data_analysis/results/hsk_pa_with_wpa.parquet (산출) — 최종 데이터
Step 6  data_analysis/results/wpa_validation_report.md (산출) — 검증 리포트
```

---

## 다음 단계 핸드오프 (팀원 영역)

본 plan 완료 시점에서 팀원에게 전달되는 산출물:

1. **`hsk_pa_with_wpa.parquet`**: `reward_wpa_computed` 컬럼이 추가된 11,984 PA 데이터
2. **검증 리포트**: 네이버 원본과의 일치도, 타석 결과별 ΔWE 분포
3. **`we_re_lookup.py` 모듈**: 다운스트림에서 (상태) → WE 조회용 함수 노출

이후 팀원은:

- CatBoost: `reward_wpa_computed` 또는 `pa_result` one-hot을 타겟으로 학습
- Monte Carlo: `get_we_with_boundary()` 호출로 시뮬레이션 시점의 WE 산출
- DQN: `reward_wpa_computed` 를 보상 신호로 사용, `Var[reward_wpa_computed]` 로 마코위츠 항 구성

---

## 결정 사항 요약 (Open Questions)

| 결정        | 채택                 | 대안                      | 재검토 트리거                      |
| ----------- | -------------------- | ------------------------- | ---------------------------------- |
| RE 출처     | 논문 Table 2.1 MC    | EMP 컬럼 / 자체 추정      | RE-EMP 편차가 큰 셀에 표본 집중 시 |
| WE 출처     | 논문 Table 3.2       | 자체 추정                 | 표본 4,000경기 이상 확보 시        |
| 이닝 보간   | 3회·7회 선형 + 외삽 | 자체 모든 이닝 추정       | 이닝별 데이터 충분 시              |
| 아웃 차원   | Option A (1사 값만)  | Option B (RE 보정)        | Phase 4 검증 실패 시               |
| 점수차 외삽 | ±4로 clip           | 점근값 (0/1) 외삽         | clip 경계 빈도 ≥ 5% 시            |
| 끝내기 처리 | WE=1.0 즉시          | 정상 계산                 | 부호 일치율 < 80% 시               |
| 네이버 WPA  | 검증용 보존          | 결측치 채우기에 직접 사용 | 본 plan 전면 폐기 시               |
