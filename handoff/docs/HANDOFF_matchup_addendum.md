# HANDOFF_FULL.md 추가분 — 투수·타자 매치업 이력 (2026-08-27)

## 1. `naver_vshra_raw` (구 `pitcher_vs_batter_avg`) — **사용 금지**

`all_pa_enriched_v2.parquet` 에서 `pitcher_vs_batter_avg` → `naver_vshra_raw` 로 이름만 변경(값 동일, 삭제 아님).

**사용 금지 사유:**
- 네이버 relay API `batterRecord.vsHra` 원본. 60%가 `0.000` 이며 **결측과 실제 0안타를 구분 불가**
  (자체 데이터에서 20타수 이상 맞붙었는데 `0.000` 인 행 54,631개, 그 행들의 실제 상대타율 0.289).
- 같은 경기에서 투수가 바뀌어도 99.4% 불변 → **투수 개별 정보 없음** (사실상 타자×상대 선발 값).
- 실제 상대타율과 상관 r ≈ 0.13. CatBoost v5 importance 0.86(12개 중 10위), v6 ablation Δlog-loss +0.0009(무의미).
- 시점 자체는 안전(경기일 기준 통산값, 미래 미포함)하나 위 결측 문제로 학습 피처로 부적합.
- 이름에 `vs_batter` 가 있으면 다운스트림에서 매치업 정보로 오인 → `naver_vshra_raw` 로 개명하고 **참고용으로만 보존**.

## 2. `matchup_*` 컬럼 — 사용법 (`all_pa_enriched_v2.parquet`, `matchup_history.parquet`)

자체 PA 이력으로 재구축한 leakage-free 매치업 피처. 3층 구조.

| 컬럼 | 의미 | 결측 처리 |
|---|---|---|
| `handedness_key` | `{L,R,U,UNK}_{L,R,S,U}` (투수손_타자유효좌우). U=언더핸드, UNK=투수손 결측(2016~2017 집중) | 없음 |
| `handedness_base_woba` | 좌우 조합별 리그 평균 wOBA (base 층). UNK_* 는 전체 평균으로 대체 | 없음 |
| `prior_matchup_pa` / `_ab` / `_hits` / `_1b`/`_2b`/`_3b`/`_hr` / `_bb` / `_so` / `_sf` | **그 PA 이전까지**의 (pitcher_id, batter_id) 누적 (expanding().shift(1)) | 첫 대결이면 0 |
| `prior_matchup_avg` | 누적 hits/ab | **ab==0 이면 NaN** (0.0 채우지 않음) |
| `prior_matchup_woba` | 누적 wOBA (가중치 1B .77 / 2B 1.08 / 3B 1.37 / HR 1.70 / BB .62, 분모 AB+BB+SF) | **분모 0 이면 NaN** |
| `matchup_weight` | shrinkage 가중치 `w = ab / (ab + 10)` (신뢰도, 0~1) | 첫 대결 0 |
| `matchup_woba_shrunk` | **주 피처.** `w·prior_matchup_woba + (1−w)·handedness_base_woba` | 없음 (base 항상 정의) |
| `matchup_avg_shrunk` | 위의 AVG 버전 (참고) | 없음 |
| `batter_side_inferred` | 스위치 타자를 통상 규칙(vs RHP→좌타석)으로 변환한 행 True | — |

**권장 사용:**
- CatBoost 수치 피처: `matchup_woba_shrunk` + `prior_matchup_ab` (신뢰도 신호). 둘을 함께 넣어야 모델이 소표본 매치업의 불확실성을 학습.
- `prior_matchup_avg` / `prior_matchup_woba` 원본을 직접 쓰려면 **NaN 을 반드시 그대로 두거나 명시적 결측 처리** (0.0 금지 — 그게 vsHra 의 실패 지점).
- `BASE_K = 10` 고정 (도메인 판단). 검증에서 k=5/10/20/30 비교했고 k=20 이 근소 우위였으나 차이 미미, 변경은 팀 결정 사항.

**예측력(검증 결과):** 개별 상대 전적의 PA 단위 예측력은 **약함** (표본 중앙값 3타수, 95%tile 20타수, 단일 PA 노이즈 지배).
- per-PA wOBA 회귀 R² ≈ 0 (base/shrunk/full 모두).
- **장타(XBH) 예측에서는 일관된 개선**: XBH-AUC base 0.480 → shrunk 0.513 → full 0.522. (handedness base wOBA 단독은 XBH 를 역방향 예측 → AUC<0.5. 개별 이력이 파워 신호를 복원.)
- 모든 ab 구간에서 shrunk > base (XBH-AUC). 매치업 효과는 타율보다 **장타 차원**에서 나타남.

## 3. `handedness_baseline.json`

좌우 조합별 wOBA/AVG/OBP/ISO/HR%/SO%/BB% + 메타(wOBA 가중치, BASE_K 근거, 스위치 처리, pitcher_throws 결측 구조).

**게이트 변경 기록:** 좌우 도메인 검증을 raw AVG → **wOBA** 로 교체.
- 좌완 투수의 좌타자 억제는 안타(AVG)가 아니라 장타·볼넷 억제로 나타남(세이버메트릭스 정설).
- raw AVG: 좌투 L_L(.2793) vs L_R(.2789) Δ=+.0014 → 표준오차(±.002) 미만, z≈0.5, 통계적으로 0. 게이트 실패.
- wOBA: 좌투 Δ=−.0132, 우투 Δ=−.0099. HR% L_L .0166 vs L_R .0308, ISO L_L .101 vs L_R .147. **양방향 정상.**
- → 데이터 문제 아님. AVG 게이트 설계 오류. AVG 는 참고 지표로만 기록.

## 4. 부수 발견 — `pitcher_throws` 결측 구조

- 결측 13.66%는 무작위가 아님: **2016 시즌 100%, 2017 시즌 34%, 2018+ ~0%** (preview API 미제공 구간과 일치).
- 이 구간(`handedness_key` 가 `UNK_*`)은 좌우 매치업 base 를 전체 평균으로 설정. **학습 시 2016~2017 제외 여부 별도 검토 권고.**
- `pitcher_throws == 'U'` (3~6%, 58명, 22,701행)는 **언더핸드**(정상 값). 독립 그룹 유지.
