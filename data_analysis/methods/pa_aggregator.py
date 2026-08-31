"""
PA Aggregator
  - 투구 단위(pitch-level) DataFrame을 타석 단위(PA-level)로 집계한다.
  - 단일 책임: 집계만 담당. 크롤링·저장 로직은 건드리지 않는다.
  - 핵심 함수: aggregate_pa(df) -> pd.DataFrame

연속 그룹 인덱스(_pa_seq)를 사용하여 동일 이닝 내 같은 투수-타자 조합이
두 번 나올 경우(타순 순환) 두 타석이 하나로 합쳐지는 오류를 방지한다.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  정규표현식 기반 타석 결과 분류
#
#  unk_audit.py(2017-2025 9개년 전수 감사)로 확인된 표현 변형을 반영:
#    - HBP: "몸에 맞는 공"뿐 아니라 "몸에 맞는 볼"도 쓰임(연 700~950건)
#    - IBB: "고의사구"뿐 아니라 "고의4구"도 쓰임(연 55~185건)
#    - 1B : "1루타" 외에 "OO수 왼쪽/앞 내야안타", "번트안타"도 안타(연 900~1000건)
#    - ROE: 실책으로 출루/야수선택으로 출루/타격방해로 출루/플라이 실책/
#           라인드라이브 실책 — 전부 "타자가 1루 도달"이라는 결과는 동일.
#           야수선택의 선행주자 아웃률을 3개년 실측한 결과 0.7%(145건 중 1건)로
#           5% 미만이어서 1B 병합 채택(out_count/base_state는 pa_result 라벨과
#           무관하게 API 원본 상태값으로 산정되므로 병합이 WE 계산에 영향 없음).
# ────────────────────────────────────────────────────────────────────────

# 세분화된 원본 분류 — pa_result_raw 컬럼에 그대로 보존된다.
PA_RESULT_PATTERNS_RAW = [
    (r"홈런",                                                    "HR"),
    (r"3루타",                                                   "3B"),
    (r"2루타",                                                   "2B"),
    (r"1루타|내야안타|번트안타",                                  "1B"),
    (r"고의사구|고의4구",                                         "IBB"),
    (r"볼넷",                                                    "BB"),
    (r"몸에 맞는 공|몸에 맞는 볼|사구",                            "HBP"),
    (r"실책으로 출루|야수선택으로 출루|타격방해로 출루|플라이 실책|라인드라이브 실책", "ROE"),
    (r"삼진",                                                    "SO"),
    (r"병살타",                                                  "GDP"),
    (r"희생플라이|희생타",                                        "SF"),
    (r"땅볼|뜬공|파울플라이|내야플라이|파울 아웃|아웃",             "OUT"),
]

_COMPILED_PATTERNS_RAW = [(re.compile(p), label) for p, label in PA_RESULT_PATTERNS_RAW]

# 메인 pa_result로 병합되는 규칙 — HBP/IBB→BB(출루 이벤트 동일 취급),
# ROE→1B(타자 1루 도달이라는 결과 동일). 매핑에 없는 라벨은 그대로 사용.
_MERGE_TO_MAIN = {"HBP": "BB", "IBB": "BB", "ROE": "1B"}


def _classify_pa_result_raw(text: str) -> str:
    """relay_text를 받아 세분화된 타석 결과 레이블을 반환. 미매칭 시 'UNK'."""
    if not isinstance(text, str) or not text.strip():
        return "UNK"
    for pattern, label in _COMPILED_PATTERNS_RAW:
        if pattern.search(text):
            return label
    return "UNK"


def _classify_pa_result(text: str) -> str:
    """relay_text를 받아 병합된(메인) 타석 결과 레이블을 반환. 미매칭 시 'UNK'."""
    raw = _classify_pa_result_raw(text)
    return _MERGE_TO_MAIN.get(raw, raw)


# ────────────────────────────────────────────────────────────────────────
#  PA 집계 메인 함수
# ────────────────────────────────────────────────────────────────────────

def aggregate_pa(df: pd.DataFrame) -> pd.DataFrame:
    """
    투구 단위 DataFrame → 타석 단위 DataFrame으로 집계.

    Parameters
    ----------
    df : pitch-level DataFrame (relay_text 컬럼 필수)

    Returns
    -------
    pa_df : PA-level DataFrame
    """
    df = df.copy()

    # relay_text가 없으면 빈 문자열로 채움 (하위 호환)
    if "relay_text" not in df.columns:
        logger.warning("relay_text 컬럼 없음 — 빈 문자열로 대체")
        df["relay_text"] = ""

    # pa_result 생성 (relay_text는 타석 내 모든 행에 동일하게 저장됨)
    # pa_result_raw: 병합 전 세분화 라벨(HBP/IBB/ROE 등) 보존
    df["pa_result_raw"] = df["relay_text"].apply(_classify_pa_result_raw)
    df["pa_result"] = df["pa_result_raw"].map(lambda r: _MERGE_TO_MAIN.get(r, r))

    # ── 연속 그룹 인덱스(_pa_seq) 부여 ──────────────────────────────────
    # 동일 이닝에서 같은 투수-타자 조합이 두 번 나오는 경우(타순 순환)를
    # 별도 타석으로 분리하기 위해 연속 변화 감지(cumsum) 방식 사용.
    key_cols = ["game_id", "inning", "pitcher_id", "batter_id"]
    # 존재하는 key_cols만 사용 (방어 코드)
    key_cols = [c for c in key_cols if c in df.columns]
    df["_pa_seq"] = (
        df[key_cols].ne(df[key_cols].shift()).any(axis=1).cumsum()
    )

    # ── 타석 첫 투구 기준 상태 변수 ──────────────────────────────────────
    first_cols = [
        c for c in [
            "game_id", "inning", "home_or_away",
            "pitcher_id", "batter_id", "batter_hit_type",
            "pitcher_vs_batter_avg", "batter_recent_avg",
            "score_diff", "out_count",
            "is_base1", "is_base2", "is_base3",
            "total_pitch_count", "inning_pitch_count",
        ] if c in df.columns
    ]
    pa_first = df.groupby("_pa_seq")[first_cols].first()

    # ── 타석 마지막 투구 기준 타겟/보상 ───────────────────────────────────
    last_cols = [c for c in ["pa_result", "pa_result_raw", "reward_wpa"] if c in df.columns]
    pa_last = df.groupby("_pa_seq")[last_cols].last()

    # ── 집계 피처 ────────────────────────────────────────────────────────
    agg_dict = {}
    if "pitch_result" in df.columns:
        agg_dict["pitches_per_pa"] = ("pitch_result", "count")
    if "pitch_speed" in df.columns:
        agg_dict["pa_avg_pitch_speed"] = ("pitch_speed", "mean")

    if agg_dict:
        pa_agg = df.groupby("_pa_seq").agg(**agg_dict)
        pa_df = pd.concat([pa_first, pa_last, pa_agg], axis=1).reset_index(drop=True)
    else:
        pa_df = pd.concat([pa_first, pa_last], axis=1).reset_index(drop=True)

    # ── UNK 비율 QA 로깅 ────────────────────────────────────────────────
    if "pa_result" in pa_df.columns:
        unk_rate = (pa_df["pa_result"] == "UNK").mean()
        logger.info("pa_result UNK 비율: %.1f%% (%d / %d 타석)",
                    unk_rate * 100, (pa_df["pa_result"] == "UNK").sum(), len(pa_df))
        if unk_rate > 0.05:
            logger.warning(
                "pa_result UNK 비율 %.1f%% — 5%% 초과, regex 패턴 보강 필요",
                unk_rate * 100
            )

    return pa_df
