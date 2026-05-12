"""
PitchDataParser
  - fetch_game() 이 반환한 raw JSON 을 Feature Mapping Rule 에 따라 파싱
  - 파생 변수 (recent_5_pitch_speed_avg, inning_pitch_count) 를 스크립트 내 직접 연산
  - 경기 전환 시 reset() 을 반드시 호출하여 누적 상태를 초기화해야 함

JSON 확정 경로 (수정됨):
  root
  └─ result
       └─ textRelayData
            └─ textRelays[]                ← 타석 단위 (1 relay = 1 타석)
                 ├─ homeOrAway             ← 공격 팀 판별
                 ├─ metricOption           ← WPA 등 지표
                 │    └─ wpaByPlate
                 └─ textOptions[]          ← 이벤트 배열 (타석 내 모든 투구)
                      ├─ [0]  type=8       ← 타자 등장 텍스트 (batterRecord 포함)
                      ├─ [1…N-1] type=1    ← ★ 실제 투구 이벤트 (speed, stuff, pitchResult)
                      └─ [N]  type=13/23   ← 타석 결과 텍스트
"""

import logging
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  모듈 수준 유틸
# ────────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    """None / 빈 문자열 / 변환 불가 값을 default 로 처리."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _base_flag(val) -> int:
    """루상 값이 0 / None / '0' 이면 0, 그 외 1."""
    if val is None:
        return 0
    try:
        return 0 if int(val) == 0 else 1
    except (ValueError, TypeError):
        return 0


def _parse_hit_type(raw_hit_type: str) -> Optional[str]:
    """
    타자 타석 유형을 L / R / S 로 변환한다.
    '우투좌타', '좌투좌타' → 'L'
    '우투우타', '좌투우타' → 'R'
    '양타' 포함            → 'S' (Switch Hitter)
    기타 / 빈 문자열       → None
    """
    if not raw_hit_type:
        return None
    if "양타" in raw_hit_type:
        return "S"
    if "좌타" in raw_hit_type:
        return "L"
    if "우타" in raw_hit_type:
        return "R"
    return None


# ────────────────────────────────────────────────────────────────────────
#  Parser 클래스
# ────────────────────────────────────────────────────────────────────────

class PitchDataParser:
    """
    투구 단위(Pitch-by-Pitch) Row 를 생성하는 파서.

    ── 상태 캐싱 ──────────────────────────────────────────────────────────
    _speed_cache        : {pitcher_id → deque(maxlen=5)}
                          최근 5구 구속을 저장 → recent_5_pitch_speed_avg 계산
    _pitcher_inning     : {pitcher_id → 마지막 처리 이닝}
                          이닝 변경 감지 → inning_pitch_count 리셋
    _inning_pitch_count : {pitcher_id → 현재 이닝 투구 수}
    """

    def __init__(self):
        self._speed_cache: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._pitcher_inning: dict[str, int] = {}
        self._inning_pitch_count: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """경기 전환 시 누적 상태 초기화 (반드시 호출)."""
        self._speed_cache.clear()
        self._pitcher_inning.clear()
        self._inning_pitch_count.clear()

    def parse_game(self, game_id: str, innings_data: list[dict]) -> list[dict]:
        """
        fetch_game() 반환값 전체를 받아 모든 Row 를 반환한다.

        Parameters
        ----------
        game_id     : 경기 ID 문자열
        innings_data: [{'inning': int, 'data': dict}, ...]
        """
        rows: list[dict] = []
        for item in innings_data:
            inning_num: int = item["inning"]
            raw: dict = item["data"]
            rows.extend(self._parse_inning(game_id, inning_num, raw))
        return rows

    # ------------------------------------------------------------------ #
    #  이닝 단위 파싱                                                       #
    # ------------------------------------------------------------------ #

    def _parse_inning(
        self, game_id: str, inning_num: int, raw: dict
    ) -> list[dict]:
        rows: list[dict] = []

        result_node = raw.get("result", {}) or {}
        relay_data = result_node.get("textRelayData", {}) or {}
        text_relays: list[dict] = relay_data.get("textRelays", []) or []

        for relay in text_relays:
            # ★ 수정: relay 하나(=타석 하나)에서 투구 여러 개가 나올 수 있음
            relay_rows = self._parse_relay(game_id, inning_num, relay)
            rows.extend(relay_rows)

        return rows

    # ------------------------------------------------------------------ #
    #  타석(relay) 단위 파싱 → 투구 이벤트별 Row 리스트 반환                    #
    # ------------------------------------------------------------------ #

    def _parse_relay(
        self,
        game_id: str,
        inning_num: int,
        relay: dict,
    ) -> list[dict]:
        """
        하나의 relay(=타석)에서 모든 투구 이벤트(type=1)를 추출하여 Row 리스트를 반환.

        실제 API 구조:
          relay
          ├─ homeOrAway          ← 공격 팀
          ├─ metricOption        ← wpaByPlate
          └─ textOptions[]
               ├─ [0]  type=8   ← 타자 등장 (batterRecord, currentPlayersInfo 포함)
               ├─ [1…] type=1   ← ★ 투구 이벤트 (speed, stuff, pitchResult, currentGameState)
               └─ [N]  type=13  ← 타석 결과
        """
        text_options: list = relay.get("textOptions") or []
        if not text_options:
            return []

        # ── relay-level 공통 정보 ─────────────────────────────────────
        home_or_away = str(relay.get("homeOrAway", "0"))
        metric = relay.get("metricOption", {}) or {}
        wpa_by_plate = metric.get("wpaByPlate")
        inn = relay.get("inn", inning_num)

        # ── 타자 등장 텍스트(type=8)에서 batterRecord 추출 ────────────
        # textOptions[0] 이 보통 type=8 이지만, 안전하게 탐색
        batter_record: dict = {}
        for opt in text_options:
            if opt.get("type") == 8:
                batter_record = opt.get("batterRecord", {}) or {}
                break

        # ── 타석 결과 텍스트(type=13/23)에서 relay_text 추출 ─────────
        relay_text: str = ""
        for opt in text_options:
            if opt.get("type") in (13, 23):
                relay_text = opt.get("text", "") or ""
                break

        # ── 투구 이벤트(type=1) 순회 ──────────────────────────────────
        rows: list[dict] = []
        for opt in text_options:
            if opt.get("type") != 1:
                continue
            if "pitchResult" not in opt:
                continue

            row = self._parse_pitch_option(
                game_id=game_id,
                inning_num=int(inn),
                home_or_away=home_or_away,
                wpa_by_plate=wpa_by_plate,
                batter_record=batter_record,
                opt=opt,
            )
            if row is not None:
                row["relay_text"] = relay_text
                rows.append(row)

        return rows

    # ------------------------------------------------------------------ #
    #  개별 투구 이벤트(textOption) 파싱                                      #
    # ------------------------------------------------------------------ #

    def _parse_pitch_option(
        self,
        game_id: str,
        inning_num: int,
        home_or_away: str,
        wpa_by_plate,
        batter_record: dict,
        opt: dict,
    ) -> Optional[dict]:
        """type=1 인 단일 textOption 을 받아 하나의 투구 Row dict 를 반환."""

        state: dict = opt.get("currentGameState", {}) or {}
        players_info: dict = opt.get("currentPlayersInfo", {}) or {}

        # ── 투수 팀 판별 ──────────────────────────────────────────────
        # homeOrAway == "1" → 홈팀 공격 → 원정팀(away)이 투수
        # homeOrAway == "0" → 원정팀 공격 → 홈팀(home)이 투수
        pitcher_side = "away" if home_or_away == "1" else "home"

        pitcher_obj: dict = players_info.get(pitcher_side, {}) or {}
        pitcher_stats: dict = pitcher_obj.get("currentGamePlayerStats", {}) or {}

        # ── 기본 ID ───────────────────────────────────────────────────
        pitcher_id = str(state.get("pitcher", ""))
        batter_id = str(state.get("batter", ""))

        # ── 구속 / 구종은 opt 바로 아래 ──────────────────────────────
        pitch_speed: Optional[float] = None
        raw_speed = opt.get("speed")
        if raw_speed is not None and raw_speed != "":
            try:
                pitch_speed = float(raw_speed)
            except (ValueError, TypeError):
                pitch_speed = None

        pitch_type: str = opt.get("stuff", "") or ""

        # ── 누적 투구 수 (투수 프로필) ────────────────────────────────
        total_pitch_count = pitcher_stats.get("ballCount", 0)

        # ── 파생 1: 최근 5구 구속 평균 ───────────────────────────────
        recent_5_avg = self._calc_recent_speed_avg(pitcher_id, pitch_speed)

        # ── 파생 2: 이닝 내 투구 수 ──────────────────────────────────
        inning_pc = self._calc_inning_pitch_count(pitcher_id, inning_num)

        # ── 상황 변수 ─────────────────────────────────────────────────
        home_score = _safe_float(state.get("homeScore"), 0)
        away_score = _safe_float(state.get("awayScore"), 0)
        score_diff = int(home_score - away_score)

        # ── 타자 평균 & 타석 유형 ─────────────────────────────────────
        batter_recent_avg = _safe_float(
            batter_record.get("todayHra") or batter_record.get("seasonHra")
        )
        batter_hit_type = _parse_hit_type(batter_record.get("hitType", ""))

        return {
            # 메타
            "game_id": game_id,
            "inning": inning_num,
            "home_or_away": home_or_away,
            # ── [1] 상황 변수 ────────────────────────────────────────
            "score_diff": score_diff,
            "out_count": state.get("out", 0),
            "ball_count_B": state.get("ball", 0),
            "ball_count_S": state.get("strike", 0),
            "is_base1": _base_flag(state.get("base1")),
            "is_base2": _base_flag(state.get("base2")),
            "is_base3": _base_flag(state.get("base3")),
            # ── [2] 투수 & 타자 프로필 ────────────────────────────────
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "batter_hit_type": batter_hit_type,
            "pitcher_vs_batter_avg": _safe_float(batter_record.get("vsHra")),
            "batter_recent_avg": batter_recent_avg,
            # ── [3] 투구 & 피로도 ─────────────────────────────────────
            "pitch_speed": pitch_speed,
            "pitch_type": pitch_type,
            "total_pitch_count": total_pitch_count,
            "recent_5_pitch_speed_avg": (
                round(recent_5_avg, 2) if recent_5_avg is not None else None
            ),
            "inning_pitch_count": inning_pc,
            # ── [4] 타겟 & 보상 ────────────────────────────────────────
            "pitch_result": opt.get("pitchResult", ""),
            "reward_wpa": wpa_by_plate,
        }

    # ------------------------------------------------------------------ #
    #  파생 변수 연산 (상태 캐싱)                                            #
    # ------------------------------------------------------------------ #

    def _calc_recent_speed_avg(
        self, pitcher_id: str, speed: Optional[float]
    ) -> Optional[float]:
        """
        현재 구를 캐시에 추가한 뒤 최근 최대 5구 평균을 반환.
        speed=None 이면 캐시에 추가하지 않고 즉시 None 반환.
        """
        if speed is None:
            return None

        self._speed_cache[pitcher_id].append(speed)
        cache = self._speed_cache[pitcher_id]
        return sum(cache) / len(cache)

    def _calc_inning_pitch_count(self, pitcher_id: str, inning: int) -> int:
        """
        이닝이 바뀌면 카운터를 0으로 리셋하고 1을 반환.
        동일 이닝이면 카운터를 1 증가시키고 반환.
        """
        if self._pitcher_inning.get(pitcher_id) != inning:
            self._pitcher_inning[pitcher_id] = inning
            self._inning_pitch_count[pitcher_id] = 0

        self._inning_pitch_count[pitcher_id] += 1
        return self._inning_pitch_count[pitcher_id]