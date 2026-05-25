# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pipeline Status

Phase 1–4 complete. `data_analysis/results/hsk_pa_with_wpa.parquet` (11,984 PA rows) is the primary training artifact. Next: CatBoost / Monte Carlo / DQN (team handoff).

## Commands

```bash
# Install dependencies (uses uv.lock for reproducible env)
uv sync

# Run the crawler (edit GAME_IDS in main.py first)
uv run python main.py

# Run the full HSK pipeline: crawl 153 games + aggregate to PA-level Parquet
uv run python data_analysis/methods/run_hsk.py

# Regenerate the HSK game ID list (writes hsk_game_ids_2015_2024.txt)
uv run python get_hsk_game_ids.py

# Regenerate WPA data (run in order if rebuilding from scratch)
uv run python -m data_analysis.methods.state_transition  # adds we_before/we_after
uv run python -m data_analysis.methods.inject_wpa        # adds reward_wpa_computed
uv run python -m data_analysis.methods.validate_wpa      # Phase 4 validation report

# Background run with log capture
nohup uv run python data_analysis/methods/run_hsk.py > hsk.log 2>&1 &
```

There are no tests or linting configured in this project.

## Architecture

KBO (Korean Baseball Organization) pitch-by-pitch data crawler that fetches from Naver Sports' text relay API and outputs CSVs for CatBoost/DQN model training. The pipeline has two layers: pitch-level crawling and plate appearance (PA) aggregation.

### Pitch-level crawling (`main.py` / `kbo_crawler/`)

```
main.py → KBODataPipeline.run()
             → NaverSportsAPIFetcher.fetch_game()   # fetcher.py
                  → fetch_inning() × up to 12 innings (random 0.5–1.5s delay)
                  → returns [{'inning': int, 'data': raw_json}, ...]
             → PitchDataParser.parse_game()          # parser.py
                  → _parse_inning() → _parse_relay() → _parse_pitch_option()
                  → maintains stateful caches for derived features
             → pd.DataFrame → {output_dir}/{game_id}.csv
```

**Key design decisions:**
- `KBODataPipeline` shares a single `aiohttp.ClientSession` across all games (connection pool reuse)
- Resume support: games with existing CSVs are skipped automatically
- `PitchDataParser` is **stateful** — call `reset()` between games when reusing the same instance. In `run_hsk.py`, a fresh instance is created per game instead (state isolation without explicit reset)
- Column order is enforced by `ORDERED_COLS` in `pipeline.py`; unexpected columns are appended at the end

### PA aggregation layer (`data_analysis/`)

```
run_hsk.py
  → crawl_all()                          # asyncio.Semaphore(5) parallel crawl
       → process_one() per game          # pitch-level CSV → data_analysis/results/pbp/
  → aggregate_all_csvs()
       → concat all CSVs → aggregate_pa()   # pa_aggregator.py
  → data_analysis/results/hsk_pa.parquet
```

`pa_aggregator.aggregate_pa(df)` groups by a **sequential PA index** (`_pa_seq`, derived via `cumsum` of row-group changes) rather than raw key columns directly, so the same pitcher–batter matchup appearing twice in one inning is treated as two separate plate appearances. Aggregation extracts:
- First pitch → state columns (`score_diff`, `out_count`, base flags, etc.)
- Last pitch → target columns (`pa_result`, `reward_wpa`)
- Count/mean → `pitches_per_pa`, `pa_avg_pitch_speed`

`pa_result` is classified by regex over `relay_text` (type=13 textOption): HR, 3B, 2B, 1B, IBB, BB, HBP, SO, GDP, SF, OUT, or UNK. UNK rate >5% triggers a warning log.

### WPA computation layer (`data_analysis/methods/`)

```
state_transition.py   → adds base_state, half, score_diff_attacker, we_before,
                         runs_scored, inning_ended to hsk_pa.parquet
inject_wpa.py         → computes we_after, reward_wpa_computed (ΔWE), writes
                         hsk_pa_with_wpa.parquet
validate_wpa.py       → Phase 4 validation: sign-match vs Naver WPA, Spearman ρ
we_re_lookup.py       → RE/WE lookup for Monte Carlo / DQN (import this module)
```

**WE table source**: 문형우 외(2016) KBO Markov-chain paper (Table 3.2).
- Innings 3 and 7 are fully tabulated; other innings are linearly interpolated.
- Out dimension: Option A — all out counts use the 1-out row (known limitation; sign-match 78.5%, Spearman ρ = 0.676).
- Score diff clipped to [−4, +4]; walk-off → WE = 1.0 immediately.

`get_we_with_boundary(inning, half, score_diff, out_count, base_state)` is the primary entry point for downstream simulation.

### Data Chronology Policy (CRITICAL)

**CRITICAL: 네이버 API 응답은 역순(최신 이벤트가 상단)이므로, `parser.py`의 `_parse_inning()`에서 각 경기별/이닝별 데이터를 파싱할 때 반드시 실제 경기 시간순으로 정렬을 뒤집어야 한다.**

- 정렬 기준: `['game_id', 'inning', 'home_or_away']` 오름차순 정렬을 보장하되, `home_or_away`는 오름차순(초=0이 말=1보다 앞)으로 처리하여 초→말 실제 이닝 순서를 보존한다.
- 단일 `(inning, home_or_away)` 그룹 내의 타석(PA) 배열은 API가 역순으로 반환하므로, `_parse_inning()`에서 `homeOrAway`별로 relay 목록을 그룹화한 뒤 각 그룹을 `reversed()`로 순회하여 최초 타석이 먼저 오도록 처리한다.
- 결과: CSV 2행 = 1회초 1번 타자 첫 투구, 마지막 행 = 9회말(또는 연장 최후 이닝) 마지막 타자 마지막 투구.
- `state_transition.py`의 역순 보정 워크어라운드는 소스 수준 수정 이후 불필요하므로 제거되었다.

### Naver API structure (critical for parser changes)

```
result.textRelayData.textRelays[]     ← one element = one plate appearance (PA)
  relay.homeOrAway                    ← "0"=away batting (초/top), "1"=home batting (말/bot)
  relay.metricOption.wpaByPlate       ← WPA reward signal for the PA
  relay.textOptions[]
    type=8  → batter appearance; contains batterRecord (hitType, vsHra, todayHra/seasonHra)
    type=1  → pitch event (speed, stuff/pitch type, pitchResult, currentGameState, currentPlayersInfo)
    type=13 or 23 → PA result text (relay_text)
```

`currentPlayersInfo` inside a type=1 option has `home` and `away` keys, each containing `currentGamePlayerStats.ballCount` (pitcher's cumulative pitch count).

### Game ID format

`YYYYMMDD{AwayTeamCode}{HomeTeamCode}{0|1|2}{SeasonYear}`
- Middle digit: `0`=single game, `1`=DH game 1, `2`=DH game 2
- Team codes: LG, KT, NC, SS(Samsung), HH(Hanwha), HT(KIA), OB(Doosan), SK(SSG), LT(Lotte), WO(Kiwoom)
- Example: `20160317SKHH02016` → Away=SK, Home=HH, single game, 2016 season
- `hsk_game_ids_2016_2024.txt` — 153 HH vs SK/SSG matchups used by `run_hsk.py`

## Data Conventions (DO NOT MODIFY — verified against ground truth)

- `home_or_away`: 0=away batting (top/초), 1=home batting (bot/말)
- `score_diff` = home_score - away_score (H1 convention)
- Score updates with **1-PA delay**: a run scored in PA N appears in
  PA N+1's score_diff. This is by design; do not "fix" it.
- Naver API returns half-innings in reverse chronological order (bot
  before top within same inning). parser.py compensates this.
- `score_diff_attacker` (in PA-level outputs) is already converted to
  attacker's perspective; use it directly without further sign flipping.
- `reward_wpa_computed` = we_after − we_before, attacker's perspective.
  For the pitcher (defending team) perspective use `-reward_wpa_computed`.
- `reward_wpa` (Naver original): 89.46% missing, non-standard scale
  (min=−21, max=+59). **Do not use as a training target.** Keep for
  validation reference only.
- `data_quality_flag`: `""` = clean (11,937 rows); `"inning1_nonzero_start"` = 47 rows
  with non-zero score_diff at 1st inning start (suspected crawl gap). Exclude with
  `df[df["data_quality_flag"] == ""]` before training.

These were established by comparing 20160317 SK vs HH game data against
external ground truth (starting lineup + box score). DO NOT re-derive
these from data internals — that path leads to errors due to
multiple consistent sign flips in the dataset.

### Output schemas

**Pitch-level CSV** (one row per pitch, written to `data/pbp/` or `data_analysis/results/pbp/`):
- State: `game_id`, `inning`, `home_or_away`, `score_diff`, `out_count`, `ball_count_B/S`, `is_base1/2/3`
- Profile: `pitcher_id`, `batter_id`, `batter_hit_type` (L/R/S), `pitcher_vs_batter_avg`, `batter_recent_avg`
- Pitch/fatigue: `pitch_speed`, `pitch_type`, `total_pitch_count`, `recent_5_pitch_speed_avg` (rolling 5-pitch deque per pitcher), `inning_pitch_count` (resets each inning per pitcher)
- Target: `pitch_result`, `relay_text`, `reward_wpa`

**PA-level Parquet** (`data_analysis/results/hsk_pa.parquet`): all pitch-level state columns (first-pitch values) plus `pa_result`, `reward_wpa`, `pitches_per_pa`, `pa_avg_pitch_speed`.

**WPA-enriched PA Parquet** (`data_analysis/results/hsk_pa_with_wpa.parquet`, **primary training artifact**, 11,984 rows): all hsk_pa columns plus:
- `base_state` (str): "0"/"1"/"2"/"3"/"12"/"13"/"23"/"123"
- `half` (str): "top"/"bot"
- `score_diff_attacker` (int): score diff from attacker's perspective (sign-converted, use directly)
- `we_before` / `we_after` (float, [0,1]): attacker win probability before/after PA
- `runs_scored` (int): runs scored in this PA
- `inning_ended` (bool): whether this PA ended the inning
- `reward_wpa_computed` (float, [−1,+1]): ΔWE = we_after − we_before (primary reward signal)
- `data_quality_flag` (str): "" or "inning1_nonzero_start"

**Recommended training setup**:
```python
cat_features = ["half", "base_state", "batter_hit_type", "pitcher_id", "batter_id"]
num_features = ["inning", "score_diff_attacker", "out_count",
                "total_pitch_count", "inning_pitch_count",
                "pitcher_vs_batter_avg", "batter_recent_avg"]
# pitcher_vs_batter_avg, batter_recent_avg NaN → fill 0.250
# y options: pa_result (multiclass), reward_wpa_computed (regression)
```
