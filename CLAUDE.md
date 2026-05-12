# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Naver API structure (critical for parser changes)

```
result.textRelayData.textRelays[]     ← one element = one plate appearance (PA)
  relay.homeOrAway                    ← "1"=home batting, "0"=away batting
  relay.metricOption.wpaByPlate       ← WPA reward signal for the PA
  relay.textOptions[]
    type=8  → batter appearance; contains batterRecord (hitType, vsHra, todayHra/seasonHra)
    type=1  → pitch event (speed, stuff/pitch type, pitchResult, currentGameState, currentPlayersInfo)
    type=13 or 23 → PA result text (relay_text)
```

`currentPlayersInfo` inside a type=1 option has `home` and `away` keys, each containing `currentGamePlayerStats.ballCount` (pitcher's cumulative pitch count).

### Game ID format

`YYYYMMDD{HomeTeamCode}{AwayTeamCode}{0|1|2}{SeasonYear}`
- Middle digit: `0`=single game, `1`=DH game 1, `2`=DH game 2
- Team codes: LG, KT, NC, SS(Samsung), HH(Hanwha), HT(KIA), OB(Doosan), SK(SSG), LT(Lotte), WO(Kiwoom)
- `hsk_game_ids_2016_2024.txt` — 153 HH vs SK/SSG matchups used by `run_hsk.py`

### Output schemas

**Pitch-level CSV** (one row per pitch, written to `data/pbp/` or `data_analysis/results/pbp/`):
- State: `game_id`, `inning`, `home_or_away`, `score_diff`, `out_count`, `ball_count_B/S`, `is_base1/2/3`
- Profile: `pitcher_id`, `batter_id`, `batter_hit_type` (L/R/S), `pitcher_vs_batter_avg`, `batter_recent_avg`
- Pitch/fatigue: `pitch_speed`, `pitch_type`, `total_pitch_count`, `recent_5_pitch_speed_avg` (rolling 5-pitch deque per pitcher), `inning_pitch_count` (resets each inning per pitcher)
- Target: `pitch_result`, `relay_text`, `reward_wpa`

**PA-level Parquet** (`data_analysis/results/hsk_pa.parquet`): all pitch-level state columns (first-pitch values) plus `pa_result`, `reward_wpa`, `pitches_per_pa`, `pa_avg_pitch_speed`.
