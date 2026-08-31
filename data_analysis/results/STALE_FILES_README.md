# 구버전(STALE) 파일 안내

다음 파일들은 **이번 라운드(전 구단 10년 확장 + 터미널 PA 보정 + 연장전 정책 + UNK 정규식 수정)
반영 이전**에 생성된 153경기 전용 산출물이다. 최종본으로 오인하지 말 것.

- `hsk_pa_enriched_STALE.parquet` (구 `hsk_pa_enriched.parquet`, 2024-08-21 생성)
- `enrichment_report_STALE.md` (구 `enrichment_report.md`)

**미반영 사항**: UNK 정규식 보강(HBP/IBB/내야안타/번트안타/ROE), 터미널 PA
ground-truth 보정(끝내기 득점 누락 수정), 연장전 정책(9회말 동점=0.5,
10회 이후 제외), `pa_result_raw`/`is_extra_innings` 컬럼.

**현재 최종본**: `all_pa_enriched.parquet` (10개년 2016-2025, 7,197경기,
558,064 PA, 68컬럼). 상세는 `all_enrichment_report.md` 참고.

153경기 코퍼스만 필요하면 `hsk_pa_with_wpa_corrected.parquet`(이번 라운드
보정 반영, 원본 `hsk_pa_with_wpa.parquet`은 read-only 보존)을 사용할 것.
