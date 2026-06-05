# Full-station SMDE validation report

Prepared: 2026-06-05

This note completes the validation handoff for the full-station SMDE Detection
Audit and Regime Composition Analysis. It checks output completeness, figure
readability, key numerical results, and station-level quality-control flags.

## Source and output files

Main script:

`../fawn_full_smde_audit.py`

Output directory:

`./`

Core outputs checked:

- `full_smde_event_audit.csv`
- `full_smde_loss_points.parquet`
- `full_smde_summary_by_layer.csv`
- `full_smde_summary_by_site_layer.csv`
- `full_smde_summary_by_year_layer.csv`
- `processing_status_by_site.csv`
- `fig_full_detection_audit.png`
- `fig_full_regime_composition.png`
- `fig_full_site_layer_heatmaps.png`
- `full_smde_analysis_manifest.json`

## Completion check

The manifest agrees with the actual output tables:

| Check | Manifest | Actual table |
|---|---:|---:|
| SMDE event rows | 7,147 | 7,147 |
| Loss-rate point rows | 201,018 | 201,018 |
| Sites with detected events | 41 | 41 |

No files listed in `full_smde_analysis_manifest.json` were missing from the
output directory.

## Detection audit: what was validated

The detection audit used soil moisture only to detect decreasing events. Rainfall
was added only after detection as an independent validation layer.

The audit classified detected events using rainfall before and during the event:

- Rain-associated clean <=24h
- Rain-associated clean 24-48h
- Rain-associated but interrupted
- Not associated within 48h

Across all depths, the full-station run detected 7,147 SMDE events. Of these,
5,520 events were associated with rainfall within 48 h, and 4,044 events were
both associated within 48 h and not interrupted by rain. The clean 48 h plus
stage-II-like subset contained 1,873 events.

## Layer-level results

| Layer | Events | Assoc. 48h | Interrupted | Clean 48h | Stage-II-like | Early-transient-heavy | Clean 48h + stage-II-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| moisture_4in | 4,276 | 79.6% | 18.6% | 61.4% | 56.3% | 31.2% | 34.4% |
| moisture_8in | 1,014 | 79.4% | 35.9% | 44.4% | 47.8% | 42.0% | 20.1% |
| moisture_12in | 624 | 82.7% | 27.7% | 55.9% | 32.5% | 59.5% | 15.1% |
| moisture_16in | 634 | 75.7% | 17.4% | 59.1% | 19.2% | 77.3% | 9.6% |
| moisture_20in | 599 | 52.6% | 12.2% | 40.7% | 12.4% | 84.8% | 7.2% |

Interpretation:

- `moisture_4in` is the strongest primary CSR construction layer.
- `moisture_8in` is useful as a secondary sensitivity layer.
- `moisture_12in`, `moisture_16in`, and `moisture_20in` should not be pooled
  into a main CSR curve without separate sensitivity analysis because
  early-transient-heavy events dominate.

## Regime-composition audit: what was validated

The regime audit estimated diagnostic proxies for each event:

- Full, trim1, trim3, and trim6 exponential-fit R2
- Trimmed exponential tau
- First-3h drop share
- Post-3h loss-storage correlation and slope
- Post-3h loss coefficient of variation
- Event-level atmospheric-demand context, including VPD

Proxy classes were assigned as:

- `stage-II-like`: trim3 R2 >= 0.7 and post-3h loss-storage correlation >= 0.2
- `stage-I-like`: post-3h loss approximately storage-invariant and loss CV < 0.6
- `early-transient-heavy`: first 3h drop share >= 0.4
- `mixed_or_uncertain`: events not matching the other proxy classes

The high median trim3 R2 values show that the detected events are regular after
trimming, but they do not prove physical stage-II behavior by themselves. The
loss-storage and early-drop diagnostics are needed to avoid overclaiming.

## Figure QA

### Figure: SMDE Detection Audit

File: `fig_full_detection_audit.png`

Status: readable.

Main signals:

- Event counts are dominated by `moisture_4in`.
- `site 405` is a strong outlier in the station-depth heatmap.
- Lag from previous rain is concentrated near event start, especially in the
  shallow layer.

Revision note for manuscript use: the lower-right legend occupies a large part
of the plotting area and should be moved or split in a final publication figure.

### Figure: Regime Composition Analysis

File: `fig_full_regime_composition.png`

Status: readable.

Main signals:

- Trimmed exponential fits have very high R2 across depths.
- Deeper layers have much higher early-transient-heavy shares.
- The VPD scatter panel is dense and useful as exploratory context, but should
  be simplified for a manuscript figure.

Revision note for manuscript use: consider faceting or filtering the VPD panel,
or showing only `moisture_4in` and `moisture_8in`.

### Figure: Site-layer Heatmaps

File: `fig_full_site_layer_heatmaps.png`

Status: readable.

Main signals:

- Rain association is generally high for many station-depth combinations.
- Rain interruption varies strongly by station and depth.
- Stage-II-like rates are strongest in selected shallow-layer combinations and
  weaker or absent for several deeper-layer combinations.

Revision note for manuscript use: blank cells should be explicitly labeled as
no detected events or insufficient data in the caption.

## Station QA

### Site 405

Site 405 is the largest event-count outlier:

- 1,832 events, or 25.6% of all detected events.
- Rank 1 of 44 stations by event count.
- Non-405 station median event count: 63.
- Non-405 station 95th percentile event count: 309.1.

Site 405 by layer:

| Layer | Events | Assoc. 48h | Interrupted | Stage-II-like | Early-transient-heavy | Median duration h | Median drop mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| moisture_4in | 555 | 77.7% | 7.6% | 11.4% | 83.4% | 2.00 | 1.03 |
| moisture_8in | 152 | 78.3% | 16.4% | 25.7% | 57.2% | 3.75 | 1.18 |
| moisture_12in | 293 | 75.8% | 7.8% | 15.7% | 74.7% | 2.75 | 0.89 |
| moisture_16in | 430 | 70.7% | 4.9% | 7.0% | 89.8% | 1.50 | 0.84 |
| moisture_20in | 402 | 41.3% | 1.7% | 1.5% | 97.3% | 1.00 | 0.73 |

Interpretation:

Site 405 has many short events, and most are early-transient-heavy rather than
stage-II-like. It should not dominate the main CSR curve without sensitivity
checks. For the main CSR construction, run at least one variant excluding site
405, or cap station contribution through site-balanced resampling.

### Sites with no detected events

Sites with zero detected events:

- 241
- 302
- 410

QC interpretation:

- Site 410 has soil rows but no non-null soil moisture values in the exported
  moisture fields. This is a missing-soil-data case.
- Site 241 has complete non-null values, but the values are around 0.1 rather
  than around 10. This suggests a possible fraction-vs-percent scale issue. The
  current absolute mm thresholds reject all candidate decreasing runs as too low
  and too small.
- Site 302 has complete data and plausible variation, but the current event
  filters reject all candidate runs. Most rejected runs fail low drop-rate and
  total-decrease thresholds; many shallow runs also include values below the
  current low-value thresholds.

Recommended handling:

- Treat site 410 as unavailable for soil-moisture-event analysis unless a
  different source column exists.
- Inspect site 241 units before excluding it permanently. If values are
  fractional volumetric water content, convert consistently before detection.
- Inspect site 302 with relaxed diagnostic-only thresholds before deciding
  whether it is genuinely unsuitable or simply too low-amplitude under current
  settings.

## Manuscript-safe conclusions

The validation supports the conservative manuscript framing:

1. Detect SMDE events from soil moisture alone.
2. Use precipitation only after detection as an independent validation layer.
3. Separate clean rainfall-associated events from rain-interrupted and
   non-associated events.
4. Diagnose drydown regime composition using loss rate, exponential-fit
   sensitivity, early-drop share, and atmospheric-demand context.
5. Build the main CSR curve from clean 48 h plus stage-II-like events, focused
   on `moisture_4in`, with all-events, clean-only, trim-3h, `moisture_8in`, and
   site-405-excluded variants as sensitivity analyses.

Do not claim every detected SMDE event is a pure stage-II drydown. The defensible
claim is that a clean, empirically stage-II-like subset can support low-input CSR
construction.

## Remaining work before manuscript figures

- Rebuild manuscript-ready versions of the three figures with cleaner legends,
  explicit no-data labeling, and fewer overplotted points.
- Run CSR sensitivity construction:
  - all events
  - clean 48 h events
  - clean 48 h plus stage-II-like events
  - trim-3h events
  - excluding site 405
  - site-balanced contribution
- Perform a threshold-sensitivity check for sites 241 and 302 before final
  station exclusion decisions.
