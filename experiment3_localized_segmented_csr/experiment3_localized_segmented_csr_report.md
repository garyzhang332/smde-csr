# Experiment 3: Localized segmented CSR entry set and full-station summary

Prepared: 2026-06-05

## Figure contract

Core conclusion: localized segmented CSR should be constructed from rainfall-validated, non-interrupted, stage-II-like drydown events, then fitted separately by location, layer, and segment.

Evidence chain: the entry table defines the eligible event and segment pool; the representative-site figure shows how one location-layer curve is built; the full-station summary figure and table show where the method is supported across the FAWN network.

Archetype: quantitative grid with one representative local example plus network-level coverage and fit-quality panels.

Export contract: Python/matplotlib only; SVG/PDF with editable text, PNG preview, and 600 dpi TIFF are exported with source-data CSVs.

## CSR entry decision

Main localized segmented CSR uses `clean_stageII_48h`: rainfall-associated within 48 h, not interrupted by rain, and classified as stage-II-like by the loss-function diagnosis.

The early 0-3 h portion is kept as a separate transient segment. The post-3 h mid-storage segment is the primary storage-dependent CSR segment, and the post-3 h low-storage tail is retained with cautious interpretation.

The local fitting threshold is at least 8 events and 40 aligned observations for each location-layer-segment.

## Representative location-layer example

Selected example: FAWN site 275, 4 in layer.

The selected site is a complete, non-405 representative candidate: all three segments are fitted, event support is not extreme, and the median RMSE is close to the network median for complete 4 in candidates.

Selection score rank: 1; total segment-events: 116; minimum segment-events: 36; median RMSE: 0.628 mm.

| Segment | Events | Points | CCC | RMSE mm | MAE mm | Bias mm |
|---|---:|---:|---:|---:|---:|---:|
| Early transient, 0-3 h | 40 | 480 | 0.981 | 0.502 | 0.320 | 0.109 |
| Post-3 h mid-storage | 36 | 704 | 0.921 | 0.924 | 0.496 | 0.092 |
| Post-3 h low-storage tail | 40 | 645 | 0.959 | 0.628 | 0.251 | 0.054 |

## Full-station summary

The main event pool contains 1,873 clean stage-II-like events across 41 FAWN sites before segment-level support filtering.

Complete three-segment local CSR coverage is strongest at 4 in (28 location-layers) and secondary at 8 in (8 location-layers). Deeper layers are retained in the table but should be described as exploratory because only a few location-layers meet the fitting threshold.

| Layer | Segment | Use | Local models | Median events/model | Total points | Median RMSE mm | Median MAE mm | Median bias mm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 in | Early transient, 0-3 h | main-text robust | 30 | 44.0 | 17,127 | 0.689 | 0.433 | 0.089 |
| 4 in | Post-3 h mid-storage | main-text robust | 28 | 36.5 | 17,823 | 0.648 | 0.438 | 0.004 |
| 4 in | Post-3 h low-storage tail | main-text robust | 30 | 44.0 | 24,488 | 0.450 | 0.251 | 0.015 |
| 8 in | Early transient, 0-3 h | secondary-supported | 10 | 16.0 | 1,836 | 0.626 | 0.428 | 0.059 |
| 8 in | Post-3 h mid-storage | secondary-supported | 8 | 13.5 | 2,410 | 0.629 | 0.383 | -0.027 |
| 8 in | Post-3 h low-storage tail | secondary-supported | 10 | 16.0 | 3,652 | 0.357 | 0.254 | -0.024 |
| 12 in | Early transient, 0-3 h | exploratory/table only | 3 | 22.0 | 672 | 0.638 | 0.493 | 0.056 |
| 12 in | Post-3 h mid-storage | exploratory/table only | 2 | 18.5 | 546 | 0.582 | 0.362 | 0.025 |
| 12 in | Post-3 h low-storage tail | exploratory/table only | 3 | 22.0 | 982 | 0.800 | 0.477 | 0.011 |
| 16 in | Early transient, 0-3 h | exploratory/table only | 2 | 18.5 | 444 | 0.901 | 0.702 | -0.008 |
| 16 in | Post-3 h mid-storage | exploratory/table only | 2 | 9.5 | 175 | 0.984 | 0.767 | 0.074 |
| 16 in | Post-3 h low-storage tail | exploratory/table only | 2 | 18.5 | 564 | 0.683 | 0.430 | 0.081 |
| 20 in | Early transient, 0-3 h | exploratory/table only | 1 | 31.0 | 372 | 0.864 | 0.622 | 0.036 |
| 20 in | Post-3 h low-storage tail | exploratory/table only | 1 | 31.0 | 337 | 0.119 | 0.099 | -0.004 |

## Recommended manuscript use

- Main example figure: use the representative site-layer figure to show the localized segmented CSR construction.
- Main network summary: report 4 in as the strongest full-network layer; report 8 in as secondary-supported.
- Table or Supplementary Table: include all fitted layer-segment combinations, including sparse 12-20 in results, but avoid strong mechanistic claims from sparse deeper layers.
- Sensitivity statement: all-events and clean-only variants can be cited as robustness checks, but they should not define the main CSR curve.

## Output files

- `fig_experiment3_representative_location_segmented_csr.svg/pdf/png/tiff`
- `fig_experiment3_full_station_segmented_csr_summary.svg/pdf/png/tiff`
- `source_data/experiment3_csr_entry_decisions.csv`
- `source_data/experiment3_main_by_layer_segment_summary.csv`
- `source_data/experiment3_location_layer_csr_availability.csv`
- `source_data/experiment3_location_layer_segment_metrics.csv`
- `source_data/experiment3_representative_site_candidates.csv`
- `source_data/experiment3_representative_location_*.csv`

## 80/20 split validation accuracy

Held-out accuracy was evaluated with an event-level 80/20 split within each eligible location-layer-segment. Training events build the local state-stitched CSR curve; held-out events are predicted from the training curve.

| Layer | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 in | 29 | 688 | 11,979 | 0.992 | 0.984 | 0.850 | 0.463 | -0.046 |
| 8 in | 8 | 65 | 1,724 | 0.992 | 0.985 | 0.887 | 0.563 | -0.053 |
| 12 in | 2 | 21 | 434 | 0.972 | 0.945 | 0.615 | 0.480 | -0.059 |
| 16 in | 2 | 16 | 236 | 0.958 | 0.927 | 1.717 | 0.914 | 0.096 |
| 20 in | 1 | 11 | 150 | 0.911 | 0.849 | 0.543 | 0.366 | 0.140 |

This split-validation result supports using 4 in as the primary localized segmented CSR layer and 8 in as secondary support; deeper layers remain exploratory because few local validation models meet the event-level split threshold.
