# Segmented CSR construction results

Prepared: 2026-06-05

This analysis replaces a single pooled CSR curve with localized segmented CSR.
Each location-layer-segment is fitted separately. Full-station results are summarized
after all local segmented CSR models are built. LOWESS is used only as a within-segment
empirical smoother, not as a pooled curve across locations, layers, or regimes.

## Segment definitions

| Segment | Coordinate | Interpretation |
|---|---|---|
| early_0_3h | local moisture-state stitching | rapid early redistribution, drainage, or sensor/layer adjustment after wetting |
| post3_mid_storage | moisture-state stitching | main post-transient storage-dependent drydown segment |
| post3_late_storage | moisture-state stitching | low-storage tail where drying may slow or become noise-limited |

The manuscript-safe main subset is `clean_stageII_48h`: rainfall-associated within 48 h,
not interrupted by rain during the event, and classified as stage-II-like by the diagnostic proxy.

## Full-station summary of local segmented CSR models

| Layer | Segment | Local models | Median events/model | Total points | Median RMSE mm | Median MAE mm | Median bias mm |
|---|---|---:|---:|---:|---:|---:|---:|
| moisture_4in | early_0_3h | 30 | 44.0 | 17,127 | 0.689 | 0.433 | 0.089 |
| moisture_4in | post3_late_storage | 30 | 44.0 | 24,488 | 0.450 | 0.251 | 0.015 |
| moisture_4in | post3_mid_storage | 28 | 36.5 | 17,823 | 0.648 | 0.438 | 0.004 |
| moisture_8in | early_0_3h | 10 | 16.0 | 1,836 | 0.626 | 0.428 | 0.059 |
| moisture_8in | post3_late_storage | 10 | 16.0 | 3,652 | 0.357 | 0.254 | -0.024 |
| moisture_8in | post3_mid_storage | 8 | 13.5 | 2,410 | 0.629 | 0.383 | -0.027 |

## Pooled diagnostic metrics from local predictions

| Layer | Events | Sites | Points | CCC | RMSE mm | MAE mm | Bias mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| moisture_4in | 1,434 | 30 | 59,438 | 0.988 | 1.031 | 0.498 | 0.041 |
| moisture_8in | 153 | 10 | 7,898 | 0.996 | 0.614 | 0.381 | -0.043 |

## Segment-level pooled diagnostics

| Layer | Segment | Events | Sites | Points | RMSE mm | MAE mm | Dynamic MAE mm/step |
|---|---|---:|---:|---:|---:|---:|---:|
| moisture_4in | early_0_3h | 1,430 | 30 | 17,127 | 1.391 | 0.655 | 0.068 |
| moisture_4in | post3_late_storage | 1,432 | 30 | 24,488 | 0.786 | 0.379 | 0.025 |
| moisture_4in | post3_mid_storage | 1,194 | 28 | 17,823 | 0.915 | 0.512 | 0.047 |
| moisture_8in | early_0_3h | 153 | 10 | 1,836 | 0.661 | 0.438 | 0.093 |
| moisture_8in | post3_late_storage | 153 | 10 | 3,652 | 0.507 | 0.298 | 0.033 |
| moisture_8in | post3_mid_storage | 114 | 8 | 2,410 | 0.717 | 0.465 | 0.052 |

## Sensitivity against main segmented CSR

| Layer | Segment | Subset | Local model pairs | Median curve RMSE to main mm | Median curve bias to main mm |
|---|---|---|---:|---:|---:|
| moisture_4in | early_0_3h | clean_48h | 30 | 1.967 | 0.452 |
| moisture_4in | early_0_3h | clean_stageII_48h_no405 | 29 | 0.000 | 0.000 |
| moisture_4in | early_0_3h | clean_stageII_48h_site_balanced | 30 | 1.722 | 0.000 |
| moisture_4in | post3_late_storage | clean_48h | 30 | 1.044 | 0.686 |
| moisture_4in | post3_late_storage | clean_stageII_48h_no405 | 29 | 0.000 | 0.000 |
| moisture_4in | post3_late_storage | clean_stageII_48h_site_balanced | 30 | 1.715 | -1.056 |
| moisture_4in | post3_mid_storage | clean_48h | 28 | 2.046 | 1.468 |
| moisture_4in | post3_mid_storage | clean_stageII_48h_no405 | 27 | 0.000 | 0.000 |
| moisture_4in | post3_mid_storage | clean_stageII_48h_site_balanced | 28 | 1.365 | -1.020 |
| moisture_8in | early_0_3h | clean_48h | 10 | 3.326 | 2.703 |
| moisture_8in | early_0_3h | clean_stageII_48h_no405 | 9 | 0.000 | 0.000 |
| moisture_8in | early_0_3h | clean_stageII_48h_site_balanced | 10 | 0.000 | 0.000 |
| moisture_8in | post3_late_storage | clean_48h | 10 | 0.990 | 0.867 |
| moisture_8in | post3_late_storage | clean_stageII_48h_no405 | 9 | 0.000 | 0.000 |
| moisture_8in | post3_late_storage | clean_stageII_48h_site_balanced | 10 | 0.000 | 0.000 |
| moisture_8in | post3_mid_storage | clean_48h | 8 | 1.959 | 1.568 |
| moisture_8in | post3_mid_storage | clean_stageII_48h_no405 | 7 | 0.000 | 0.000 |
| moisture_8in | post3_mid_storage | clean_stageII_48h_site_balanced | 8 | 0.000 | 0.000 |

## Manuscript interpretation

The localized segmented CSR framing avoids two averaging problems: mixing regimes into one
curve and mixing stations with different soil-water ranges into one pooled curve. Each local
location-layer model has its own early transient, post-3h mid-storage, and post-3h low-storage
segments. Full-station tables summarize how many local models can be fitted and how stable each
segment is across the network.

## Output files

- `segmented_csr_curves.csv`
- `segmented_csr_aligned_points.parquet`
- `segmented_csr_binned_points.csv`
- `segmented_csr_predictions.parquet`
- `segmented_csr_layer_metrics.csv`
- `segmented_csr_segment_metrics.csv`
- `segmented_csr_local_segment_metrics.csv`
- `segmented_csr_local_summary_by_layer_segment.csv`
- `segmented_csr_curve_distance_to_main.csv`
- `segmented_csr_curve_distance_summary.csv`
- `fig_segmented_csr_main_primary_layers.png`
- `fig_segmented_csr_sensitivity_primary_layers.png`
- `fig_segmented_csr_main_segment_metrics.png`
- `fig_segmented_csr_curve_distance_to_main.png`
- `fig_segmented_csr_location_layer_rmse_heatmaps.png`