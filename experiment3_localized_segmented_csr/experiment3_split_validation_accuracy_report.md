# Experiment 3 addendum: 80/20 split validation accuracy

Prepared: 2026-06-05

This addendum estimates held-out accuracy for localized segmented CSR using an event-level 80/20 split within each eligible location-layer-segment. Training events build the local state-stitched CSR curve; held-out events are aligned to the training anchor and predicted without contributing to the smoother.

## Pooled held-out accuracy by layer

| Layer | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm | nRMSE % range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 in | 29 | 688 | 11,979 | 0.992 | 0.984 | 0.850 | 0.463 | -0.046 | 2.8 |
| 8 in | 8 | 65 | 1,724 | 0.992 | 0.985 | 0.887 | 0.563 | -0.053 | 3.3 |
| 12 in | 2 | 21 | 434 | 0.972 | 0.945 | 0.615 | 0.480 | -0.059 | 4.8 |
| 16 in | 2 | 16 | 236 | 0.958 | 0.927 | 1.717 | 0.914 | 0.096 | 6.3 |
| 20 in | 1 | 11 | 150 | 0.911 | 0.849 | 0.543 | 0.366 | 0.140 | 9.0 |

## Pooled held-out accuracy by layer and segment

| Layer | Segment | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 in | Early transient, 0-3 h | 29 | 296 | 3,551 | 0.993 | 0.986 | 0.802 | 0.516 | 0.078 |
| 4 in | Post-3 h mid-storage | 27 | 248 | 3,507 | 0.994 | 0.987 | 0.724 | 0.406 | -0.008 |
| 4 in | Post-3 h low-storage tail | 29 | 297 | 4,921 | 0.990 | 0.979 | 0.959 | 0.464 | -0.163 |
| 8 in | Early transient, 0-3 h | 8 | 30 | 360 | 0.992 | 0.985 | 0.916 | 0.558 | -0.012 |
| 8 in | Post-3 h mid-storage | 8 | 26 | 546 | 0.985 | 0.968 | 1.120 | 0.747 | -0.038 |
| 8 in | Post-3 h low-storage tail | 8 | 30 | 818 | 0.996 | 0.991 | 0.671 | 0.443 | -0.082 |
| 12 in | Early transient, 0-3 h | 2 | 11 | 132 | 0.968 | 0.937 | 0.631 | 0.505 | 0.152 |
| 12 in | Post-3 h mid-storage | 2 | 9 | 129 | 0.931 | 0.882 | 0.452 | 0.367 | 0.014 |
| 12 in | Post-3 h low-storage tail | 2 | 10 | 173 | 0.969 | 0.942 | 0.702 | 0.544 | -0.274 |
| 16 in | Early transient, 0-3 h | 2 | 8 | 96 | 0.929 | 0.890 | 2.587 | 1.700 | 0.544 |
| 16 in | Post-3 h mid-storage | 1 | 2 | 26 | 0.689 | 0.607 | 0.620 | 0.536 | -0.210 |
| 16 in | Post-3 h low-storage tail | 2 | 8 | 114 | 0.993 | 0.985 | 0.616 | 0.338 | -0.212 |
| 20 in | Early transient, 0-3 h | 1 | 7 | 84 | 0.800 | 0.718 | 0.717 | 0.588 | 0.238 |
| 20 in | Post-3 h low-storage tail | 1 | 7 | 66 | 0.995 | 0.991 | 0.123 | 0.085 | 0.015 |

## Local-model median held-out accuracy

| Layer | Segment | Valid local models | Median test events | Median CCC | Median R2 | Median RMSE mm | Median MAE mm |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 in | Early transient, 0-3 h | 29 | 9.0 | 0.976 | 0.959 | 0.553 | 0.383 |
| 4 in | Post-3 h mid-storage | 27 | 8.0 | 0.988 | 0.977 | 0.403 | 0.322 |
| 4 in | Post-3 h low-storage tail | 29 | 9.0 | 0.985 | 0.971 | 0.378 | 0.281 |
| 8 in | Early transient, 0-3 h | 8 | 4.0 | 0.889 | 0.794 | 0.519 | 0.424 |
| 8 in | Post-3 h mid-storage | 8 | 3.5 | 0.922 | 0.861 | 0.975 | 0.737 |
| 8 in | Post-3 h low-storage tail | 8 | 4.0 | 0.939 | 0.884 | 0.585 | 0.425 |
| 12 in | Early transient, 0-3 h | 2 | 5.5 | 0.968 | 0.938 | 0.632 | 0.513 |
| 12 in | Post-3 h mid-storage | 2 | 4.5 | 0.848 | 0.594 | 0.483 | 0.405 |
| 12 in | Post-3 h low-storage tail | 2 | 5.0 | 0.771 | 0.518 | 0.709 | 0.555 |
| 16 in | Early transient, 0-3 h | 2 | 4.0 | 0.775 | 0.711 | 2.499 | 1.990 |
| 16 in | Post-3 h mid-storage | 1 | 2.0 | 0.689 | 0.607 | 0.620 | 0.536 |
| 16 in | Post-3 h low-storage tail | 2 | 4.0 | 0.965 | 0.927 | 0.571 | 0.358 |
| 20 in | Early transient, 0-3 h | 1 | 7.0 | 0.800 | 0.718 | 0.717 | 0.588 |
| 20 in | Post-3 h low-storage tail | 1 | 7.0 | 0.995 | 0.991 | 0.123 | 0.085 |

## Interpretation

- The 4 in layer has the strongest held-out support because it contributes the largest number of valid local validation models and test events.
- The 8 in layer is still useful as secondary evidence, but its validation sample is much smaller.
- Sparse 12-20 in validation rows should be treated as exploratory; skipped rows mostly reflect insufficient events for a strict event-level 80/20 split after preserving at least eight training events.
- CCC is reported alongside RMSE, MAE, bias, and R2 to keep consistency with the earlier CSR diagnostics.

Skipped local validation rows: 10.

## Output files

- `fig_experiment3_split_validation_accuracy.svg/pdf/png/tiff`
- `source_data/experiment3_split_validation_pooled_by_layer.csv`
- `source_data/experiment3_split_validation_pooled_by_layer_segment.csv`
- `source_data/experiment3_split_validation_local_metrics.csv`
- `source_data/experiment3_split_validation_local_summary_by_layer_segment.csv`
- `source_data/experiment3_split_validation_predictions.parquet`
