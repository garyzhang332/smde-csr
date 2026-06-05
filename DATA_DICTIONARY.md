# Data Dictionary

This file summarizes the main derived datasets in the repository. Column names use the same naming conventions as the analysis scripts.

## Common columns

| Column | Meaning |
|---|---|
| `site_id` | FAWN station identifier. |
| `layer` | Soil moisture sensor-depth variable, e.g. `moisture_4in`. |
| `event_id` | Unique SMDE identifier generated during the event audit. |
| `start`, `end` | Event start and end timestamp. |
| `duration_h` | Event duration in hours. |
| `start_mm`, `end_mm` | Event start and end equivalent soil water amount in millimeters. |
| `total_decrease_mm` | Total event decrease in equivalent soil water amount. |
| `associated_48h` | Whether rainfall was detected within 48 h before event start. |
| `interrupted_by_rain` | Whether rainfall occurred during the SMDE after event start. |
| `regime_proxy` | Diagnostic regime label: `stage-II-like`, `stage-I-like`, `early-transient-heavy`, or `mixed_or_uncertain`. |

## Experiment 1

`experiment1_smde_detection_audit/source_data/` contains depth-wise event counts, detection-funnel values, and station-layer clean-event rates used in Fig. 2 and Table 1.

## Experiment 2

`experiment2_loss_function_regime_diagnosis/source_data/` contains regime composition by layer, empirical binned loss-storage curves, and station-layer CSR eligibility summaries used in Fig. 3 and Table 2.

## Experiment 3

`experiment3_localized_segmented_csr/source_data/` contains CSR entry decisions, local segment metrics, representative-site aligned points and curves, split-validation metrics, and held-out prediction previews used in Figs. 4-6 and Tables 3-4.

## Full audit and CSR outputs

`fawn_full_smde_audit/` and `fawn_segmented_csr/` contain larger intermediate outputs used to regenerate the experiment-level source tables.
