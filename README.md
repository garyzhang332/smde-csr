# SMDE CSR Journal of Hydrology Reproducibility Package

This repository contains the code, derived data, figure source data, and manuscript-ready figures for:

**Localized segmented soil-moisture recession curves from rainfall-validated drydown events across a monitoring network**

The analysis uses 2023-2025 Florida Automated Weather Network (FAWN) soil moisture and weather observations to detect soil moisture drydown events (SMDEs), audit their precipitation association, diagnose loss-function regimes, and construct localized segmented curve-stitching regression (CSR) curves.

## Repository contents

| Path | Contents |
|---|---|
| `scripts/` | Python scripts used for SMDE detection, precipitation audit, loss-function diagnosis, CSR fitting, split validation, and figure generation. |
| `fawn_full_smde_audit/` | Derived event-level and loss-function tables from the full FAWN SMDE audit. |
| `fawn_segmented_csr/` | Derived CSR curves, fitted local metrics, sensitivity summaries, and related point-level outputs. |
| `experiment1_smde_detection_audit/` | Figure source tables and report for the SMDE detection audit. |
| `experiment2_loss_function_regime_diagnosis/` | Figure source tables and report for the loss-function regime diagnosis. |
| `experiment3_localized_segmented_csr/` | Figure source tables, representative-site data, and split-validation outputs. |
| `figures/` | Manuscript figures in PDF, SVG, and PNG formats. TIFF files are not stored here to keep the repository size manageable. |
| `fawn_db_export/coverage/` | Coverage summaries for the FAWN source-data pull. |
| `fawn_db_export/schema/` | Soil moisture and weather table schema used by the extraction scripts. |
| `manuscript/` | Current manuscript Markdown draft used to update the Data and Code Availability statements. |

## Data access model

The raw FAWN observations are not redistributed in this repository. They are maintained by FAWN and were accessed through the project database for the study period. This repository provides the derived event tables, summary tables, source-data tables, and validation outputs needed to inspect the manuscript results and regenerate the manuscript figures.

To rerun the complete workflow from raw FAWN records, place the yearly source parquet files in:

```text
fawn_db_export/data/
```

Expected filenames:

```text
soil_moisture_2023.parquet
soil_moisture_2024.parquet
soil_moisture_2025.parquet
wx_selected_2023.parquet
wx_selected_2024.parquet
wx_selected_2025.parquet
```

Alternatively, if database access is available, set `FAWN_DB_URL` and use `scripts/fawn_db_pull.py` to rebuild those files.

## Reproducing the main outputs

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The manuscript figures and summary tables can be regenerated from the included derived data:

```bash
python scripts/build_regime_segmentation_concept_figure.py
python scripts/build_experiment1_smde_detection_audit.py
python scripts/build_experiment2_loss_function_regime_diagnosis.py
python scripts/build_experiment3_localized_segmented_csr_summary.py
python scripts/build_experiment3_split_validation_accuracy.py
```

To rebuild the full derived workflow from raw yearly FAWN parquet files:

```bash
python scripts/fawn_full_smde_audit.py
python scripts/fawn_segmented_csr.py
python scripts/build_experiment1_smde_detection_audit.py
python scripts/build_experiment2_loss_function_regime_diagnosis.py
python scripts/build_experiment3_localized_segmented_csr_summary.py
python scripts/build_experiment3_split_validation_accuracy.py
```

## Main result mapping

| Manuscript result | Repository files |
|---|---|
| Experiment 1: SMDE detection audit | `experiment1_smde_detection_audit/source_data/`, `fawn_full_smde_audit/full_smde_event_audit.csv` |
| Experiment 2: loss-function regime diagnosis | `experiment2_loss_function_regime_diagnosis/source_data/`, `fawn_full_smde_audit/full_smde_binned_loss_by_layer_regime.csv` |
| Experiment 3: localized segmented CSR | `experiment3_localized_segmented_csr/source_data/`, `fawn_segmented_csr/` |
| Held-out curve agreement | `experiment3_localized_segmented_csr/source_data/experiment3_split_validation_*` |
| Manuscript figures | `figures/` |

## Notes

- Depth labels such as `moisture_4in` refer to FAWN sensor-depth records expressed as equivalent water amount over the project-defined 4 in sensing support.
- Regime labels are diagnostic proxies used for event filtering; they are not direct flux partitions.
- Held-out validation evaluates independent-event curve agreement after state alignment, not prospective real-time forecasting.
