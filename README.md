# SMDE-CSR manuscript code

This repository contains the public-facing code and PDF figures for the SMDE-CSR Journal of Hydrology manuscript workflow.

Data are not included in this repository. FAWN observations can be downloaded or obtained from the Florida Automated Weather Network (FAWN). After obtaining authorized FAWN data, place compatible local exports in the paths expected by the analysis scripts before rebuilding the workflow.

## Repository layout

- `_analysis/` - analysis, experiment, and figure-generation scripts.
- `figures/` - current manuscript figure PDFs only.
- `requirements.txt` - Python package requirements.

## Data access

The analysis uses FAWN soil moisture and rainfall observations. Please obtain the required data directly from FAWN:

- FAWN website: <https://fawn.ifas.ufl.edu/>
- For database-style local rebuilds, `_analysis/fawn_db_pull.py` expects an authorized database URL in the `FAWN_DB_URL` environment variable.
- Raw data files, database credentials, derived event tables, and generated source tables should not be committed to this repository.

## Python environment

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Rebuilding the workflow

Run commands from the repository root after preparing compatible FAWN data locally.

```bash
python _analysis/fawn_full_smde_audit.py
python _analysis/fawn_segmented_csr.py
python _analysis/build_experiment1_smde_detection_audit.py
python _analysis/build_experiment2_loss_function_regime_diagnosis.py
python _analysis/build_experiment3_adaptive_regime_csr.py
python _analysis/build_experiment4c_all_train_rate_forecast.py
```

## Figure build map

| Manuscript figure | Script | Output PDF |
|---|---|---|
| Figure 1 | `_analysis/build_figure1_study_data_overview.py` | `figures/Figure_1_study_area_and_data_overview.pdf` |
| Figure 2 | `_analysis/build_figure1_regime_segmented_framework.py` | `figures/Figure_2_regime_segmentation_concept.pdf` |
| Figure 3 | `_analysis/build_figure6_forecast_process_schematic.py` | `figures/Figure_3_forecast_process_schematic.pdf` |
| Figure 4 | `_analysis/build_experiment1_smde_detection_audit.py` | `figures/Figure_4_SMDE_detection_audit.pdf` |
| Figure 5 | `_analysis/build_experiment2_loss_function_regime_diagnosis.py` | `figures/Figure_5_loss_function_regime_diagnosis.pdf` |
| Figure 6 | `_analysis/build_figure5_regime_specific_csr_construction.py` | `figures/Figure_6_regime_specific_CSR_curve_construction.pdf` |
| Figure 7 | `_analysis/build_figure6_calibrated_recent_loss_forecast.py` | `figures/Figure_7_calibrated_recent_loss_forecast_validation.pdf` |

## Notes for public upload

This package intentionally excludes manuscript DOCX files, raw FAWN data, derived data tables, generated source-data folders, and non-PDF figure intermediates.

