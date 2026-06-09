# SMDE-CSR response-centric manuscript code

This repository is a public-facing staging package for the SMDE-CSR manuscript
workflow. It documents the response-centric framing used in the current
candidate manuscript:

> Agricultural local variability is represented as reusable soil moisture
> event-response morphology rather than only as residual error around fixed soil
> moisture coordinates.

The package is intentionally code-and-figure focused. It does not include raw
FAWN observations, raw on-farm records, private project data, generated source
tables, database credentials, local absolute paths, or manuscript DOCX files.

## Repository layout

- `_analysis/` - public-facing analysis and figure-generation scripts.
- `figures/` - manuscript figure PDFs only.
- `docs/` - public documentation for data access and code mapping.
- `data/` - empty placeholder with data-access instructions.
- `requirements.txt` - Python package requirements.

## Data availability

Data are not redistributed in this repository.

FAWN observations should be obtained directly from the Florida Automated Weather
Network (FAWN), subject to FAWN data-access policies:

<https://fawn.ifas.ufl.edu/>

On-farm CIG raw records are not redistributed because they are subject to
project data-use constraints. See `docs/DATA_AVAILABILITY.md`.

## Python environment

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Response-centric workflow map

The core theory/evidence bridge is built by:

```bash
python _analysis/build_figure2_response_centric_theory.py
```

The response-centric 2 x 2 evidence tables are prepared by:

```bash
python _analysis/build_response_centric_2x2_summary.py
python _analysis/build_response_centric_2x2_tables_and_figure.py
```

These scripts expect compatible local source tables prepared by the manuscript
analysis workflow. Private source tables are not included in this repository.

## Figure map

| Manuscript figure | Public PDF |
|---|---|
| Figure 1 | `figures/Figure_1_study_area_and_data_overview.pdf` |
| Figure 2 | `figures/Figure_2_response_centric_theory_evidence.pdf` |
| Figure 3 | `figures/Figure_3_forecast_process_schematic.pdf` |
| Figure 4 | `figures/Figure_4_SMDE_detection_audit.pdf` |
| Figure 5 | `figures/Figure_5_loss_function_regime_diagnosis.pdf` |
| Figure 6 | `figures/Figure_6_regime_specific_CSR_curve_construction.pdf` |
| Figure 7 | `figures/Figure_7_calibrated_recent_loss_forecast_validation.pdf` |

## Public upload notes

Before uploading or replacing an existing GitHub repository, verify that:

- no raw data files are present;
- no database pull scripts or connection strings are present;
- no local absolute paths are present;
- no DOCX manuscript files are present;
- only PDF figures are in `figures/`.

This staging package has not been pushed to GitHub.
