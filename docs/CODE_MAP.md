# Code map

This file maps public-facing scripts to their role in the response-centric
SMDE-CSR manuscript workflow.

## Core response-centric scripts

| Script | Role | Data expectation |
|---|---|---|
| `_analysis/build_figure2_response_centric_theory.py` | Builds the response-centric theory/evidence bridge figure. | Requires 2 x 2 summary tables in `_analysis/response_centric_2x2/`. |
| `_analysis/build_response_centric_2x2_summary.py` | Builds conservative FAWN/on-farm static-vs-response summary metrics. | Requires local FAWN and on-farm forecast-origin/metric tables. |
| `_analysis/build_response_centric_2x2_tables_and_figure.py` | Builds compact manuscript tables and supporting response-centric figures. | Requires outputs from the 2 x 2 summary workflow. |
| `_analysis/build_response_centric_2x2_static_threshold.py` | Builds a lower-bound static-threshold sensitivity comparison. | Requires local forecast source tables. |
| `_analysis/build_onfarm_response_forecast_models.py` | Fits first-pass on-farm static and response-centric forecast models from prepared on-farm origins. | Requires locally prepared on-farm forecast-origin parquet files. |

## FAWN workflow support scripts

| Script | Role | Data expectation |
|---|---|---|
| `_analysis/fawn_full_smde_audit.py` | Detects and audits FAWN SMDEs from local FAWN exports. | Requires user-supplied FAWN export files under `data/fawn_exports/` or `FAWN_EXPORT_DIR`. |
| `_analysis/fawn_segmented_csr.py` | Builds segmented CSR source tables from audited FAWN SMDEs. | Requires outputs from FAWN SMDE audit. |
| `_analysis/hydro_csr_registration.py` | Provides hydrologically constrained CSR registration utilities. | Library script. |
| `_analysis/smde_regime_audit.py` | Provides SMDE detection and regime-audit utilities. | Optional example input can be supplied through `ALACHUA_WITH_RAIN`. |

## Internal scripts intentionally excluded

The on-farm inventory, input-audit, and forecast-origin construction scripts are
not included in this public staging package because the current internal
versions reference private project directory structures. They should only be
publicized after replacing project-specific defaults with user-supplied input
paths and after confirming that no raw on-farm records are redistributed.

Database extraction scripts are also excluded. Public users should obtain FAWN
observations from FAWN directly rather than through project-specific database
connection code.
