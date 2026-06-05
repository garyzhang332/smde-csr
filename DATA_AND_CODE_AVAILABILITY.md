# Data and Code Availability

## Data availability

The raw FAWN observations used in this study are maintained by the Florida Automated Weather Network. Raw yearly soil moisture and weather parquet files are not redistributed in this repository. The repository provides derived event audit tables, loss-function diagnostic summaries, CSR source data, held-out validation outputs, and figure source tables that support the manuscript results.

If raw FAWN access is available, the extraction scripts can rebuild the yearly source parquet files from the project database by setting the `FAWN_DB_URL` environment variable and running `scripts/fawn_db_pull.py`.

## Code availability

The Python scripts used for event detection, precipitation audit, loss-function regime diagnosis, localized segmented CSR fitting, held-out validation, and figure generation are available in `scripts/`.

## Data-use note

Derived tables in this repository are provided for scholarly inspection and reproducibility of the associated manuscript. Reuse of original FAWN observations should follow FAWN data-use requirements.
