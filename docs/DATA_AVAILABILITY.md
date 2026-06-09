# Data availability

Raw data are not included in this public package.

## FAWN observations

The FAWN weather and soil moisture observations used by the manuscript are
maintained by the Florida Automated Weather Network. Users should obtain FAWN
observations directly from FAWN, subject to FAWN data-access policies:

<https://fawn.ifas.ufl.edu/>

This package does not include database access scripts, database URLs,
credentials, local exports, or generated event/source tables.

## On-farm records

The on-farm CIG records used for the first-pass managed-setting comparison are
not redistributed because they are subject to project data-use constraints.
The public package documents the analysis logic and figure-generation workflow
without exposing raw on-farm records, local paths, or connection details.

## Rebuild expectations

To rebuild the full workflow, users must provide compatible local data exports
matching the column structure expected by the analysis scripts. Public example
schemas may be added later, but private source data should not be committed.
