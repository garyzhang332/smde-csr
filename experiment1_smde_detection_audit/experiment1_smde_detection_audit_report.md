# Experiment 1: SMDE Detection Audit

Prepared: 2026-06-05

## Purpose

This experiment tests whether soil-moisture-only detection produces a credible soil moisture drying event (SMDE) library, and then uses FAWN rainfall records only as an independent post-detection validation layer.

## Core result

- Detected SMDEs: 7,147.
- Associated with rainfall within 48 h: 5,520 (77.2%).
- Clean rainfall-associated within 48 h: 4,044 (56.6%).
- Clean rainfall-associated and stage-II-like: 1,873 (26.2%).

## Layer summary

| Layer | Detected | Rain <=48 h | Clean <=48 h | Clean + stage-II-like |
|---|---:|---:|---:|---:|
| 4 in | 4,276 | 79.6% | 61.4% | 34.4% |
| 8 in | 1,014 | 79.4% | 44.4% | 20.1% |
| 12 in | 624 | 82.7% | 55.9% | 15.1% |
| 16 in | 634 | 75.7% | 59.1% | 9.6% |
| 20 in | 599 | 52.6% | 40.7% | 7.2% |

## Representative event used in panel a

- Site: 430.
- Layer: 4 in.
- Start: 2024-08-07 14:15:00.
- End: 2024-08-07 22:00:00.
- Duration: 7.75 h.
- Total drop: 1.67 mm.
- Rain lag: 0.25 h.

## Figure files

- `fig_experiment1_smde_detection_audit.svg`
- `fig_experiment1_smde_detection_audit.pdf`
- `fig_experiment1_smde_detection_audit.tiff`
- `fig_experiment1_smde_detection_audit.png`

## Draft figure legend

Figure X. Soil-moisture-only SMDE detection and independent precipitation validation across FAWN stations. (a) Representative 4-inch station-layer time series showing a detected SMDE window from soil moisture alone; rainfall bars are shown only as post-detection context. (b) Depth-wise counts of detected SMDEs grouped by post-detection rainfall-validation class. (c) Detection funnel from all soil-moisture-only SMDEs to rainfall-associated, clean rainfall-associated, and clean stage-II-like construction subsets. (d) Station-layer heatmap showing the fraction of detected events that were rainfall-associated within 48 h and not interrupted by rain during the event; blank cells indicate no detected events or insufficient data for that station-layer.

## Interpretation boundary

Rainfall is not used to detect SMDEs. It is used after detection to estimate which soil-moisture drying patterns are plausibly post-input and which are interrupted or not associated with measured rainfall.