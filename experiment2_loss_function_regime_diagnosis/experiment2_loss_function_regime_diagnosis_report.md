# Experiment 2: Loss-Function Regime Diagnosis

Prepared: 2026-06-05

## Purpose

This experiment diagnoses whether detected SMDEs behave as one uniform drydown population or as a mixture of loss-function regimes. It uses McColl-style loss-function logic as a conceptual frame, but applies event-level high-frequency FAWN diagnostics rather than assuming all events are stage-II drydowns.

## Core result

- Detected SMDEs evaluated: 7,147.
- Stage-II-like events by diagnostic proxy: 3,292 (46.1%).
- Early-transient-heavy events: 3,128 (43.8%).
- Clean rainfall-associated and stage-II-like construction subset: 1,873 (26.2%).

## Overall regime composition

| Regime proxy | Events | Fraction |
|---|---:|---:|
| Stage-II-like | 3,292 | 46.1% |
| Stage-I-like | 449 | 6.3% |
| Early transient | 3,128 | 43.8% |
| Mixed/uncertain | 278 | 3.9% |

## Layer-level regime composition

| Layer | Stage-II-like | Stage-I-like | Early transient | Mixed/uncertain |
|---|---:|---:|---:|---:|
| 4 in | 56.3% | 7.9% | 31.2% | 4.6% |
| 8 in | 47.8% | 5.7% | 42.0% | 4.4% |
| 12 in | 32.5% | 5.3% | 59.5% | 2.7% |
| 16 in | 19.2% | 1.9% | 77.3% | 1.6% |
| 20 in | 12.4% | 1.3% | 84.8% | 1.5% |

## CSR eligibility summary

A station-layer is counted as eligible when it has at least 8 clean stage-II-like events. Strong support is counted at at least 30 events.

| Layer | Eligible station-layers | Strong station-layers | Clean stage-II-like events |
|---|---:|---:|---:|
| 12 in | 3 | 0 | 94 |
| 16 in | 2 | 0 | 61 |
| 20 in | 1 | 1 | 43 |
| 4 in | 30 | 21 | 1,471 |
| 8 in | 10 | 0 | 204 |

## Figure files

- `fig_experiment2_loss_function_regime_diagnosis.svg`
- `fig_experiment2_loss_function_regime_diagnosis.pdf`
- `fig_experiment2_loss_function_regime_diagnosis.tiff`
- `fig_experiment2_loss_function_regime_diagnosis.png`

## Draft figure legend

Figure X. Loss-function regime diagnosis for detected FAWN SMDEs. (a) Conceptual three-regime soil moisture loss-function frame plotted as loss rate versus storage state; during a drydown, event time moves from wet to dry. (b) Empirical binned loss-storage relation for the 4-inch layer, with median loss rate and interquartile ranges by diagnostic regime proxy. A positive loss-storage slope means faster loss at wetter states and slower loss as storage declines. (c) Depth-wise composition of detected SMDEs by diagnostic regime proxy. Percent labels show the stage-II-like and early-transient-heavy fractions. (d) Station-layer availability of clean rainfall-associated stage-II-like events for local segmented CSR construction; numbers show station-layers with at least the minimum local event count.

## Interpretation boundary

The regime labels are diagnostic proxies, not direct physical partitioning of drainage, runoff, evaporation, transpiration, and redistribution. Stage-I-like behavior is especially conservative here and should be treated as a plausible component of the mixed/high-moisture response unless richer atmospheric, vegetation, or management data are added.