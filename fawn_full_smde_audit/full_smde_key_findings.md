# Full-station SMDE Detection Audit and Regime Composition Analysis

## Scope

- Source data: FAWN database export, 2023-2025.
- Stations with exported soil moisture and selected weather data: 44.
- Soil moisture depths analyzed: 4, 8, 12, 16, and 20 inches.
- SMDE detection input: soil moisture only.
- Weather data use: post-detection validation and regime interpretation only.

## Detection results

- Total SMDE events detected: 7,147.
- Stations with at least one detected event: 41.
- Stations with no events under the current filters: 241, 302, and 410.
- Loss-rate points generated for regime diagnostics: 201,018.

Across all depths:

- Events associated with rainfall within 48 h: 5,520 of 7,147, or 77.2%.
- Clean events associated with rainfall within 48 h and not interrupted by rain during the event: 4,044 of 7,147, or 56.6%.
- Clean 48 h + stage-II-like events: 1,873 of 7,147, or 26.2%.

## Layer-level interpretation

The 4 inch layer is the strongest candidate for CSR construction:

- 4,276 detected events.
- 79.6% associated with rainfall within 48 h.
- 18.6% interrupted by rainfall during the event.
- 61.4% clean within 48 h.
- 56.3% stage-II-like.
- 34.4% both clean within 48 h and stage-II-like.
- Median trimmed-3h exponential R2: 0.994.
- Median duration: 7.75 h.
- Median total drop: 1.66 mm.

The 8 inch layer is useful but should be treated as a secondary CSR sensitivity layer:

- 1,014 detected events.
- 79.4% associated with rainfall within 48 h.
- 35.9% interrupted by rainfall during the event.
- 44.4% clean within 48 h.
- 47.8% stage-II-like.
- 20.1% both clean within 48 h and stage-II-like.
- Median trimmed-3h exponential R2: 0.995.

The 12, 16, and 20 inch layers are weaker candidates for unfiltered CSR concatenation:

- 12 inch: 624 events; 32.5% stage-II-like; 59.5% early-transient-heavy.
- 16 inch: 634 events; 19.2% stage-II-like; 77.3% early-transient-heavy.
- 20 inch: 599 events; 12.4% stage-II-like; 84.8% early-transient-heavy.

These deeper layers should not be pooled with the shallow layer without separate sensitivity analysis.

## Regime-composition conclusion

The analysis supports a conservative manuscript claim:

CSR concatenation is best justified as a low-input empirical representation of clean, mostly stage-II-like soil moisture drydowns, especially in the 4 inch layer. The full detected SMDE set contains mixed behavior, including rain-interrupted events, non-rain-associated events, and early-transient-heavy events, so the method should explicitly define a clean CSR construction subset.

The strongest manuscript framing is:

1. Detect SMDE using soil moisture alone.
2. Validate detected events with independent precipitation records.
3. Separate clean rainfall-associated events from interrupted or non-associated events.
4. Diagnose drydown regime composition using loss-rate, trimmed exponential fit quality, and early-drop share.
5. Build the main CSR curve from clean 48 h + stage-II-like events, with all-events and trimmed-events as sensitivity analyses.

## Suggested manuscript figures

Figure: SMDE Detection Audit

- Event counts by depth and precipitation-validation class.
- Station-depth heatmap of detected event count.
- Lag from previous rain to SMDE start.
- Event duration versus total soil moisture decrease.

Figure: Regime Composition Analysis

- Binned loss rate versus normalized storage.
- Full-event versus trimmed exponential R2.
- Regime proxy composition by depth.
- Mean loss rate versus VPD during the event.

Figure: Site-layer Heatmaps

- Rain-associated within 48 h.
- Interrupted by rain.
- Stage-II-like proxy.

## Caveats

- Regime labels are diagnostic proxies, not definitive physical separations of drainage, stage-I evapotranspiration, and stage-II evapotranspiration.
- The conversion from percent volumetric water content to mm assumes a 4 inch representative layer for each FAWN sensor depth.
- Site 405 has unusually high event counts and should be inspected before final manuscript figures.
- Sites 241, 302, and 410 have no events under the current filter settings; this may reflect sensor behavior, data quality, or thresholds.
- The main paper should avoid claiming that all detected events are stage-II drydowns. The defensible claim is that a clean subset is sufficiently stage-II-like to justify CSR construction.
