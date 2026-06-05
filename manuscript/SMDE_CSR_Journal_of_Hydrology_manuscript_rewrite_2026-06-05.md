# Localized segmented soil-moisture recession curves from rainfall-validated drydown events across a monitoring network

Chi Zhang a, Sandra Guzman a, Jasmeet Judge a, Yilin Zhuang a, Chang Zhao b, Ziwen Yu a*

a Agricultural and Biological Engineering Department, University of Florida, Gainesville, FL 32603, United States

b Agronomy Department, University of Florida, Gainesville, FL 32611, United States

*Correspondence: ziwen.yu@ufl.edu

## Highlights

- Soil-moisture-only drydowns were validated with independent rainfall records.
- Loss-function diagnosis separated storage-limited events from transients.
- Localized segmented CSR was strongest at 4 in and secondary at 8 in.
- Held-out CSR curve agreement achieved CCC = 0.992 in the 4 in layer.
- Deeper-layer CSR estimates were promising but data-limited.

## Abstract

Soil moisture drydowns record the integrated effects of drainage, redistribution, evapotranspiration, and storage limitation, but noisy event detection and regime mixing can obscure recession behavior in monitoring networks. We developed a soil-moisture-first workflow that detects soil moisture drydown events (SMDEs) from soil moisture alone, audits them with independent precipitation records, diagnoses loss-function regimes, and constructs localized segmented curve-stitching regression (CSR) curves. The workflow was evaluated with 2023-2025 observations from 41 Florida Automated Weather Network sites at 4, 8, 12, 16, and 20 in depths. Soil-moisture-only detection identified 7,147 SMDEs, of which 5,520 (77.2%) were associated with rainfall within 48 h and 4,044 (56.6%) were clean rainfall-associated events with no rain interruption. Loss-function diagnosis showed that detected events were not a single drydown population: 3,292 events (46.1%) were stage-II-like, whereas 3,128 (43.8%) were early-transient-heavy. The main CSR event pool therefore used 1,873 clean stage-II-like events. Localized segmented CSR was fitted separately by location, layer, and segment, with strongest support at 4 in and secondary support at 8 in. Event-level 80/20 validation showed high held-out curve agreement after training-only curve fitting and test-event state alignment, with CCC = 0.992, R2 = 0.984, RMSE = 0.850 mm, and MAE = 0.463 mm at 4 in. This rainfall-validated and regime-aware framework provides a reproducible route for converting in situ soil moisture networks into localized recession references while identifying where deeper-layer support remains limited.

Keywords: soil moisture drydown; curve-stitching regression; loss function; localized recession modeling; FAWN; event-level validation; concordance correlation coefficient

## 1. Introduction

Soil moisture drydowns provide a compact observational record of how near-surface and root-zone storage changes after water input. Once rainfall or irrigation raises soil water content, subsequent decreases reflect a mixture of drainage, vertical redistribution, soil evaporation, plant water uptake, and atmospheric demand (Feddes et al., 1976; Laio et al., 2001; Seneviratne et al., 2010; Vereecken et al., 2016). These post-input trajectories are useful because they connect high-frequency sensor records to hydrologic behavior that is not visible from isolated soil moisture values alone. A drydown event can show whether a layer loses water rapidly or slowly, whether depletion changes with storage state, and whether different depths behave as comparable or delayed parts of the same profile response.

In situ monitoring networks now make such drydown records available across many sites, but converting raw sensor histories into process-relevant recession curves remains difficult. Soil moisture time series contain sensor noise, short reversals, missing records, and local wetting signals that do not always correspond cleanly to measured precipitation. Detected decreases may also represent different physical phases of the post-wetting response. Some periods are dominated by rapid redistribution or drainage, some may behave as approximately storage-invariant atmospheric-demand-limited loss, and others are storage-limited as soil water becomes less available. Pooling these phases into a single empirical recession relation can obscure the soil-water response that a local curve is intended to represent.

Previous drydown studies have provided the conceptual basis for using soil moisture loss functions to interpret these phases. Large-scale and satellite-based analyses have shown that drydown rates vary with climate, vegetation, soil texture, antecedent wetness, and land surface conditions (Akbar et al., 2018; McColl et al., 2017; Sehgal et al., 2021; Shellito et al., 2016, 2018; Tso et al., 2023). McColl et al. (2017) emphasized that soil moisture losses can be interpreted through regimes in which early drainage and runoff, stage-I evapotranspiration, and stage-II water-limited evapotranspiration make different contributions to the observed loss-storage relation. In situ studies provide complementary evidence but also show that drydown analyses are often designed around different targets: estimating moisture-dependent root-zone water loss (Salvucci, 2001), isolating drainage-dominated events for drainage-rate estimation (Jalilvand et al., 2018), characterizing drydown time scales across sensor networks and gridded products (Tso et al., 2023), or identifying evapotranspiration regime transitions using flux towers and in situ surface and profile soil moisture (Dong et al., 2022). These studies support the use of loss-function thinking, but they also caution against treating soil-moisture-only event labels as direct flux partitions. Applied sensor-network workflows therefore need to audit whether the event library is rainfall-associated, clean of new input, and consistent with storage-limited behavior before using it for local curve fitting.

This study addresses that gap by rebuilding localized curve-stitching regression (CSR) as a vetted drydown-event workflow. Here, CSR denotes a state-alignment and local-smoothing procedure that stitches repeated drydown segments into a station-layer recession reference; it is not used as a mechanistic flux-partitioning model. We define a soil moisture drydown event (SMDE) as a sustained decrease in measured soil water amount beginning from a local wetness peak. We then ask four linked questions. First, can SMDEs be detected from soil moisture alone and independently validated with precipitation records? Second, how much of the detected event library is consistent with storage-limited loss-function behavior rather than early transient or stage-I-like behavior? Third, which location-layer-segment combinations have enough clean stage-II-like events to support localized segmented CSR? Fourth, how closely do the resulting local CSR curves reproduce held-out drydown states under event-level 80/20 validation?

Using 2023-2025 data from 41 Florida Automated Weather Network (FAWN) sites, we show that the main contribution is the full event-to-curve chain: soil-moisture-only detection, independent rainfall audit, loss-function regime diagnosis, localized segmented curve construction, and held-out validation. The contribution is therefore not a new universal three-regime loss model. It is a reproducible sensor-network workflow that translates established drydown and loss-function concepts into explicit event filtering, local segmentation, and independent-event curve-agreement testing. The regime and segmentation definitions used throughout the study are summarized conceptually in Fig. 1. This framing positions CSR as a hydrological-methods contribution for identifying the subset of drydown observations that can defensibly enter localized recession modeling, and for quantifying where that modeling is well supported across a monitoring network.

## 2. Materials and methods

### 2.1 Conceptual framework: soil moisture loss functions and diagnostic drydown regimes

The workflow is based on the soil moisture loss-function view of post-input drydowns. For a monitored layer with effective thickness `Delta z`, volumetric water content `theta(t)`, and equivalent soil water storage `S(t) = Delta z theta(t)`, the layer water balance can be written as:

$$
\Delta z \frac{d\theta}{dt} = P(t) - L(\theta,t)
$$

or, equivalently,

$$
\frac{dS}{dt} = P(t) - L(S,t)
$$

where `P(t)` represents water input to the monitored support and `L` represents the aggregate loss term, including drainage, runoff-related profile adjustment, vertical redistribution, soil evaporation, and plant water uptake. During a post-input drydown with no new precipitation input, the balance reduces to:

$$
P(t) \approx 0, \qquad L(S,t) = -\frac{dS}{dt}
$$

This means that a drydown can be interpreted as a sampled loss-storage relation rather than only as a time series. In this representation, an upward curve in a loss-storage plot does not mean that soil water amount increases with time. It means that loss rate is larger at wetter storage states; as the same drydown evolves through time, it moves from the wet end of the storage axis toward the dry end.

Following the canonical soil moisture loss-function framework used in ecohydrology and drydown studies, three broad regimes can be distinguished conceptually (Laio et al., 2001; McColl et al., 2017; Akbar et al., 2018). At high wetness, drainage, runoff-related adjustment, and redistribution can dominate, so the loss rate increases strongly with storage. At intermediate wetness, stage-I evapotranspiration is controlled mainly by atmospheric demand and is therefore approximately invariant with respect to soil moisture. At low wetness, stage-II evapotranspiration is limited by soil water availability, and the loss rate declines as storage decreases. A compact conceptual form is:

$$
L(\theta) \approx
\begin{cases}
L_{wet}(\theta), & \theta > \theta_{fc} \\
E_{max}, & \theta^{*} < \theta \le \theta_{fc} \\
\beta(\theta) E_{max}, & \theta_w < \theta \le \theta^{*}
\end{cases}
$$

where `theta_fc` is a field-capacity-like wetness threshold, `theta*` separates stage-I-like and stage-II-like behavior, `theta_w` is a wilting-point-like lower bound, `E_max` is the atmospheric-demand-controlled loss rate, and `beta(theta)` declines as soil water availability becomes limiting.

The present study uses this three-regime framework as a diagnostic guide rather than a direct flux-partitioning model. McColl et al. (2017) analyzed SMAP surface soil moisture with a multi-day revisit interval and noted that early drainage and stage-I behavior may occur faster than the satellite sampling can resolve, making many retained drydowns effectively stage-II dominated. In contrast, FAWN in situ sensors provide 15-min observations, so the early post-wetting response can be visible within the event record. We therefore treat the first 3 h as an operational early-transient period and use the terms stage-I-like/storage-invariant and stage-II-like/storage-limited for soil-moisture-based diagnostic behavior, not for directly measured evapotranspiration, drainage, or runoff fluxes. Figure 1 summarizes this conceptual distinction and the operational segmentation used later for localized CSR.

[[FIGURE:1]]

### 2.2 Study network and data

The analysis used observations from the Florida Automated Weather Network (FAWN) for 2023-2025. The dataset included 41 FAWN sites with soil moisture observations at 4, 8, 12, 16, and 20 in depths and colocated weather observations. Soil moisture variables were obtained from FAWN records as volumetric percentage measurements and converted to equivalent soil water amount in millimeters for the monitored support following the existing project workflow. Specifically, volumetric percentage was converted as `S = (theta_pct/100) Delta z`, where `Delta z = 101.6 mm` corresponds to the 4 in sensing interval used for project thresholding. The depth labels therefore denote sensor-depth records expressed as equivalent water amount over this project-defined support, not integrated non-overlapping profile storage from the surface to each depth. Rainfall was taken from the primary 2 m rain gauge when available and from the backup 2 m rain gauge when the primary record was missing. Weather variables used as context included vapor pressure deficit, air temperature, relative humidity, wind speed, radiation, and soil temperature, but precipitation was the only weather variable used for event validation in the core experiments.

All data were processed at the native 15-min resolution after timestamp parsing, duplicate removal, numeric conversion, and basic range screening. Soil moisture values were removed when they were missing, unrealistic, or part of short internal outlier windows. Precipitation was aggregated to the same timestamp support as the soil moisture records and clipped to non-negative values. The analysis was applied separately to each FAWN site and each soil moisture layer, so that detection, regime diagnosis, CSR fitting, and validation all preserved location- and depth-specific behavior.

### 2.3 Soil-moisture-only SMDE detection

SMDE detection was performed using only the soil moisture time series. For each site-layer series, a candidate SMDE began at the first point of a sustained decreasing sequence and ended when the decreasing pattern was interrupted or the series stopped meeting the event criteria. A candidate sequence required at least three consecutive decreasing 15-min steps, corresponding to at least four soil moisture observations. Small measurement noise was tolerated by allowing a decrease condition of `S(t) <= S(t - 1) + epsilon`, where `epsilon` was set to 0.001 percentage points converted to millimeters using the project layer conversion. Candidate events were then screened for total decrease, average drop rate, internal range, low-value artifacts, short-window outliers, and dynamic upper limits.

The detection thresholds were intentionally conservative. The minimum total decrease was 0.5 percentage points expressed as soil water amount for a 4 in sensing interval, approximately 0.51 mm, and the maximum total decrease was 15 percentage points, approximately 15.24 mm. The minimum average drop rate was 0.04 mm per 15-min step. Events containing very low soil moisture values, excessive internal range, or outlier windows were excluded. This step produced an event table containing site, layer, start time, end time, duration, start soil water amount, end soil water amount, total decrease, and mean loss rate. No precipitation information was used to initiate or accept an SMDE at this stage.

### 2.4 Independent precipitation audit

After SMDE detection, each event was audited against FAWN precipitation records. The audit tested whether rainfall had occurred before the drydown and whether new rainfall interrupted the event. Rain association was evaluated over 12, 24, and 48 h before event start; the main manuscript uses the 48 h association window because rainfall-to-soil-moisture response timing can vary by depth and site. An event was considered rainfall-associated if cumulative rainfall in the prior window was greater than zero. An event was considered interrupted if cumulative rainfall during the event, excluding the event start timestamp, exceeded 0.02 in.

Four audit classes were assigned: clean rainfall-associated within 24 h, clean rainfall-associated within 24-48 h, rainfall-associated but interrupted, and not associated within 48 h. The clean 48 h class was defined as associated within 48 h and not interrupted by rainfall during the SMDE. This audit did not change how SMDEs were detected; it provided an independent validation layer that separated plausible post-input drydowns from drydowns that lacked measured rainfall support or were contaminated by new input.

### 2.5 Loss-function regime diagnosis

For each detected SMDE, we computed a loss function from consecutive soil moisture observations (Fig. 1). Let `S(t)` be soil water amount in millimeters and `Delta t_i` be the elapsed time between observations `i` and `i + 1` in hours. The event loss rate was calculated as:

$$
L_i = -\frac{S_{i+1}-S_i}{\Delta t_i}
$$

Positive values of `L_i` represent soil water loss. Each loss point was assigned a normalized event storage coordinate:

$$
x_i = \frac{S_{mid,i}-S_{min}}{S_{max}-S_{min}}
$$

where `S_mid,i` is the midpoint soil water amount of interval `i`, and `S_min` and `S_max` are the event minimum and maximum soil water amounts. This normalized storage coordinate places all drydowns on a common 0-1 scale, with larger values representing wetter event states.

The first 3 h after SMDE onset were treated as a separate early period because rapid drainage, redistribution, runoff-related adjustment, and sensor/layer equilibration may be strongest immediately after wetting. The 3 h cutoff was therefore used as an operational separation between early wet-end behavior and post-transient loss-function behavior, not as a claim that all early losses are one physical process. The early drop share was calculated as:

$$
E_3 = \frac{S(0)-S(\min(3h,t_{end}))}{S(0)-S(t_{end})}
$$

The regime diagnosis used three event-level diagnostics: the early drop share, the goodness of fit of an exponential drydown after trimming the first 3 h, and the post-3 h correlation between loss rate and normalized storage. Events were classified as stage-II-like when the post-3 h exponential fit had `R2 >= 0.7` and the post-3 h loss-storage correlation was at least 0.2:

$$
\mathrm{Stage\ II\!-\!like}: \quad R^2_{\mathrm{trim3}} \ge 0.70 \ \mathrm{and}\ r(L,x)_{\mathrm{post3}} \ge 0.20
$$

This diagnostic corresponds to events in which the loss rate is higher at wetter event states and decreases as the event moves toward lower storage, consistent with a storage-limited drydown. In the loss-storage plots, the horizontal axis is storage state rather than elapsed time; therefore, drydowns move from right to left through time. Events were classified as stage-I-like when the absolute post-3 h loss-storage correlation was less than 0.2 and the coefficient of variation of post-3 h loss rate was less than 0.6:

$$
\mathrm{Stage\ I\!-\!like}: \quad |r(L,x)_{\mathrm{post3}}| < 0.20 \ \mathrm{and}\ CV(L)_{\mathrm{post3}} < 0.60
$$

This diagnostic corresponds to an approximately storage-invariant post-3 h loss rate. Events were classified as early-transient-heavy when at least 40% of the total event decrease occurred within the first 3 h:

$$
\mathrm{Early\ transient\!-\!heavy}: \quad E_3 \ge 0.40
$$

Remaining events were classified as mixed or uncertain:

$$
\mathrm{Mixed/uncertain}: \quad \mathrm{not\ meeting\ the\ above\ diagnostic\ rules}
$$

These categories are diagnostic proxies rather than direct physical partitions of drainage, runoff, evaporation, transpiration, and redistribution. They were used to protect the main CSR construction from mixing hydrologically different event types. The main CSR event pool was defined as clean stage-II-like events within the 48 h precipitation audit window.

### 2.6 Localized segmented CSR construction

Localized segmented curve-stitching regression (CSR) was fitted separately for each location, layer, and segment. The main event subset was `clean_stageII_48h`, defined as rainfall-associated within 48 h, not interrupted by rainfall during the event, and classified as stage-II-like by the loss-function diagnosis. The all-events and clean-rainfall-only variants were retained as sensitivity contrasts, but they did not define the main recession curves because they mix non-rain-associated, rain-interrupted, or regime-mixed drydowns into the fitting population. Before fitting, each event was converted into point-level records with elapsed time, soil water amount, total event decrease, and normalized event storage.

Each event was divided into three segments using elapsed time and event storage state (Fig. 1a). For CSR segmentation, event storage was normalized by the event start and end soil water amounts:

$$
x_s(t) = \frac{S(t)-S_{end}}{S_{start}-S_{end}}
$$

The three CSR segments were then defined as:

$$
\mathrm{early\_0\_3h}: \quad t < 3h
$$

$$
\mathrm{post3\_mid\_storage}: \quad t \ge 3h \ \mathrm{and}\ x_s(t) \ge 0.25
$$

$$
\mathrm{post3\_late\_storage}: \quad t \ge 3h \ \mathrm{and}\ x_s(t) < 0.25
$$

The `early_0_3h` segment was retained as a separate transient segment even within stage-II-like events because the stage-II-like diagnosis was based on post-3 h behavior. The `post3_mid_storage` segment was treated as the primary post-transient storage-dependent portion of the drydown. The `post3_late_storage` segment was interpreted cautiously as a low-storage tail where drying may slow or become more sensitive to noise.

Within each location-layer-segment unit, events were aligned by state-stitching. Events were sorted by their soil water ranges, and each event was placed into a common CSR coordinate by matching overlapping or adjacent soil water states to a median anchor derived from previously aligned observations. This allowed events with different initial wetness and duration to contribute to a common recession coordinate without forcing them to share calendar timing. A locally weighted scatterplot smoother (LOWESS) was then fitted to binned aligned observations. The LOWESS fraction was chosen adaptively as `min(0.5, max(0.12, 40/n))`, where `n` was the number of binned observations. A local segment was fitted only when at least eight events and 40 aligned observations were available.

### 2.7 Held-out validation and agreement metrics

Held-out curve agreement was evaluated using event-level 80/20 validation within each eligible location-layer-segment unit. Events, rather than individual 15-min observations, were split into training and test sets to avoid leakage from the same drydown trajectory into both fitting and evaluation. A local validation unit was retained when at least eight training events, two test events, 40 training points, and eight test points were available. Training events were state-stitched and smoothed to build the local CSR curve. Test events did not contribute to the training anchor or LOWESS smoother.

For evaluation, each held-out event was aligned to the training-event anchor using its observed soil water states and segment time to obtain a CSR coordinate; the fitted training curve was then evaluated at that coordinate. The resulting estimates therefore test independent-event state agreement of the learned local recession reference after state alignment. They should not be interpreted as prospective real-time forecasts made from event-start information alone.

Agreement was summarized using the concordance correlation coefficient (CCC), coefficient of determination (R2), Pearson correlation, root mean square error (RMSE), mean absolute error (MAE), mean bias, normalized RMSE as a percentage of observed range, and dynamic MAE. CCC was included because it measures agreement with the 1:1 line and therefore penalizes both correlation error and scale or location bias (Lin, 1989). Dynamic MAE measured the absolute error in consecutive-step depletion increments, comparing curve-derived and observed 15-min soil water losses within each held-out event.

## 3. Results

### 3.1 Soil-moisture-only detection produced a large event library that was mostly rainfall-associated

Soil-moisture-only detection identified 7,147 SMDEs across the FAWN network (Fig. 2). Independent precipitation auditing showed that most detected drydowns were associated with measured rainfall, although a substantial subset required filtering before modeling. Overall, 5,520 SMDEs (77.2%) had rainfall within 48 h before event start. Of the full detected event library, 4,044 events (56.6%) were clean rainfall-associated events, meaning they were associated with rainfall within 48 h and were not interrupted by rain during the event. After adding the loss-function screen, 1,873 events (26.2% of detected SMDEs) remained in the clean stage-II-like construction subset.

Rainfall association varied by depth (Table 1). The 4 in layer produced the largest event count, with 4,276 detected SMDEs, of which 79.6% were rainfall-associated within 48 h and 61.4% were clean rainfall-associated. The 8, 12, and 16 in layers also showed high 48 h rainfall association, ranging from 75.7% to 82.7%, but the share of events retained after clean and stage-II-like filtering declined with depth. The 20 in layer had the weakest rainfall association, with 52.6% of detected events associated within 48 h and 7.2% retained as clean stage-II-like events.

These results support the use of soil-moisture-only detection as a first step, while also showing why independent precipitation auditing is necessary. The detection procedure captured many plausible post-input drydowns, but the audit identified events that were either not linked to measured rainfall in the 48 h window or were interrupted by new input. The clean stage-II-like subset therefore represents a deliberately narrower modeling population rather than all observed decreases in soil moisture.

Table 1. SMDE detection and precipitation audit summary by layer.

| Layer | Detected SMDEs | Rain <=48 h | Clean rain <=48 h | Clean + stage-II-like |
|---|---:|---:|---:|---:|
| 4 in | 4,276 | 79.6% | 61.4% | 34.4% |
| 8 in | 1,014 | 79.4% | 44.4% | 20.1% |
| 12 in | 624 | 82.7% | 55.9% | 15.1% |
| 16 in | 634 | 75.7% | 59.1% | 9.6% |
| 20 in | 599 | 52.6% | 40.7% | 7.2% |

[[FIGURE:2]]

### 3.2 Loss-function diagnosis showed strong regime mixing in detected SMDEs

Loss-function diagnosis showed that detected SMDEs did not behave as one uniform drydown population (Fig. 3). In Fig. 3a and b, the curves are plotted as loss rate versus storage state, not as soil water amount versus elapsed time. Thus, an upward curve means that wetter states lose water faster, whereas the same drydown moves from the wet end toward the dry end as time advances. Of the 7,147 detected events, 3,292 (46.1%) were classified as stage-II-like by the diagnostic proxy, 3,128 (43.8%) were early-transient-heavy, 449 (6.3%) were stage-I-like, and 278 (3.9%) were mixed or uncertain (Table 2). Thus, nearly half of the detected event library was dominated by early loss behavior rather than the post-transient storage-limited relation targeted by the main CSR construction.

The regime composition changed systematically with depth. The stage-II-like fraction was highest at 4 in (56.3%) and 8 in (47.8%) and declined to 32.5%, 19.2%, and 12.4% at 12, 16, and 20 in, respectively. In contrast, early-transient-heavy behavior increased with depth, from 31.2% at 4 in to 84.8% at 20 in. Stage-I-like events were present but comparatively uncommon, ranging from 1.3% to 7.9% across layers.

The loss-function results justify filtering the event library before constructing local CSR curves. If all detected events were pooled into one recession relation, early transient behavior and storage-limited behavior would be mixed, especially in deeper layers. The main CSR subset therefore used only events that were both clean in the precipitation audit and stage-II-like in the loss-function diagnosis. This filtering reduced sample size, but it made the modeling target more hydrologically coherent.

Table 2. Loss-function diagnostic regime composition by layer.

| Layer | Stage-II-like | Stage-I-like | Early transient | Mixed/uncertain |
|---|---:|---:|---:|---:|
| 4 in | 56.3% | 7.9% | 31.2% | 4.6% |
| 8 in | 47.8% | 5.7% | 42.0% | 4.4% |
| 12 in | 32.5% | 5.3% | 59.5% | 2.7% |
| 16 in | 19.2% | 1.9% | 77.3% | 1.6% |
| 20 in | 12.4% | 1.3% | 84.8% | 1.5% |

[[FIGURE:3]]

### 3.3 Localized segmented CSR was best supported at shallow depths

The clean stage-II-like event pool contained 1,873 events across 41 FAWN sites before segment-level support filtering. Localized segmented CSR support was strongest in the 4 in layer and secondary in the 8 in layer (Figs. 4 and 5). At 4 in, 30 local models met the event and point thresholds for the early transient segment, 28 met the threshold for the post-3 h mid-storage segment, and 30 met the threshold for the post-3 h low-storage tail. At 8 in, the corresponding counts were 10, 8, and 10 local models. Complete three-segment local CSR coverage was therefore strongest at 4 in, where 28 location-layers supported all three segments, and secondary at 8 in, where 8 location-layers supported all three segments.

Deeper layers had limited local support after precipitation and regime filtering. At 12 in, only two to three local models met the fitting threshold per segment. At 16 in, two local models were available per segment. At 20 in, one local model was available for the early and low-storage segments, and no mid-storage model met the main threshold. These deeper-layer fits were moved to the supplementary summary and treated as exploratory rather than primary evidence.

A representative complete local example was selected at FAWN site 275 in the 4 in layer (Fig. 4). This site-layer had all three segments fitted, non-extreme event support, and a median RMSE close to the network median for complete 4 in candidates. The early transient segment used 40 events and 480 points, with CCC = 0.981 and RMSE = 0.502 mm. The post-3 h mid-storage segment used 36 events and 704 points, with CCC = 0.921 and RMSE = 0.924 mm. The low-storage tail used 40 events and 645 points, with CCC = 0.959 and RMSE = 0.628 mm. This example illustrates how the segmented CSR separates an early post-wetting response from post-transient storage-dependent behavior while preserving local station-layer structure.

[[FIGURE:4]]

Table 3. Localized segmented CSR support and in-sample fit summary for primary and secondary layers. Sparse 12-20 in fits are reported in Supplementary Table S1 as exploratory results.

| Layer | Segment | Use | Local models | Median events/model | Total points | Median RMSE mm | Median MAE mm |
|---|---|---|---:|---:|---:|---:|---:|
| 4 in | Early transient, 0-3 h | main-text robust | 30 | 44.0 | 17,127 | 0.689 | 0.433 |
| 4 in | Post-3 h mid-storage | main-text robust | 28 | 36.5 | 17,823 | 0.648 | 0.438 |
| 4 in | Post-3 h low-storage tail | main-text robust | 30 | 44.0 | 24,488 | 0.450 | 0.251 |
| 8 in | Early transient, 0-3 h | secondary-supported | 10 | 16.0 | 1,836 | 0.626 | 0.428 |
| 8 in | Post-3 h mid-storage | secondary-supported | 8 | 13.5 | 2,410 | 0.629 | 0.383 |
| 8 in | Post-3 h low-storage tail | secondary-supported | 10 | 16.0 | 3,652 | 0.357 | 0.254 |

[[FIGURE:5]]

### 3.4 Held-out validation showed high shallow-layer curve agreement

Event-level 80/20 validation showed that localized segmented CSR reproduced held-out soil water states with high agreement in the shallow layers after test-event state alignment (Fig. 6). The 4 in validation set included 29 sites, 688 held-out events, and 11,979 held-out observations. Pooled across valid 4 in local validation units, CSR achieved CCC = 0.992, R2 = 0.984, Pearson r = 0.992, RMSE = 0.850 mm, MAE = 0.463 mm, bias = -0.046 mm, and nRMSE = 2.8% of the observed range. The 8 in layer included 8 sites, 65 held-out events, and 1,724 held-out observations, with CCC = 0.992, R2 = 0.985, RMSE = 0.887 mm, MAE = 0.563 mm, bias = -0.053 mm, and nRMSE = 3.3%.

Segment-level validation showed consistently strong 4 in agreement across the early, mid-storage, and low-storage segments. At 4 in, CCC ranged from 0.990 to 0.994 and RMSE ranged from 0.724 to 0.959 mm across segments. At 8 in, the low-storage tail had the strongest segment-level agreement (CCC = 0.996, RMSE = 0.671 mm), whereas the mid-storage segment had higher error (CCC = 0.985, RMSE = 1.120 mm). These results indicate that the segmented CSR curves did not merely describe the fitted event library; they remained consistent with independent held-out events when enough local event support was available.

Held-out agreement in deeper layers was numerically promising but underpowered. The 12 in layer included only 2 sites and 21 held-out events, the 16 in layer included 2 sites and 16 held-out events, and the 20 in layer included 1 site and 11 held-out events. CCC values remained high in these sparse validations, ranging from 0.911 to 0.972, but the small number of eligible local validation units limits inference. The deeper-layer results are therefore best interpreted as exploratory evidence that the method can be applied when support exists, not as network-wide validation of deep-layer CSR.

Table 4. Pooled held-out 80/20 curve-agreement metrics by layer. The 12-20 in rows are exploratory because few local validation units met the event-level split threshold.

| Layer | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm | nRMSE % range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 in | 29 | 688 | 11,979 | 0.992 | 0.984 | 0.850 | 0.463 | -0.046 | 2.8 |
| 8 in | 8 | 65 | 1,724 | 0.992 | 0.985 | 0.887 | 0.563 | -0.053 | 3.3 |
| 12 in | 2 | 21 | 434 | 0.972 | 0.945 | 0.615 | 0.480 | -0.059 | 4.8 |
| 16 in | 2 | 16 | 236 | 0.958 | 0.927 | 1.717 | 0.914 | 0.096 | 6.3 |
| 20 in | 1 | 11 | 150 | 0.911 | 0.849 | 0.543 | 0.366 | 0.140 | 9.0 |

[[FIGURE:6]]

## 4. Discussion

### 4.1 Main advance of the rainfall-validated, regime-aware CSR workflow

The main advance of this study is a complete workflow for turning high-frequency soil moisture records into vetted localized recession curves. Rather than treating every decreasing sequence as a modeling event, the workflow separates four tasks that are often conflated: detecting candidate drydowns from soil moisture, validating their relation to precipitation, diagnosing whether the loss-function behavior is suitable for storage-dependent recession modeling, and fitting local segment-specific curves. The resulting CSR curves are therefore not simply smoothed histories of all detected decreases. They are constructed from a narrower event library designed to represent clean, rainfall-associated, stage-II-like drydowns.

This framing is important for Journal of Hydrology because it makes the contribution a network-scale hydrological method rather than a site-specific curve fitting exercise. The study does not claim that CSR replaces mechanistic water-balance or Richards-equation models. Instead, it provides an empirical interpretation layer for sensor networks where detailed hydraulic parameters, complete management records, and full process partitioning may not be available. The high held-out agreement in the 4 in and 8 in layers indicates that this empirical layer can reproduce local drydown states after state alignment when event support is sufficient.

The regime interpretation should be read in that same empirical sense. The three-regime loss-function idea has strong precedent in ecohydrological theory and in satellite drydown studies, and Dong et al. (2022) provides direct support from flux tower and in situ soil moisture observations for identifying evapotranspiration regime transitions from surface soil moisture information. However, not all in situ drydown studies classify every event into three regimes. Some estimate a moisture-dependent loss function over longer records, some isolate drainage-dominated events, and others emphasize exponential drydown time scales. Our stage-I-like/storage-invariant, stage-II-like/storage-limited, and early-transient-heavy labels therefore organize the FAWN event library for conservative CSR construction; they do not claim that evapotranspiration, drainage, runoff, and redistribution have been separately measured.

### 4.2 Why precipitation audit is needed even when detection uses soil moisture alone

Detecting SMDEs from soil moisture alone has a practical advantage: it allows the event library to be defined from the variable being modeled, without requiring precipitation records to initiate the event window. However, the audit results show that detection alone is not enough. About three quarters of detected SMDEs were associated with rainfall within 48 h, but only 56.6% were both rainfall-associated and uninterrupted by new rain. The difference between detected and clean rainfall-associated events represents the portion of the event library that could otherwise introduce ambiguous or contaminated trajectories into CSR fitting.

This audit structure also protects against circular reasoning. Rainfall was not used to detect the drydown; it was used afterward as independent context. A drydown that is detected from soil moisture and then independently associated with recent rainfall is more defensible as a post-input event than a drydown defined by rainfall timing alone. Conversely, events not associated with measured rainfall are not necessarily false; they may reflect unmeasured irrigation, spatial rainfall mismatch, sensor behavior, or delayed profile response. The workflow does not discard their existence, but it excludes them from the main CSR construction to keep the modeling target conservative.

### 4.3 Regime filtering prevents over-averaging of different drydown processes

The loss-function diagnosis showed strong regime mixing: 46.1% of detected SMDEs were stage-II-like, whereas 43.8% were early-transient-heavy. This balance explains why unfiltered drydown modeling can be misleading. Early transient behavior after wetting may include rapid redistribution, drainage, runoff-related profile adjustment, or sensor equilibration. Stage-II-like behavior, by contrast, is more consistent with losses that decline as soil water availability decreases. Combining these phases in one local curve would make the fitted relation depend strongly on the mixture of event types available at each site and depth.

The depth pattern reinforces this point. The stage-II-like fraction declined from 56.3% at 4 in to 12.4% at 20 in, while early-transient-heavy behavior increased from 31.2% to 84.8%. This does not imply that deeper layers lack hydrological information. Rather, it suggests that deeper observed decreases are more likely to reflect delayed, damped, or mixed profile processes under the current diagnostic rules. For this dataset, shallow layers provided the clearest and most frequent storage-dependent drydown signal for localized segmented CSR.

### 4.4 Localization and segmentation define where CSR is reliable

Localized segmented CSR was intentionally fitted by location, layer, and segment instead of pooling across the network. This structure respects site-specific soil water ranges, sensor behavior, and local drydown histories. It also prevents the early post-wetting period from controlling the same smoother that represents post-transient storage-dependent depletion. The three-segment design made this explicit: the first 3 h were retained as a separate transient segment, the post-3 h mid-storage portion formed the primary CSR target, and the post-3 h low-storage tail was retained with cautious interpretation.

The support map provides a practical boundary on where the method should be used. The 4 in layer had robust network support, with 28 location-layers supporting all three segments and 29 sites entering held-out validation. The 8 in layer had secondary support, with fewer sites and events but still strong curve agreement. The 12, 16, and 20 in layers did not provide enough eligible local segments for strong network-scale claims. In future applications, these deeper layers should either be analyzed with longer records, relaxed support thresholds justified by independent validation, or separate lag-aware methods designed for delayed profile responses.

### 4.5 Agreement interpretation and practical implications

The held-out validation results provide the new curve-agreement conclusion for the manuscript. At 4 in, localized segmented CSR achieved CCC = 0.992 and RMSE = 0.850 mm on 11,979 held-out observations from 688 test events. At 8 in, it achieved the same pooled CCC of 0.992 and RMSE = 0.887 mm, although with a much smaller validation set. These values indicate strong agreement in both trajectory and magnitude after state alignment, because CCC penalizes scale and location bias in addition to correlation. Low mean bias in the 4 in and 8 in layers further suggests that the local curves did not systematically overestimate or underestimate held-out soil water amount.

For hydrological interpretation, these results support using segmented CSR as a localized recession reference for shallow soil moisture layers. The curves can summarize how a station-layer typically dries after rainfall-validated events, identify whether an observed drydown follows the local reference after alignment, and provide a compact empirical basis for comparing sites or layers. They should not be interpreted as complete water-balance reconstructions or as prospective forecasts. CSR estimates soil water states along aligned observed drydown trajectories; it does not independently estimate drainage, evapotranspiration, redistribution, or root uptake fluxes.

### 4.6 Limitations and future work

Several limitations define the scope of the results. First, precipitation validation depends on station rain gauge representativeness. Convective rainfall, local irrigation, or spatial mismatch between the gauge and the soil moisture sensor could cause true wetting events to appear unassociated with measured rainfall. Second, the loss-function regimes are diagnostic proxies. They are useful for filtering and organizing events, but they do not directly partition physical fluxes without additional atmospheric demand, vegetation-state, soil hydraulic, management, and preferably flux-tower or lysimeter information. Third, LOWESS provides an empirical smoother rather than a mechanistic model. It is appropriate for constructing local references from repeated events, but it should not be used to infer hydraulic parameters without further modeling. Fourth, the held-out validation uses observed soil water states from test events to align those events to the training-derived CSR coordinate. This evaluates independent-event agreement of the learned curve after alignment, not real-time forecasting based only on information available at event onset.

The event support thresholds also create an important boundary. Requiring clean, rainfall-associated, stage-II-like events and at least eight events per local segment improves interpretability but reduces coverage, especially at deeper layers. This tradeoff is appropriate for a conservative main analysis. The all-events and clean-rainfall-only variants are useful sensitivity checks because they show what is gained or lost by relaxing rainfall and regime filters, but they should not define the main CSR curve without an explicit hydrological reason for mixing event types. Future work could examine longer FAWN records, additional sensor networks, irrigation logs, and vegetation-state data to determine whether deeper-layer CSR can be strengthened. Future development should also test sensitivity to the 48 h rainfall association window, the 3 h transient cutoff, the 0.25 storage threshold between mid-storage and low-storage segments, and the regime classification criteria.

## 5. Conclusions

This study presents a rainfall-validated and regime-aware workflow for constructing localized segmented CSR curves from soil moisture drydowns. Across 41 FAWN sites from 2023 to 2025, soil-moisture-only detection identified 7,147 SMDEs, independent precipitation auditing identified 4,044 clean rainfall-associated events, and loss-function diagnosis selected 1,873 clean stage-II-like drydowns for the main CSR construction. Localized segmented CSR was robustly supported in the 4 in layer, secondarily supported in the 8 in layer, and exploratory in the 12-20 in layers. Event-level 80/20 validation showed strong held-out curve agreement at shallow depths, especially at 4 in, where CCC = 0.992, R2 = 0.984, RMSE = 0.850 mm, and MAE = 0.463 mm after state alignment. These results indicate that soil moisture recession modeling should explicitly audit precipitation association, diagnose loss-function regimes, and fit local segment-specific curves rather than treating all detected drydowns as one modeling population.

## Data availability

The raw FAWN observations used in this study are maintained by the Florida Automated Weather Network and are not redistributed by the authors. The derived event audit tables, loss-function diagnostic summaries, CSR source data, held-out validation outputs, figure source tables, and manuscript figures are available at https://github.com/garyzhang332/smde-csr-jhydrol. Users with authorized access to the FAWN project database can regenerate the yearly source parquet files using the database extraction scripts in the repository after setting the `FAWN_DB_URL` environment variable.

## Code availability

The Python scripts used for SMDE detection, precipitation audit, loss-function regime diagnosis, localized segmented CSR construction, held-out validation, and figure generation are available at https://github.com/garyzhang332/smde-csr-jhydrol.

## Acknowledgements

The authors acknowledge the Florida Automated Weather Network for supporting data collection and access to weather and soil moisture observations.

## Funding

This work was supported by USDA CIG under Grant No. NR213A750013G018.

## Author contributions

Chi Zhang led study design, data processing, analysis, method development, result interpretation, and manuscript drafting. Sandra Guzman and Jasmeet Judge contributed to conceptual development, methodological guidance, data interpretation, and manuscript revision. Yilin Zhuang and Chang Zhao contributed to manuscript drafting, revision, and critical review. Ziwen Yu supervised the study and contributed to study design, methodological development, data interpretation, manuscript revision, and overall guidance. All authors reviewed and approved the final manuscript.

## Declarations

Ethical considerations: this article does not contain any studies with human or animal participants.

Declaration of competing interests: the authors declare no competing interests.

## Figure captions

Figure 1. Regime diagnosis and segmentation logic for localized segmented CSR. (a) SMDE segmentation used for CSR, showing the first 3 h early segment, the post-3 h mid-storage segment, and the post-3 h low-storage tail defined by normalized event storage. (b) Conceptual loss-function regimes used to interpret post-wetting drydowns as loss rate versus storage, not loss rate versus elapsed time. Stage-II-like/storage-limited events show loss rates that increase with storage, stage-I-like/storage-invariant events have approximately storage-invariant post-3 h loss rates, and early-transient-heavy events are dominated by rapid wet-end loss. (c) Event-level diagnostic rules used for classification. These labels are diagnostic proxies for filtering and segmentation, not direct flux partitioning.

Figure 2. Soil-moisture-only SMDE detection and independent precipitation validation across FAWN stations. (a) Representative 4 in station-layer time series showing a detected SMDE window from soil moisture alone; rainfall bars are shown only as post-detection context. (b) Depth-wise counts of detected SMDEs grouped by post-detection rainfall-validation class. (c) Detection funnel from all soil-moisture-only SMDEs to rainfall-associated, clean rainfall-associated, and clean stage-II-like construction subsets. (d) Station-layer heatmap showing the fraction of detected events that were rainfall-associated within 48 h and not interrupted by rain during the event.

Figure 3. Loss-function regime diagnosis for detected FAWN SMDEs. (a) Conceptual three-regime soil moisture loss-function frame plotted as loss rate versus storage state; during a drydown, event time moves from wet to dry. (b) Empirical binned loss-storage relation for the 4 in layer, with median loss rate and interquartile ranges by diagnostic regime proxy. A positive loss-storage slope means faster loss at wetter states and slower loss as storage declines. (c) Depth-wise composition of detected SMDEs by diagnostic regime proxy. (d) Station-layer availability of clean rainfall-associated stage-II-like/storage-limited events for localized segmented CSR construction.

Figure 4. Representative localized segmented CSR construction for FAWN site 275 at the 4 in layer. Aligned event points and segmented CSR curves are shown for the early transient, post-3 h mid-storage, and post-3 h low-storage segments. The example illustrates how the local event library supports separate segment curves within one station-layer.

Figure 5. Network-level support for localized segmented CSR across FAWN station-layers. Panels summarize local model availability and fit quality by layer and segment after applying the clean stage-II-like event filter and minimum local support thresholds.

Figure 6. Held-out curve agreement of localized segmented CSR. (a) Segment-level 80/20 held-out RMSE by depth. (b) Segment-level held-out concordance correlation coefficient (CCC) by depth. (c) Training-curve estimates versus observed soil water amount for held-out 4 in and 8 in events after state alignment; the red diagonal is the 1:1 perfect-agreement line, not a drydown trajectory. Sparse 12-20 in results are exploratory because few local validation units met the event-level split threshold.

## Supplementary material plan

Supplementary Table S1. Full location-layer-segment CSR fit metrics, including sparse exploratory 12-20 in fits.

Supplementary Table S2. Full held-out validation metrics by location, layer, and segment.

Supplementary Table S3. Local validation units skipped because of insufficient events or points.

Supplementary Fig. S1. Sensitivity of CSR support to event-subset choice, including all-events, clean-rainfall-only, and clean stage-II-like variants.

Supplementary Note S1. Legacy CSR comparison material from the earlier manuscript version, retained only as historical sensitivity context if needed.

## References

Adla, S., Rai, N. K., Karumanchi, S. H., Tripathi, S., Disse, M., and Pande, S. (2020). Laboratory calibration and performance evaluation of low-cost capacitive and very low-cost resistive soil moisture sensors. Sensors, 20, 363.

Akbar, R., Short Gianotti, D. J., McColl, K. A., Haghighi, E., Salvucci, G. D., and Entekhabi, D. (2018). Estimation of landscape soil water losses from satellite observations of soil moisture. Journal of Hydrometeorology, 19, 871-889. https://doi.org/10.1175/JHM-D-17-0200.1

Bennett, N. D., Croke, B. F. W., Guariso, G., Guillaume, J. H. A., Hamilton, S. H., Jakeman, A. J., and others. (2013). Characterising performance of environmental models. Environmental Modelling & Software, 40, 1-20.

Chai, T., and Draxler, R. R. (2014). Root mean square error or mean absolute error? Geoscientific Model Development, 7, 1247-1250.

Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. Journal of the American Statistical Association, 74, 829-836.

Cleveland, W. S., and Devlin, S. J. (1988). Locally weighted regression: an approach to regression analysis by local fitting. Journal of the American Statistical Association, 83, 596-610.

Datta, S., and Taghvaeian, S. (2023). Soil water sensors for irrigation scheduling in the United States: a systematic review of literature. Agricultural Water Management, 278, 108148.

Dorigo, W. A., Wagner, W., Hohensinn, R., Hahn, S., Paulik, C., Xaver, A., Gruber, A., Drusch, M., Mecklenburg, S., van Oevelen, P., Robock, A., and Jackson, T. (2011). The International Soil Moisture Network: a data hosting facility for global in situ soil moisture measurements. Hydrology and Earth System Sciences, 15, 1675-1698.

Dong, J., Akbar, R., Short Gianotti, D. J., Feldman, A. F., Crow, W. T., and Entekhabi, D. (2022). Can surface soil moisture information identify evapotranspiration regime transitions? Geophysical Research Letters, 49, e2021GL097697. https://doi.org/10.1029/2021GL097697

Evett, S. R., Tolk, J. A., and Howell, T. A. (2006). Soil profile water content determination: sensor accuracy, axial response, calibration, temperature dependence, and precision. Vadose Zone Journal, 5, 894-907.

Feddes, R. A., Kowalik, P., Kolinska-Malinka, K., and Zaradny, H. (1976). Simulation of field water uptake by plants using a soil water dependent root extraction function. Journal of Hydrology, 31, 13-26.

Ghannam, K., Nakai, T., Paschalis, A., Oishi, C. A., Kotani, A., Igarashi, Y., Kumagai, T., and Katul, G. G. (2016). Persistence and memory timescales in root-zone soil moisture dynamics. Water Resources Research, 52, 1427-1445.

Jalilvand, E., Tajrishy, M., Brocca, L., Massari, C., Ghazi Zadeh Hashemi, S., and Ciabatta, L. (2018). Estimating the drainage rate from surface soil moisture drydowns: application of DfD model to in situ soil moisture data. Journal of Hydrology, 565, 489-501. https://doi.org/10.1016/j.jhydrol.2018.08.035

Laio, F., Porporato, A., Ridolfi, L., and Rodriguez-Iturbe, I. (2001). Plants in water-controlled ecosystems: active role in hydrologic processes and response to water stress. Advances in Water Resources, 24, 707-723.

Lin, L. I. K. (1989). A concordance correlation coefficient to evaluate reproducibility. Biometrics, 45, 255.

McColl, K. A., Wang, W., Peng, B., Akbar, R., Short Gianotti, D. J., Lu, H., Pan, M., and Entekhabi, D. (2017). Global characterization of surface soil moisture drydowns. Geophysical Research Letters, 44, 3682-3690. https://doi.org/10.1002/2017GL072819

Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. Water Resources Research, 12, 513-522.

Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. Physics, 1, 318-333.

Salvucci, G. D. (2001). Estimating the moisture dependence of root zone water loss using conditionally averaged precipitation. Water Resources Research, 37, 1357-1366. https://doi.org/10.1029/2000WR900336

Sehgal, V., Gaur, N., and Mohanty, B. P. (2021). Global surface soil moisture drydown patterns. Water Resources Research, 57, e2020WR027588.

Seneviratne, S. I., Corti, T., Davin, E. L., Hirschi, M., Jaeger, E. B., Lehner, I., Orlowsky, B., and Teuling, A. J. (2010). Investigating soil moisture-climate interactions in a changing climate: a review. Earth-Science Reviews, 99, 125-161.

Shellito, P. J., Small, E. E., Colliander, A., Bindlish, R., Cosh, M. H., Berg, A. A., Bosch, D. D., and others. (2016). SMAP soil moisture drying more rapid than observed in situ following rainfall events. Geophysical Research Letters, 43, 8068-8075.

Shellito, P. J., Small, E. E., and Livneh, B. (2018). Controls on surface soil drying rates observed by SMAP and simulated by the Noah land surface model. Hydrology and Earth System Sciences, 22, 1649-1663.

Simunek, J., van Genuchten, M. T., and Sejna, M. (2008). Development and applications of the HYDRUS and STANMOD software packages and related codes. Vadose Zone Journal, 7, 587-600.

Tso, C. H. M., Blyth, E., Tanguy, M., Levy, P. E., Robinson, E. L., Bell, V., Zha, Y., and Fry, M. (2023). Multiproduct characterization of surface soil moisture drydowns in the United Kingdom. Journal of Hydrometeorology, 24, 2299-2319. https://doi.org/10.1175/JHM-D-23-0018.1

van Genuchten, M. T. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. Soil Science Society of America Journal, 44, 892-898.

Vereecken, H., Schnepf, A., Hopmans, J. W., Javaux, M., Or, D., Roose, T., Vanderborght, J., Young, M. H., Amelung, W., Aitkenhead, M., Allison, S. D., Assouline, S., Baveye, P., Berli, M., Brueggemann, N., and others. (2016). Modeling soil processes: review, key challenges, and new perspectives. Vadose Zone Journal, 15, vzj2015.09.0131.

Willmott, C., and Matsuura, K. (2005). Advantages of the mean absolute error over the root mean square error in assessing average model performance. Climate Research, 30, 79-82.
