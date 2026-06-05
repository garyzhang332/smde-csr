from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT
SEGCSR = ANALYSIS / "fawn_segmented_csr"
AUDIT = ANALYSIS / "fawn_full_smde_audit"
OUT = ANALYSIS / "experiment3_localized_segmented_csr"
SOURCE = OUT / "source_data"

MAIN_SUBSET = "clean_stageII_48h"
REP_LAYER = "moisture_4in"
EXCLUDE_REP_SITE = 405
MIN_LOCAL_SEGMENT_EVENTS = 8
MIN_LOCAL_SEGMENT_POINTS = 40

LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
SEGMENT_ORDER = ["early_0_3h", "post3_mid_storage", "post3_late_storage"]
SEGMENT_LABELS = {
    "early_0_3h": "Early transient, 0-3 h",
    "post3_mid_storage": "Post-3 h mid-storage",
    "post3_late_storage": "Post-3 h low-storage tail",
}
SEGMENT_SHORT = {
    "early_0_3h": "Early",
    "post3_mid_storage": "Mid-storage",
    "post3_late_storage": "Low-storage",
}
SEGMENT_COLORS = {
    "early_0_3h": "#C78D4B",
    "post3_mid_storage": "#355C7D",
    "post3_late_storage": "#6FA38A",
}
SUPPORT_COLORS = {
    "none": "#EFEFEF",
    "insufficient": "#D7C8A5",
    "eligible": "#7EADCA",
    "strong": "#355C7D",
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
    "curve": "#A64B56",
    "point": "#6E7176",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )


def layer_sort(frame: pd.DataFrame, column: str = "layer") -> pd.DataFrame:
    out = frame.copy()
    out[column] = pd.Categorical(out[column], categories=LAYER_ORDER, ordered=True)
    return out.sort_values(column)


def segment_sort(frame: pd.DataFrame, column: str = "segment") -> pd.DataFrame:
    out = frame.copy()
    out[column] = pd.Categorical(out[column], categories=SEGMENT_ORDER, ordered=True)
    return out.sort_values(column)


def read_inputs() -> dict[str, pd.DataFrame]:
    required = {
        "curves": SEGCSR / "segmented_csr_curves.csv",
        "binned": SEGCSR / "segmented_csr_binned_points.csv",
        "local_metrics": SEGCSR / "segmented_csr_local_segment_metrics.csv",
        "local_summary": SEGCSR / "segmented_csr_local_summary_by_layer_segment.csv",
        "subset_counts": SEGCSR / "segmented_csr_subset_counts.csv",
        "events": AUDIT / "full_smde_event_audit.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    frames = {
        name: pd.read_csv(path, parse_dates=["start", "end"] if name == "events" else None)
        for name, path in required.items()
    }
    return frames


def build_entry_decisions() -> pd.DataFrame:
    records = [
        {
            "decision_level": "event subset",
            "candidate": "all detected SMDEs",
            "enter_localized_segmented_csr": "no",
            "recommended_use": "Sensitivity or diagnostic comparison only",
            "implementation": "subset == all_events",
            "rationale": "Includes events without rainfall validation and events interrupted by rain; useful for showing why the audit filter matters.",
        },
        {
            "decision_level": "event subset",
            "candidate": "rainfall-associated clean events",
            "enter_localized_segmented_csr": "sensitivity",
            "recommended_use": "Rainfall-validation sensitivity analysis",
            "implementation": "associated_48h and not interrupted_by_rain",
            "rationale": "Confirms that events are tied to FAWN precipitation but still mixes loss-function regimes.",
        },
        {
            "decision_level": "event subset",
            "candidate": "rainfall-associated clean and stage-II-like events",
            "enter_localized_segmented_csr": "yes",
            "recommended_use": "Main localized segmented CSR event pool",
            "implementation": "associated_48h and not interrupted_by_rain and regime_proxy == stage-II-like",
            "rationale": "This is the manuscript-safe subset: rainfall-validated, not rain-interrupted, and storage-limited by the loss-function diagnosis.",
        },
        {
            "decision_level": "event subset",
            "candidate": "stage-I-like events",
            "enter_localized_segmented_csr": "no",
            "recommended_use": "Loss-function diagnostic only",
            "implementation": "regime_proxy == stage-I-like",
            "rationale": "Stage-I drying is approximately atmospheric-demand controlled and is not appropriate for the storage-dependent CSR calibration.",
        },
        {
            "decision_level": "event subset",
            "candidate": "early-transient-heavy events",
            "enter_localized_segmented_csr": "no",
            "recommended_use": "Quality-control diagnostic; do not use for main CSR",
            "implementation": "regime_proxy == early-transient-heavy",
            "rationale": "These events are dominated by early redistribution, drainage, runoff, or sensor adjustment rather than sustained storage-limited loss.",
        },
        {
            "decision_level": "event subset",
            "candidate": "mixed or uncertain events",
            "enter_localized_segmented_csr": "no",
            "recommended_use": "Quality-control diagnostic only",
            "implementation": "regime_proxy == mixed_or_uncertain",
            "rationale": "Regime identity is not strong enough to support a local CSR calibration.",
        },
        {
            "decision_level": "segment",
            "candidate": "early_0_3h",
            "enter_localized_segmented_csr": "yes, separate segment",
            "recommended_use": "Report separately as the post-wetting transient segment",
            "implementation": "t_h < 3 h within clean_stageII_48h events",
            "rationale": "Can be modeled locally, but should not be pooled with the storage-dependent post-3 h segments.",
        },
        {
            "decision_level": "segment",
            "candidate": "post3_mid_storage",
            "enter_localized_segmented_csr": "yes",
            "recommended_use": "Primary storage-dependent local CSR segment",
            "implementation": "t_h >= 3 h and event_storage_norm >= 0.25",
            "rationale": "This is the clearest post-transient storage-dependent interval for localized CSR.",
        },
        {
            "decision_level": "segment",
            "candidate": "post3_late_storage",
            "enter_localized_segmented_csr": "yes, cautious",
            "recommended_use": "Low-storage tail; include but interpret as availability-limited/noise-sensitive",
            "implementation": "t_h >= 3 h and event_storage_norm < 0.25",
            "rationale": "Useful for the tail of the local CSR curve, but low-storage noise and slow loss can dominate.",
        },
        {
            "decision_level": "local support",
            "candidate": "location-layer-segment fit",
            "enter_localized_segmented_csr": "yes if supported",
            "recommended_use": "Fit local curve only when both thresholds are met",
            "implementation": f">= {MIN_LOCAL_SEGMENT_EVENTS} events and >= {MIN_LOCAL_SEGMENT_POINTS} points",
            "rationale": "Keeps the localized CSR from being driven by very small site-layer-segment samples.",
        },
    ]
    return pd.DataFrame.from_records(records)


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "layer" in out.columns:
        out["layer"] = out["layer"].astype(str)
        out["layer_label"] = out["layer"].map(LAYER_LABELS)
    if "segment" in out.columns:
        out["segment"] = out["segment"].astype(str)
        out["segment_label"] = out["segment"].map(SEGMENT_LABELS)
        out["segment_short"] = out["segment"].map(SEGMENT_SHORT)
    return out


def classify_summary_use(row: pd.Series) -> str:
    models = int(row.get("local_models", 0))
    median_events = float(row.get("median_events_per_model", 0))
    if models >= 20 and median_events >= 30:
        return "main-text robust"
    if models >= 8 and median_events >= MIN_LOCAL_SEGMENT_EVENTS:
        return "secondary-supported"
    if models >= 1:
        return "exploratory/table only"
    return "not fitted"


def build_complete_layer_segment_summary(local_summary: pd.DataFrame) -> pd.DataFrame:
    main = local_summary[local_summary["subset"] == MAIN_SUBSET].copy()
    full = pd.MultiIndex.from_product([LAYER_ORDER, SEGMENT_ORDER], names=["layer", "segment"]).to_frame(index=False)
    summary = full.merge(main, on=["layer", "segment"], how="left")
    summary["subset"] = summary["subset"].fillna(MAIN_SUBSET)

    count_cols = ["local_models", "total_points"]
    for col in count_cols:
        summary[col] = summary[col].fillna(0).astype(int)
    numeric_cols = [
        "median_events_per_model",
        "median_rmse_mm",
        "mean_rmse_mm",
        "median_mae_mm",
        "mean_mae_mm",
        "median_bias_mm",
        "mean_bias_mm",
    ]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary = add_labels(summary)
    summary["recommended_use"] = summary.apply(classify_summary_use, axis=1)
    return layer_sort(segment_sort(summary), "layer")


def build_availability(events: pd.DataFrame, local_metrics: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["clean_48h"] = events["associated_48h"] & ~events["interrupted_by_rain"]
    events["clean_stageII_48h"] = events["clean_48h"] & (events["regime_proxy"] == "stage-II-like")

    all_sites = sorted(events["site_id"].dropna().astype(int).unique())
    full = pd.MultiIndex.from_product([all_sites, LAYER_ORDER], names=["site_id", "layer"]).to_frame(index=False)
    counts = (
        events.groupby(["site_id", "layer"], observed=False)
        .agg(
            detected_events=("event_id", "count"),
            associated_48h_events=("associated_48h", "sum"),
            clean_48h_events=("clean_48h", "sum"),
            clean_stageII_48h_events=("clean_stageII_48h", "sum"),
            stageII_like_events=("regime_proxy", lambda x: int((x == "stage-II-like").sum())),
        )
        .reset_index()
    )
    availability = full.merge(counts, on=["site_id", "layer"], how="left")
    for col in [
        "detected_events",
        "associated_48h_events",
        "clean_48h_events",
        "clean_stageII_48h_events",
        "stageII_like_events",
    ]:
        availability[col] = availability[col].fillna(0).astype(int)

    fitted = (
        local_metrics[local_metrics["subset"] == MAIN_SUBSET]
        .groupby(["site_id", "layer"], observed=False)
        .agg(
            fitted_segments=("segment", "nunique"),
            fitted_segment_names=("segment", lambda x: ",".join(sorted(set(x), key=SEGMENT_ORDER.index))),
            min_segment_events=("events", "min"),
            median_segment_events=("events", "median"),
            total_segment_points=("points", "sum"),
            median_local_rmse_mm=("rmse_mm", "median"),
            median_local_mae_mm=("mae_mm", "median"),
        )
        .reset_index()
    )
    availability = availability.merge(fitted, on=["site_id", "layer"], how="left")
    availability["fitted_segments"] = availability["fitted_segments"].fillna(0).astype(int)
    availability["fitted_segment_names"] = availability["fitted_segment_names"].fillna("")
    availability["min_segment_events"] = availability["min_segment_events"].fillna(0).astype(int)
    availability["total_segment_points"] = availability["total_segment_points"].fillna(0).astype(int)

    availability["support_class"] = pd.cut(
        availability["clean_stageII_48h_events"],
        bins=[-0.1, 0.1, MIN_LOCAL_SEGMENT_EVENTS - 0.1, 29.9, 10_000],
        labels=["none", "insufficient", "eligible", "strong"],
    ).astype(str)
    availability["csr_status"] = pd.cut(
        availability["fitted_segments"],
        bins=[-0.1, 0.1, 2.9, 3.1],
        labels=["not fitted", "partial", "complete"],
    ).astype(str)
    availability = add_labels(availability)
    return layer_sort(availability, "layer").sort_values(["site_id", "layer"])


def choose_representative_site(local_metrics: pd.DataFrame) -> tuple[int, str, pd.DataFrame]:
    main = local_metrics[
        (local_metrics["subset"] == MAIN_SUBSET)
        & (local_metrics["layer"] == REP_LAYER)
        & (local_metrics["site_id"] != EXCLUDE_REP_SITE)
    ].copy()
    complete = main.groupby("site_id").filter(lambda x: set(SEGMENT_ORDER).issubset(set(x["segment"])))
    if complete.empty:
        raise RuntimeError(f"No complete representative candidates found for {REP_LAYER}.")

    rows = []
    for site_id, group in complete.groupby("site_id"):
        wide = group.pivot_table(index="site_id", columns="segment", values="rmse_mm", aggfunc="median")
        row = {
            "site_id": int(site_id),
            "events_sum": int(group["events"].sum()),
            "events_min": int(group["events"].min()),
            "points_sum": int(group["points"].sum()),
            "rmse_median_mm": float(group["rmse_mm"].median()),
            "rmse_max_mm": float(group["rmse_mm"].max()),
            "rmse_spread_mm": float(group["rmse_mm"].max() - group["rmse_mm"].min()),
            "mae_median_mm": float(group["mae_mm"].median()),
            "abs_bias_median_mm": float(group["bias_mm"].abs().median()),
        }
        for segment in SEGMENT_ORDER:
            row[f"rmse_{segment}_mm"] = float(wide[segment].iloc[0])
        rows.append(row)

    candidates = pd.DataFrame(rows)
    network_median_rmse = float(main["rmse_mm"].median())
    median_events_sum = float(candidates["events_sum"].median())
    median_rmse_max = float(candidates["rmse_max_mm"].median())
    median_rmse_spread = float(candidates["rmse_spread_mm"].median())
    candidates["representative_score"] = (
        (candidates["rmse_median_mm"] - network_median_rmse).abs() / max(network_median_rmse, 1e-9)
        + 0.5 * (np.log(candidates["events_sum"]) - np.log(median_events_sum)).abs()
        + 0.4 * (candidates["rmse_max_mm"] - median_rmse_max).abs() / max(median_rmse_max, 1e-9)
        + 0.4 * (candidates["rmse_spread_mm"] - median_rmse_spread).abs() / max(median_rmse_spread, 1e-9)
        + 0.2 * candidates["abs_bias_median_mm"]
    )
    candidates = candidates.sort_values(["representative_score", "site_id"]).reset_index(drop=True)
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    return int(candidates.loc[0, "site_id"]), REP_LAYER, candidates


def build_source_tables(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int, str]:
    local_metrics = frames["local_metrics"].copy()
    local_summary = frames["local_summary"].copy()
    events = frames["events"].copy()

    entry = build_entry_decisions()
    summary = build_complete_layer_segment_summary(local_summary)
    availability = build_availability(events, local_metrics)

    local_main = local_metrics[local_metrics["subset"] == MAIN_SUBSET].copy()
    local_main = add_labels(local_main)
    local_main = layer_sort(segment_sort(local_main), "layer").sort_values(["layer", "site_id", "segment"])

    rep_site, rep_layer, rep_candidates = choose_representative_site(local_metrics)
    rep_metrics = local_main[(local_main["site_id"] == rep_site) & (local_main["layer"] == rep_layer)].copy()

    curves = frames["curves"]
    binned = frames["binned"]
    rep_curves = curves[
        (curves["subset"] == MAIN_SUBSET) & (curves["site_id"] == rep_site) & (curves["layer"] == rep_layer)
    ].copy()
    rep_binned = binned[
        (binned["subset"] == MAIN_SUBSET) & (binned["site_id"] == rep_site) & (binned["layer"] == rep_layer)
    ].copy()

    aligned_path = SEGCSR / "segmented_csr_aligned_points.parquet"
    rep_points = pd.read_parquet(aligned_path)
    rep_points = rep_points[
        (rep_points["subset"] == MAIN_SUBSET)
        & (rep_points["site_id"] == rep_site)
        & (rep_points["layer"] == rep_layer)
        & (rep_points["segment"].isin(SEGMENT_ORDER))
    ].copy()

    tables = {
        "experiment3_csr_entry_decisions": entry,
        "experiment3_main_by_layer_segment_summary": summary,
        "experiment3_location_layer_csr_availability": availability,
        "experiment3_location_layer_segment_metrics": local_main,
        "experiment3_representative_site_candidates": rep_candidates,
        "experiment3_representative_location_segment_metrics": rep_metrics,
        "experiment3_representative_location_curves": rep_curves,
        "experiment3_representative_location_binned_points": rep_binned,
        "experiment3_representative_location_aligned_points": rep_points,
    }
    for name, table in tables.items():
        table.to_csv(SOURCE / f"{name}.csv", index=False)
    return tables, rep_site, rep_layer


def plot_representative_location(
    curves: pd.DataFrame,
    binned: pd.DataFrame,
    points: pd.DataFrame,
    metrics: pd.DataFrame,
    rep_site: int,
    rep_layer: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    metric_lookup = metrics.set_index("segment").to_dict("index")
    handles = []
    labels = []

    for idx, segment in enumerate(SEGMENT_ORDER):
        ax = axes[idx]
        curve = curves[curves["segment"] == segment].sort_values("csr_x_h")
        bins = binned[binned["segment"] == segment].sort_values("csr_x_h")
        panel_points = points[points["segment"] == segment].sort_values("csr_x_h")

        if not panel_points.empty:
            max_points = 900
            if len(panel_points) > max_points:
                panel_points = panel_points.sample(max_points, random_state=20260605)
            sc = ax.scatter(
                panel_points["csr_x_h"],
                panel_points["moisture_mm"],
                s=7,
                color=COLORS["point"],
                alpha=0.18,
                linewidths=0,
                rasterized=True,
                label="Aligned observations",
            )
            if idx == 0:
                handles.append(sc)
                labels.append("Aligned observations")

        if not bins.empty:
            fill = ax.fill_between(
                bins["csr_x_h"].to_numpy(dtype=float),
                bins["q25_mm"].to_numpy(dtype=float),
                bins["q75_mm"].to_numpy(dtype=float),
                color=SEGMENT_COLORS[segment],
                alpha=0.16,
                linewidth=0,
                label="Binned IQR",
            )
            med = ax.scatter(
                bins["csr_x_h"],
                bins["moisture_mm"],
                s=10,
                color=SEGMENT_COLORS[segment],
                alpha=0.65,
                linewidths=0,
                label="Binned median",
            )
            if idx == 0:
                handles.extend([fill, med])
                labels.extend(["Binned IQR", "Binned median"])

        if not curve.empty:
            line = ax.plot(
                curve["csr_x_h"],
                curve["csr_mm"],
                color=COLORS["curve"],
                lw=1.8,
                label="Local segmented CSR",
            )[0]
            if idx == 0:
                handles.append(line)
                labels.append("Local segmented CSR")

        m = metric_lookup.get(segment, {})
        ax.text(
            0.03,
            0.04,
            f"events = {int(m.get('events', 0))}\nRMSE = {float(m.get('rmse_mm', np.nan)):.2f} mm",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
            color=COLORS["neutral_dark"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["neutral_light"], linewidth=0.6),
        )
        ax.set_title(SEGMENT_LABELS[segment], color=SEGMENT_COLORS[segment], fontsize=7.3)
        ax.set_xlabel("Segment coordinate (h)")
        if idx == 0:
            ax.set_ylabel("Soil water amount (mm)")
        ax.grid(axis="y", color=COLORS["neutral_light"], lw=0.6)
        add_panel_label(ax, chr(ord("a") + idx))

    fig.suptitle(
        f"Representative localized segmented CSR: site {rep_site}, {LAYER_LABELS[rep_layer]} layer",
        x=0.5,
        y=1.05,
        fontsize=8.2,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    save_pub(fig, "fig_experiment3_representative_location_segmented_csr")
    plt.close(fig)


def plot_full_station_summary(summary: pd.DataFrame, availability: pd.DataFrame) -> None:
    summary = summary.copy()
    availability = availability.copy()

    fig = plt.figure(figsize=(7.2, 5.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55], hspace=0.42, wspace=0.34)
    ax_count = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    x = np.arange(len(LAYER_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(SEGMENT_ORDER))

    for offset, segment in zip(offsets, SEGMENT_ORDER):
        panel = summary[summary["segment"] == segment].set_index("layer").reindex(LAYER_ORDER)
        bars = ax_count.bar(
            x + offset,
            panel["local_models"],
            width=width,
            color=SEGMENT_COLORS[segment],
            alpha=0.9,
            label=SEGMENT_SHORT[segment],
        )
        for bar, value in zip(bars, panel["local_models"]):
            if value > 0:
                ax_count.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color=COLORS["neutral_dark"],
                )

        rmse = panel["median_rmse_mm"].to_numpy(dtype=float)
        ax_rmse.plot(
            x + offset,
            rmse,
            marker="o",
            ms=4.2,
            lw=1.2,
            color=SEGMENT_COLORS[segment],
            label=SEGMENT_SHORT[segment],
        )

    ax_count.set_xticks(x)
    ax_count.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_count.set_ylabel("Local models")
    ax_count.set_title("Fitted location-layer-segment models")
    ax_count.set_ylim(0, max(34, summary["local_models"].max() + 5))
    ax_count.grid(axis="y", color=COLORS["neutral_light"], lw=0.6)
    ax_count.legend(loc="upper right", fontsize=6.2, handlelength=1.2)
    add_panel_label(ax_count, "a")

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_rmse.set_ylabel("Median local RMSE (mm)")
    ax_rmse.set_title("Within-local-model error")
    ax_rmse.set_ylim(0, max(1.15, np.nanmax(summary["median_rmse_mm"].to_numpy(dtype=float)) + 0.15))
    ax_rmse.grid(axis="y", color=COLORS["neutral_light"], lw=0.6)
    add_panel_label(ax_rmse, "b")

    heat = availability.pivot_table(
        index="site_id", columns="layer", values="fitted_segments", aggfunc="median", observed=False
    )
    sort_frame = availability[availability["layer"] == "moisture_4in"][["site_id", "clean_stageII_48h_events", "fitted_segments"]]
    sort_sites = sort_frame.sort_values(["fitted_segments", "clean_stageII_48h_events", "site_id"], ascending=[False, False, True])[
        "site_id"
    ].tolist()
    heat = heat.reindex(index=sort_sites, columns=LAYER_ORDER).fillna(0)

    cmap = ListedColormap(["#F3F3F3", "#D4E3F1", "#8AB6D6", "#355C7D"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax_heat.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
    ax_heat.set_xticks(np.arange(len(LAYER_ORDER)))
    ax_heat.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels([str(int(site)) for site in heat.index], fontsize=4.8)
    ax_heat.set_xlabel("Layer")
    ax_heat.set_ylabel("FAWN site ID")
    ax_heat.set_title("Localized segmented CSR availability by location and layer")
    ax_heat.tick_params(axis="both", length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.020, pad=0.012, ticks=[0, 1, 2, 3])
    cbar.set_label("Fitted segments")
    add_panel_label(ax_heat, "c", x=-0.055, y=1.04)

    save_pub(fig, "fig_experiment3_full_station_segmented_csr_summary")
    plt.close(fig)


def format_int(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def format_float(value: float | int | None, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_summary_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Layer | Segment | Use | Local models | Median events/model | Total points | Median RMSE mm | Median MAE mm | Median bias mm |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        if row.local_models == 0:
            continue
        lines.append(
            f"| {row.layer_label} | {row.segment_label} | {row.recommended_use} | "
            f"{format_int(row.local_models)} | {format_float(row.median_events_per_model, 1)} | "
            f"{format_int(row.total_points)} | {format_float(row.median_rmse_mm)} | "
            f"{format_float(row.median_mae_mm)} | {format_float(row.median_bias_mm)} |"
        )
    return lines


def write_report(tables: dict[str, pd.DataFrame], rep_site: int, rep_layer: str, subset_counts: pd.DataFrame) -> None:
    summary = tables["experiment3_main_by_layer_segment_summary"]
    availability = tables["experiment3_location_layer_csr_availability"]
    rep_metrics = tables["experiment3_representative_location_segment_metrics"]
    rep_candidates = tables["experiment3_representative_site_candidates"]

    main_count_row = subset_counts[subset_counts["subset"] == MAIN_SUBSET]
    main_events = int(main_count_row["events_before_segment_filter"].iloc[0]) if not main_count_row.empty else 0
    main_sites = int(main_count_row["sites_before_segment_filter"].iloc[0]) if not main_count_row.empty else 0
    complete_4in = int(
        ((availability["layer"] == "moisture_4in") & (availability["fitted_segments"] == 3)).sum()
    )
    complete_8in = int(
        ((availability["layer"] == "moisture_8in") & (availability["fitted_segments"] == 3)).sum()
    )

    best = rep_candidates.iloc[0]
    lines = [
        "# Experiment 3: Localized segmented CSR entry set and full-station summary",
        "",
        "Prepared: 2026-06-05",
        "",
        "## Figure contract",
        "",
        "Core conclusion: localized segmented CSR should be constructed from rainfall-validated, non-interrupted, stage-II-like drydown events, then fitted separately by location, layer, and segment.",
        "",
        "Evidence chain: the entry table defines the eligible event and segment pool; the representative-site figure shows how one location-layer curve is built; the full-station summary figure and table show where the method is supported across the FAWN network.",
        "",
        "Archetype: quantitative grid with one representative local example plus network-level coverage and fit-quality panels.",
        "",
        "Export contract: Python/matplotlib only; SVG/PDF with editable text, PNG preview, and 600 dpi TIFF are exported with source-data CSVs.",
        "",
        "## CSR entry decision",
        "",
        "Main localized segmented CSR uses `clean_stageII_48h`: rainfall-associated within 48 h, not interrupted by rain, and classified as stage-II-like by the loss-function diagnosis.",
        "",
        "The early 0-3 h portion is kept as a separate transient segment. The post-3 h mid-storage segment is the primary storage-dependent CSR segment, and the post-3 h low-storage tail is retained with cautious interpretation.",
        "",
        f"The local fitting threshold is at least {MIN_LOCAL_SEGMENT_EVENTS} events and {MIN_LOCAL_SEGMENT_POINTS} aligned observations for each location-layer-segment.",
        "",
        "## Representative location-layer example",
        "",
        f"Selected example: FAWN site {rep_site}, {LAYER_LABELS[rep_layer]} layer.",
        "",
        "The selected site is a complete, non-405 representative candidate: all three segments are fitted, event support is not extreme, and the median RMSE is close to the network median for complete 4 in candidates.",
        "",
        f"Selection score rank: {int(best['rank'])}; total segment-events: {int(best['events_sum'])}; minimum segment-events: {int(best['events_min'])}; median RMSE: {best['rmse_median_mm']:.3f} mm.",
        "",
        "| Segment | Events | Points | CCC | RMSE mm | MAE mm | Bias mm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in segment_sort(rep_metrics).itertuples(index=False):
        lines.append(
            f"| {row.segment_label} | {format_int(row.events)} | {format_int(row.points)} | "
            f"{format_float(row.ccc)} | {format_float(row.rmse_mm)} | {format_float(row.mae_mm)} | {format_float(row.bias_mm)} |"
        )

    lines.extend(
        [
            "",
            "## Full-station summary",
            "",
            f"The main event pool contains {main_events:,} clean stage-II-like events across {main_sites:,} FAWN sites before segment-level support filtering.",
            "",
            f"Complete three-segment local CSR coverage is strongest at 4 in ({complete_4in} location-layers) and secondary at 8 in ({complete_8in} location-layers). Deeper layers are retained in the table but should be described as exploratory because only a few location-layers meet the fitting threshold.",
            "",
        ]
    )
    lines.extend(markdown_summary_table(summary))
    lines.extend(
        [
            "",
            "## Recommended manuscript use",
            "",
            "- Main example figure: use the representative site-layer figure to show the localized segmented CSR construction.",
            "- Main network summary: report 4 in as the strongest full-network layer; report 8 in as secondary-supported.",
            "- Table or Supplementary Table: include all fitted layer-segment combinations, including sparse 12-20 in results, but avoid strong mechanistic claims from sparse deeper layers.",
            "- Sensitivity statement: all-events and clean-only variants can be cited as robustness checks, but they should not define the main CSR curve.",
            "",
            "## Output files",
            "",
            "- `fig_experiment3_representative_location_segmented_csr.svg/pdf/png/tiff`",
            "- `fig_experiment3_full_station_segmented_csr_summary.svg/pdf/png/tiff`",
            "- `source_data/experiment3_csr_entry_decisions.csv`",
            "- `source_data/experiment3_main_by_layer_segment_summary.csv`",
            "- `source_data/experiment3_location_layer_csr_availability.csv`",
            "- `source_data/experiment3_location_layer_segment_metrics.csv`",
            "- `source_data/experiment3_representative_site_candidates.csv`",
            "- `source_data/experiment3_representative_location_*.csv`",
            "",
        ]
    )

    (OUT / "experiment3_localized_segmented_csr_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out()
    frames = read_inputs()
    tables, rep_site, rep_layer = build_source_tables(frames)
    plot_representative_location(
        tables["experiment3_representative_location_curves"],
        tables["experiment3_representative_location_binned_points"],
        tables["experiment3_representative_location_aligned_points"],
        tables["experiment3_representative_location_segment_metrics"],
        rep_site,
        rep_layer,
    )
    plot_full_station_summary(
        tables["experiment3_main_by_layer_segment_summary"],
        tables["experiment3_location_layer_csr_availability"],
    )
    write_report(tables, rep_site, rep_layer, frames["subset_counts"])
    print(f"Experiment 3 outputs written to {OUT}")
    print(f"Representative location-layer: site {rep_site}, {rep_layer}")


if __name__ == "__main__":
    main()


