from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from fawn_full_smde_audit import normalize_moisture_pct


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
AUDIT = ANALYSIS / "fawn_full_smde_audit"
DATA = ANALYSIS / "fawn_db_export" / "data"
OUT = ANALYSIS / "experiment1_smde_detection_audit"
SOURCE = OUT / "source_data"
SUBMISSION_FIGURES = ROOT / "figures"

YEARS = [2023, 2024, 2025]
EVENTS_FILE = AUDIT / "full_smde_event_audit.csv"
SITE_LAYER_FILE = AUDIT / "full_smde_summary_by_site_layer.csv"
LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
MOISTURE_RENAME = {
    "moisture_sms_4_inch_pct": "moisture_4in",
    "moisture_sms_8_inch_pct": "moisture_8in",
    "moisture_sms_12_inch_pct": "moisture_12in",
    "moisture_sms_16_inch_pct": "moisture_16in",
    "moisture_sms_20_inch_pct": "moisture_20in",
}
PCT_TO_MM_4IN = 4.0 * 25.4 / 100.0

AUDIT_CLASS_ORDER = [
    "rain-associated clean <=24h",
    "rain-associated clean 24-48h",
    "rain-associated but interrupted",
    "not associated within 48h",
]
AUDIT_LABELS = {
    "rain-associated clean <=24h": "Clean <=24 h",
    "rain-associated clean 24-48h": "Clean 24-48 h",
    "rain-associated but interrupted": "Interrupted",
    "not associated within 48h": "No rain <=48 h",
}
COLORS = {
    "rain-associated clean <=24h": "#355C7D",
    "rain-associated clean 24-48h": "#6C9BC6",
    "rain-associated but interrupted": "#C78D4B",
    "not associated within 48h": "#B7B7B7",
    "soil": "#2F3A42",
    "rain": "#5B8DB8",
    "accent": "#A64B56",
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
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
    SUBMISSION_FIGURES.mkdir(parents=True, exist_ok=True)


def save_pub(fig: plt.Figure, stem: str) -> None:
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    shutil.copyfile(OUT / f"{stem}.pdf", SUBMISSION_FIGURES / "Figure_4_SMDE_detection_audit.pdf")


def layer_sort(frame: pd.DataFrame, column: str = "layer") -> pd.DataFrame:
    out = frame.copy()
    out[column] = pd.Categorical(out[column], categories=LAYER_ORDER, ordered=True)
    return out.sort_values(column)


def read_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])
    events["layer"] = pd.Categorical(events["layer"], categories=LAYER_ORDER, ordered=True)
    return events.sort_values(["layer", "site_id", "start"])


def read_station_data(site_id: int) -> pd.DataFrame:
    soil_frames = []
    wx_frames = []
    for year in YEARS:
        soil_path = DATA / f"soil_moisture_{year}.parquet"
        wx_path = DATA / f"wx_selected_{year}.parquet"
        soil = pd.read_parquet(soil_path, filters=[("ID", "==", int(site_id))])
        wx = pd.read_parquet(wx_path, filters=[("ID", "==", int(site_id))])
        if not soil.empty:
            soil_frames.append(soil)
        if not wx.empty:
            wx_frames.append(wx)

    if not soil_frames:
        raise RuntimeError(f"No soil moisture rows found for site {site_id}.")
    soil = pd.concat(soil_frames, ignore_index=True)
    soil["UTC"] = pd.to_datetime(soil["UTC"], errors="coerce")
    soil = soil.dropna(subset=["UTC"]).sort_values("UTC").drop_duplicates("UTC", keep="last")
    for source, target in MOISTURE_RENAME.items():
        soil[target] = normalize_moisture_pct(soil[source]) * PCT_TO_MM_4IN
    soil = soil.set_index("UTC")[list(MOISTURE_RENAME.values())]

    if wx_frames:
        wx = pd.concat(wx_frames, ignore_index=True)
        wx["UTC"] = pd.to_datetime(wx["UTC"], errors="coerce")
        wx = wx.dropna(subset=["UTC"]).sort_values("UTC")
        rain = pd.to_numeric(wx["rain_2m_inches"], errors="coerce")
        backup = pd.to_numeric(wx["rain_backup_2m_inches"], errors="coerce")
        wx["Rain"] = rain.where(rain.notna(), backup).fillna(0.0).clip(lower=0.0)
        wx = wx.groupby("UTC", as_index=True)["Rain"].sum().sort_index().to_frame()
    else:
        wx = pd.DataFrame(index=soil.index, data={"Rain": 0.0})

    merged = soil.join(wx, how="left")
    merged["Rain"] = merged["Rain"].fillna(0.0)
    return merged


def pick_representative_event(events: pd.DataFrame) -> pd.Series:
    pool = events[
        (events["site_id"] != 405)
        & (events["layer"] == "moisture_4in")
        & (events["audit_class"] == "rain-associated clean <=24h")
        & (~events["interrupted_by_rain"])
        & (events["trim3_r2"] >= 0.8)
        & (events["duration_h"].between(6.0, 24.0))
        & (events["total_drop_mm"].between(1.5, 6.0))
        & (events["start_mm"].between(8.0, 14.5))
        & (events["rain_lag_h"].between(0.10, 24.0))
    ].copy()
    if pool.empty:
        pool = events[(events["layer"] == "moisture_4in") & (~events["interrupted_by_rain"])].copy()
    pool["score"] = (
        pool["total_drop_mm"] * 1.8
        + pool["trim3_r2"].fillna(0.0) * 0.8
        - (pool["duration_h"] - 12.0).abs() * 0.05
        - pool["peak_start_shift_h"].fillna(0.0).abs() * 0.02
    )
    return pool.sort_values("score", ascending=False).iloc[0]


def build_source_tables(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    audit_counts = (
        events.groupby(["layer", "audit_class"], observed=False)
        .size()
        .reset_index(name="events")
        .query("events > 0")
    )
    audit_counts["layer_label"] = audit_counts["layer"].astype(str).map(LAYER_LABELS)
    audit_counts["audit_label"] = audit_counts["audit_class"].map(AUDIT_LABELS)
    layer_totals = audit_counts.groupby("layer", observed=False)["events"].transform("sum")
    audit_counts["fraction_of_layer"] = audit_counts["events"] / layer_totals
    audit_counts = layer_sort(audit_counts)

    total = len(events)
    funnel_rows = [
        ("Detected SMDEs", total),
        ("Rain <=48 h", int(events["associated_48h"].sum())),
        ("Clean rain <=48 h", int((events["associated_48h"] & ~events["interrupted_by_rain"]).sum())),
        (
            "Clean + regime label",
            int(
                (
                    events["associated_48h"]
                    & ~events["interrupted_by_rain"]
                    & events["regime_proxy"].isin(["stage-II-like", "stage-I-like", "early-transient-heavy"])
                ).sum()
            ),
        ),
    ]
    funnel = pd.DataFrame(funnel_rows, columns=["step", "events"])
    funnel["fraction_of_detected"] = funnel["events"] / total

    site_layer = pd.read_csv(SITE_LAYER_FILE)
    site_layer["layer"] = pd.Categorical(site_layer["layer"], categories=LAYER_ORDER, ordered=True)
    site_layer["layer_label"] = site_layer["layer"].astype(str).map(LAYER_LABELS)
    site_layer = site_layer.sort_values(["site_id", "layer"])

    site_heat = site_layer[["site_id", "layer", "layer_label", "events", "clean_48h_rate"]].copy()

    layer_summary = (
        events.groupby("layer", observed=False)
        .agg(
            events=("event_id", "count"),
            associated_48h=("associated_48h", "sum"),
            clean_48h=("interrupted_by_rain", lambda x: int((~x).sum())),
        )
        .reset_index()
    )
    layer_summary["clean_48h"] = [
        int(((events["layer"] == row.layer) & events["associated_48h"] & ~events["interrupted_by_rain"]).sum())
        for row in layer_summary.itertuples()
    ]
    layer_summary["stageII_clean_48h"] = [
        int(
            (
                (events["layer"] == row.layer)
                & events["associated_48h"]
                & ~events["interrupted_by_rain"]
                & (events["regime_proxy"] == "stage-II-like")
            ).sum()
        )
        for row in layer_summary.itertuples()
    ]
    layer_summary["interpretable_clean_48h"] = [
        int(
            (
                (events["layer"] == row.layer)
                & events["associated_48h"]
                & ~events["interrupted_by_rain"]
                & events["regime_proxy"].isin(["stage-II-like", "stage-I-like", "early-transient-heavy"])
            ).sum()
        )
        for row in layer_summary.itertuples()
    ]
    layer_summary["associated_48h_rate"] = layer_summary["associated_48h"] / layer_summary["events"]
    layer_summary["clean_48h_rate"] = layer_summary["clean_48h"] / layer_summary["events"]
    layer_summary["stageII_clean_48h_rate"] = layer_summary["stageII_clean_48h"] / layer_summary["events"]
    layer_summary["interpretable_clean_48h_rate"] = layer_summary["interpretable_clean_48h"] / layer_summary["events"]
    layer_summary["layer_label"] = layer_summary["layer"].astype(str).map(LAYER_LABELS)
    layer_summary = layer_sort(layer_summary)

    tables = {
        "experiment1_audit_class_by_layer": audit_counts,
        "experiment1_detection_funnel": funnel,
        "experiment1_site_layer_clean_rate": site_heat,
        "experiment1_layer_summary": layer_summary,
    }
    for name, table in tables.items():
        table.to_csv(SOURCE / f"{name}.csv", index=False)
    return tables


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )


def plot_representative_event(ax: plt.Axes, events: pd.DataFrame) -> pd.Series:
    event = pick_representative_event(events)
    site_id = int(event["site_id"])
    layer = str(event["layer"])
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])
    merged = read_station_data(site_id)
    display_start = start
    window = merged.loc[display_start - pd.Timedelta(hours=10) : end + pd.Timedelta(hours=3), [layer, "Rain"]].copy()
    window["elapsed_h"] = (window.index - display_start).total_seconds() / 3600.0
    event_end_h = (end - display_start).total_seconds() / 3600.0

    ax.plot(window["elapsed_h"], window[layer], color=COLORS["soil"], lw=1.4)
    ax.axvspan(0, event_end_h, color="#DDE8F1", alpha=0.7, lw=0)
    ax.axvline(0, color=COLORS["accent"], lw=1.0)
    ax.axvline(event_end_h, color=COLORS["neutral_mid"], lw=1.0, ls="--")
    ax.scatter([0], [float(window.loc[display_start, layer]) if display_start in window.index else event["start_mm"]], s=20, color=COLORS["accent"], zorder=5)
    ax.scatter(
        [event_end_h],
        [event["end_mm"]],
        s=20,
        color=COLORS["neutral_mid"],
        zorder=5,
    )
    ax.set_xlabel("Hours from SMDE start")
    ax.set_ylabel("Soil water amount (mm)")
    ax.set_ylim(0, 15)
    ax.set_title(f"Representative soil moisture-only detection, site {site_id}, {LAYER_LABELS[layer]}", loc="left", fontsize=8)

    ax_r = ax.twinx()
    rain_mm = (window["Rain"] * 25.4).where(window.index <= display_start, 0.0)
    bar_width_h = 0.18
    ax_r.bar(window["elapsed_h"], rain_mm, width=bar_width_h, color=COLORS["rain"], alpha=0.35, linewidth=0)
    ax_r.set_ylabel("")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.set_ylim(0, max(1.0, float(rain_mm.max()) * 3.5))
    ax.text(
        0.98,
        0.94,
        "blue bars: pre-SMDE rainfall",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=COLORS["rain"],
    )
    add_panel_label(ax, "a")
    return event


def plot_stacked_audit(ax: plt.Axes, audit_counts: pd.DataFrame) -> None:
    pivot = (
        audit_counts.pivot(index="layer", columns="audit_class", values="events")
        .reindex(LAYER_ORDER)
        .reindex(columns=AUDIT_CLASS_ORDER)
        .fillna(0)
    )
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    totals = pivot.sum(axis=1).to_numpy(dtype=float)
    for audit_class in AUDIT_CLASS_ORDER:
        vals = pivot[audit_class].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.68,
            color=COLORS[audit_class],
            edgecolor="white",
            linewidth=0.4,
            label=AUDIT_LABELS[audit_class],
        )
        for xi, val, bot, total in zip(x, vals, bottom, totals):
            if total <= 0:
                continue
            frac = val / total
            force_small_label = audit_class == "rain-associated but interrupted" and val > 0
            if frac >= 0.065 or force_small_label:
                text_color = "white" if audit_class in AUDIT_CLASS_ORDER[:2] else COLORS["neutral_dark"]
                ax.text(
                    xi,
                    bot + val / 2,
                    f"{frac:.0%}",
                    ha="center",
                    va="center",
                    fontsize=4.8 if force_small_label and frac < 0.065 else 5.8,
                    color=text_color,
                )
        bottom += vals
    ax.set_xticks(x, [LAYER_LABELS[str(layer)] for layer in pivot.index])
    ax.set_ylabel("Detected SMDEs")
    ax.set_title("Post-detection rainfall validation classes", loc="left", fontsize=8)
    ax.legend(ncol=2, fontsize=6.3, handlelength=1.2, columnspacing=0.9, loc="upper right")
    add_panel_label(ax, "b")


def plot_funnel(ax: plt.Axes, funnel: pd.DataFrame) -> None:
    y = np.arange(len(funnel))
    widths = funnel["fraction_of_detected"].to_numpy()
    counts = funnel["events"].to_numpy()
    colors = ["#303030", "#5B8DB8", "#355C7D", "#A64B56"]
    ax.barh(y, widths, color=colors, height=0.58)
    ax.set_yticks(y, funnel["step"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Fraction of detected SMDEs")
    ax.set_title("Detection-to-regime-audit funnel", loc="left", fontsize=8)
    for yi, width, count in zip(y, widths, counts):
        if width >= 0.94:
            ax.text(
                width - 0.02,
                yi,
                f"{count:,} ({width:.1%})",
                va="center",
                ha="right",
                fontsize=6.6,
                color="white",
            )
            continue
        ax.text(
            min(width + 0.025, 1.0),
            yi,
            f"{count:,} ({width:.1%})",
            va="center",
            ha="left" if width < 0.86 else "right",
            fontsize=6.6,
            color=COLORS["neutral_dark"],
        )
    add_panel_label(ax, "c")


def plot_site_layer_heatmap(ax: plt.Axes, site_heat: pd.DataFrame) -> None:
    clean = site_heat.copy()
    clean["site_id"] = clean["site_id"].astype(int)
    heat = clean.pivot_table(index="site_id", columns="layer", values="clean_48h_rate", observed=False)
    heat = heat.reindex(columns=LAYER_ORDER)
    event_counts = clean.pivot_table(index="site_id", columns="layer", values="events", observed=False).reindex(columns=LAYER_ORDER)
    complete_sites = heat.dropna(how="any").index
    heat = heat.loc[complete_sites]
    event_counts = event_counts.loc[complete_sites]

    arr = heat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(LAYER_ORDER)), [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    y_labels = [str(int(s)) for s in heat.index]
    ax.set_yticks(np.arange(len(heat.index)), y_labels, fontsize=4.9)
    ax.set_xlabel("Sensor depth")
    ax.set_ylabel("FAWN station")
    ax.set_title("Clean rainfall-associated rate by complete station-layer rows", loc="left", fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(LAYER_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.42)
    ax.tick_params(which="minor", bottom=False, left=False)

    event_count_arr = event_counts.to_numpy(dtype=float)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = arr[i, j]
            n = event_count_arr[i, j]
            if np.isfinite(value):
                color = "white" if value > 0.58 else COLORS["neutral_dark"]
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=4.1, color=color)
            elif not np.isfinite(value):
                ax.text(j, i, "-", ha="center", va="center", fontsize=4.4, color="#AAAAAA")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.030)
    cbar.set_label("Clean <=48 h rate")
    add_panel_label(ax, "d")


def write_report(events: pd.DataFrame, tables: dict[str, pd.DataFrame], representative: pd.Series) -> None:
    funnel = tables["experiment1_detection_funnel"]
    layer_summary = tables["experiment1_layer_summary"]
    report = [
        "# Experiment 1: SMDE Detection Audit",
        "",
        "Prepared: 2026-06-05",
        "",
        "## Purpose",
        "",
        "This experiment tests whether soil moisture-only detection produces a credible soil moisture drying event (SMDE) library, and then uses FAWN rainfall records only as an independent post-detection validation layer.",
        "",
        "## Core result",
        "",
        f"- Detected SMDEs: {len(events):,}.",
        f"- Associated with rainfall within 48 h: {int(funnel.loc[funnel['step'] == 'Rain <=48 h', 'events'].iloc[0]):,} ({float(funnel.loc[funnel['step'] == 'Rain <=48 h', 'fraction_of_detected'].iloc[0]):.1%}).",
        f"- Clean rainfall-associated within 48 h: {int(funnel.loc[funnel['step'] == 'Clean rain <=48 h', 'events'].iloc[0]):,} ({float(funnel.loc[funnel['step'] == 'Clean rain <=48 h', 'fraction_of_detected'].iloc[0]):.1%}).",
        f"- Clean rainfall-associated and assigned to an interpretable loss-function regime: {int(funnel.loc[funnel['step'] == 'Clean + regime label', 'events'].iloc[0]):,} ({float(funnel.loc[funnel['step'] == 'Clean + regime label', 'fraction_of_detected'].iloc[0]):.1%}).",
        "",
        "## Layer summary",
        "",
        "| Layer | Detected | Rain <=48 h | Clean <=48 h | Clean + regime label |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in layer_summary.itertuples(index=False):
        report.append(
            f"| {row.layer_label} | {row.events:,} | {row.associated_48h_rate:.1%} | {row.clean_48h_rate:.1%} | {row.interpretable_clean_48h_rate:.1%} |"
        )
    report.extend(
        [
            "",
            "## Representative event used in panel a",
            "",
            f"- Site: {int(representative['site_id'])}.",
            f"- Layer: {LAYER_LABELS[str(representative['layer'])]}.",
            f"- Start: {pd.Timestamp(representative['start'])}.",
            f"- End: {pd.Timestamp(representative['end'])}.",
            f"- Duration: {float(representative['duration_h']):.2f} h.",
            f"- Total drop: {float(representative['total_drop_mm']):.2f} mm.",
            f"- Rain lag: {float(representative['rain_lag_h']):.2f} h.",
            "",
        "## Figure files",
        "",
        "- `fig_experiment1_smde_detection_audit.svg`",
        "- `fig_experiment1_smde_detection_audit.pdf`",
        "- `fig_experiment1_smde_detection_audit.tiff`",
        "- `fig_experiment1_smde_detection_audit.png`",
        "",
        "## Draft figure legend",
        "",
        "Figure X. Soil moisture-only SMDE detection and independent precipitation validation across FAWN stations. (a) Representative 4-inch station-layer time series showing a detected SMDE window from soil moisture alone; rainfall bars are shown only as post-detection context. (b) Depth-wise counts of detected SMDEs grouped by post-detection rainfall-validation class. (c) Detection funnel from all soil moisture-only SMDEs to rainfall-associated, clean rainfall-associated, and clean rainfall-associated events assigned to an interpretable loss-function regime. (d) Station-layer heatmap showing the fraction of detected events that were rainfall-associated within 48 h and not interrupted by rain during the event; blank cells indicate no detected events or insufficient data for that station-layer.",
        "",
        "## Interpretation boundary",
            "",
            "Rainfall is not used to detect SMDEs. It is used after detection to estimate which soil moisture drying patterns are plausibly post-input and which are interrupted or not associated with measured rainfall.",
        ]
    )
    (OUT / "experiment1_smde_detection_audit_report.md").write_text("\n".join(report), encoding="utf-8")


def build_figure() -> None:
    ensure_out()
    events = read_events()
    tables = build_source_tables(events)

    fig = plt.figure(figsize=(7.8, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 1.0],
        height_ratios=[0.9, 1.1],
        wspace=0.48,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    representative = plot_representative_event(ax_a, events)
    plot_stacked_audit(ax_b, tables["experiment1_audit_class_by_layer"])
    plot_funnel(ax_c, tables["experiment1_detection_funnel"])
    plot_site_layer_heatmap(ax_d, tables["experiment1_site_layer_clean_rate"])

    fig.suptitle(
        "Soil moisture-only SMDE detection with independent rainfall validation",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )
    fig.text(
        0.01,
        0.955,
        "FAWN 2023-2025; rainfall is used after detection, not as an event-detection input.",
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["neutral_mid"],
    )
    fig.subplots_adjust(top=0.89)
    save_pub(fig, "fig_experiment1_smde_detection_audit")
    plt.close(fig)
    write_report(events, tables, representative)


if __name__ == "__main__":
    build_figure()
