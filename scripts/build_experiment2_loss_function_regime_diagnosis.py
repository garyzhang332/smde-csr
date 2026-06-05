from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT
AUDIT = ANALYSIS / "fawn_full_smde_audit"
OUT = ANALYSIS / "experiment2_loss_function_regime_diagnosis"
SOURCE = OUT / "source_data"

EVENTS_FILE = AUDIT / "full_smde_event_audit.csv"
BINNED_FILE = AUDIT / "full_smde_binned_loss_by_layer_regime.csv"

LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
REGIME_ORDER = ["stage-II-like", "stage-I-like", "early-transient-heavy", "mixed_or_uncertain"]
REGIME_LABELS = {
    "stage-II-like": "Stage-II-like",
    "stage-I-like": "Stage-I-like",
    "early-transient-heavy": "Early transient",
    "mixed_or_uncertain": "Mixed/uncertain",
}
REGIME_COLORS = {
    "stage-II-like": "#355C7D",
    "stage-I-like": "#6FA38A",
    "early-transient-heavy": "#C78D4B",
    "mixed_or_uncertain": "#B7B7B7",
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
    "accent": "#A64B56",
}

MIN_LOCAL_SEGMENT_EVENTS = 8
STRONG_SUPPORT_EVENTS = 30

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


def read_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])
    events["layer"] = pd.Categorical(events["layer"], categories=LAYER_ORDER, ordered=True)
    events["regime_proxy"] = pd.Categorical(events["regime_proxy"], categories=REGIME_ORDER, ordered=True)
    events["clean_48h"] = events["associated_48h"] & ~events["interrupted_by_rain"]
    events["clean_stageII_48h"] = events["clean_48h"] & (events["regime_proxy"].astype(str) == "stage-II-like")
    return events.sort_values(["layer", "site_id", "start"])


def read_binned() -> pd.DataFrame:
    binned = pd.read_csv(BINNED_FILE)
    binned["layer"] = pd.Categorical(binned["layer"], categories=LAYER_ORDER, ordered=True)
    binned["regime_proxy"] = pd.Categorical(binned["regime_proxy"], categories=REGIME_ORDER, ordered=True)
    return binned.sort_values(["layer", "regime_proxy", "storage_norm"])


def build_source_tables(events: pd.DataFrame, binned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    regime_counts = (
        events.groupby(["layer", "regime_proxy"], observed=False)
        .size()
        .reset_index(name="events")
        .query("events > 0")
    )
    regime_counts["layer_label"] = regime_counts["layer"].astype(str).map(LAYER_LABELS)
    regime_counts["regime_label"] = regime_counts["regime_proxy"].astype(str).map(REGIME_LABELS)
    regime_counts["fraction_of_layer"] = regime_counts["events"] / regime_counts.groupby("layer", observed=False)["events"].transform("sum")

    diagnostic_summary = (
        events.groupby(["layer", "regime_proxy"], observed=False)
        .agg(
            events=("event_id", "count"),
            median_trim3_r2=("trim3_r2", "median"),
            median_early3_drop_share=("early3_drop_share", "median"),
            median_post3_loss_storage_corr=("post3_loss_storage_corr", "median"),
            median_post3_loss_cv=("post3_loss_cv", "median"),
            median_mean_loss_mm_h=("mean_loss_mm_h", "median"),
            median_vpd_kpa=("during_vp_def_2m_kPa_mean", "median"),
        )
        .reset_index()
        .query("events > 0")
    )
    diagnostic_summary["layer_label"] = diagnostic_summary["layer"].astype(str).map(LAYER_LABELS)
    diagnostic_summary["regime_label"] = diagnostic_summary["regime_proxy"].astype(str).map(REGIME_LABELS)

    eligible = (
        events[events["clean_stageII_48h"]]
        .groupby(["site_id", "layer"], observed=False)
        .size()
        .reset_index(name="clean_stageII_48h_events")
    )
    all_sites = sorted(events["site_id"].dropna().astype(int).unique())
    full_index = pd.MultiIndex.from_product([all_sites, LAYER_ORDER], names=["site_id", "layer"]).to_frame(index=False)
    eligible = full_index.merge(eligible, on=["site_id", "layer"], how="left")
    eligible["clean_stageII_48h_events"] = eligible["clean_stageII_48h_events"].fillna(0).astype(int)
    eligible["layer_label"] = eligible["layer"].astype(str).map(LAYER_LABELS)
    eligible["support_class"] = pd.cut(
        eligible["clean_stageII_48h_events"],
        bins=[-0.1, 0.1, MIN_LOCAL_SEGMENT_EVENTS - 0.1, STRONG_SUPPORT_EVENTS - 0.1, 10_000],
        labels=["none", "insufficient", "eligible", "strong"],
    )

    eligibility_by_layer = (
        eligible.groupby("layer", observed=False)
        .agg(
            station_layers=("site_id", "count"),
            stations_with_any=("clean_stageII_48h_events", lambda x: int((x > 0).sum())),
            eligible_station_layers=("clean_stageII_48h_events", lambda x: int((x >= MIN_LOCAL_SEGMENT_EVENTS).sum())),
            strong_station_layers=("clean_stageII_48h_events", lambda x: int((x >= STRONG_SUPPORT_EVENTS).sum())),
            total_clean_stageII_48h_events=("clean_stageII_48h_events", "sum"),
        )
        .reset_index()
    )
    eligibility_by_layer["layer_label"] = eligibility_by_layer["layer"].astype(str).map(LAYER_LABELS)

    empirical_loss_primary = binned[
        (binned["layer"].astype(str) == "moisture_4in")
        & (binned["regime_proxy"].astype(str).isin(["stage-II-like", "stage-I-like", "early-transient-heavy"]))
    ].copy()
    empirical_loss_primary["layer_label"] = empirical_loss_primary["layer"].astype(str).map(LAYER_LABELS)
    empirical_loss_primary["regime_label"] = empirical_loss_primary["regime_proxy"].astype(str).map(REGIME_LABELS)

    tables = {
        "experiment2_regime_composition_by_layer": regime_counts,
        "experiment2_event_diagnostic_summary": diagnostic_summary,
        "experiment2_csr_eligibility_by_station_layer": eligible,
        "experiment2_csr_eligibility_by_layer": eligibility_by_layer,
        "experiment2_empirical_loss_storage_4in": empirical_loss_primary,
    }
    for name, table in tables.items():
        table.to_csv(SOURCE / f"{name}.csv", index=False)
    return tables


def plot_conceptual_loss_function(ax: plt.Axes) -> None:
    x = np.linspace(0.02, 1.0, 300)
    # Low-storage stage-II: storage limitation; mid-storage stage-I: approximate plateau;
    # wet end: rapid transient drainage/runoff/redistribution.
    y_stage2 = 0.12 + 0.95 * (x / 0.42) ** 1.35
    y = np.piecewise(
        x,
        [x <= 0.42, (x > 0.42) & (x <= 0.76), x > 0.76],
        [
            lambda z: 0.12 + 0.95 * (z / 0.42) ** 1.35,
            lambda z: 1.06 + 0.03 * np.sin((z - 0.42) / 0.34 * np.pi),
            lambda z: 1.08 + 4.6 * (z - 0.76) ** 1.8,
        ],
    )
    ax.axvspan(0.02, 0.42, color="#DDE8F1", alpha=0.9, lw=0)
    ax.axvspan(0.42, 0.76, color="#E4EFE8", alpha=0.9, lw=0)
    ax.axvspan(0.76, 1.0, color="#F1E3D8", alpha=0.95, lw=0)
    ax.plot(x, y, color=COLORS["neutral_dark"], lw=1.7)
    ax.plot(x[x <= 0.42], y_stage2[x <= 0.42], color=REGIME_COLORS["stage-II-like"], lw=2.2)

    labels = [
        ("Stage-II-like\nloss rises with storage", 0.22, 0.86, REGIME_COLORS["stage-II-like"]),
        ("Stage-I-like\nstorage-invariant", 0.59, 1.28, REGIME_COLORS["stage-I-like"]),
        ("Early wet transient\nredistribution/drainage", 0.88, 1.68, REGIME_COLORS["early-transient-heavy"]),
    ]
    for text, xx, yy, color in labels:
        ax.text(xx, yy, text, ha="center", va="center", fontsize=6.8, color=color)

    ax.annotate(
        "loss function\nL = -dS/dt",
        xy=(0.33, 0.75),
        xytext=(0.09, 1.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["neutral_mid"], lw=0.8),
        fontsize=6.6,
        color=COLORS["neutral_mid"],
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.95)
    ax.annotate(
        "drydown over time\nmoves wet -> dry",
        xy=(0.72, 0.18),
        xytext=(0.38, 0.18),
        arrowprops=dict(arrowstyle="<-", color=COLORS["neutral_mid"], lw=0.8),
        fontsize=6.4,
        color=COLORS["neutral_mid"],
        ha="center",
        va="center",
    )
    ax.set_xlabel("Storage state, $x_s$ (dry 0 -> wet 1)")
    ax.set_ylabel("Loss rate, $L=-dS/dt$")
    ax.set_title("Conceptual loss rate versus storage", loc="left", fontsize=8)
    ax.set_xticks([0.1, 0.5, 0.9], ["dry", "mid", "wet"])
    ax.set_yticks([])
    add_panel_label(ax, "a")


def plot_empirical_loss_storage(ax: plt.Axes, empirical: pd.DataFrame) -> None:
    for regime in ["stage-II-like", "stage-I-like", "early-transient-heavy"]:
        panel = empirical[empirical["regime_proxy"].astype(str) == regime].sort_values("storage_norm")
        if panel.empty:
            continue
        x = panel["storage_norm"].to_numpy(dtype=float)
        y = panel["loss_mm_h_median"].to_numpy(dtype=float)
        y1 = panel["loss_mm_h_q25"].to_numpy(dtype=float)
        y2 = panel["loss_mm_h_q75"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            lw=1.8 if regime == "stage-II-like" else 1.3,
            color=REGIME_COLORS[regime],
            label=REGIME_LABELS[regime],
        )
        band_alpha = 0.08 if regime == "early-transient-heavy" else 0.12
        ax.fill_between(x, y1, y2, color=REGIME_COLORS[regime], alpha=band_alpha, lw=0)
    ax.annotate(
        "wetter event states",
        xy=(0.86, 0.06),
        xytext=(0.50, 0.06),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="->", color=COLORS["neutral_mid"], lw=0.8),
        fontsize=6.4,
        color=COLORS["neutral_mid"],
        ha="center",
        va="center",
    )
    ax.set_xlabel("Normalized event storage, $x_s$ (0=end/dry, 1=start/wet)")
    ax.set_ylabel("Median loss rate (mm h$^{-1}$)")
    ax.set_title("Empirical loss rate versus storage, 4 in layer", loc="left", fontsize=8)
    ax.legend(fontsize=6.2, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    add_panel_label(ax, "b")


def plot_regime_composition(ax: plt.Axes, regime_counts: pd.DataFrame) -> None:
    pivot = (
        regime_counts.pivot(index="layer", columns="regime_proxy", values="fraction_of_layer")
        .reindex(LAYER_ORDER)
        .reindex(columns=REGIME_ORDER)
        .fillna(0)
    )
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for regime in REGIME_ORDER:
        vals = pivot[regime].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=REGIME_COLORS[regime],
            edgecolor="white",
            linewidth=0.4,
            width=0.72,
            label=REGIME_LABELS[regime],
        )
        bottom += vals
    ax.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of detected SMDEs")
    ax.set_title("Diagnostic regime composition by depth", loc="left", fontsize=8)
    ax.legend(ncol=2, fontsize=6.0, handlelength=1.0, columnspacing=0.8, loc="upper right")

    stage2 = pivot["stage-II-like"].to_numpy(dtype=float)
    early = pivot["early-transient-heavy"].to_numpy(dtype=float)
    for i, (s2, et) in enumerate(zip(stage2, early)):
        ax.text(i, s2 / 2, f"{s2:.0%}", ha="center", va="center", fontsize=6.0, color="white")
        ax.text(i, 1 - et / 2, f"{et:.0%}", ha="center", va="center", fontsize=6.0, color=COLORS["neutral_dark"])
    add_panel_label(ax, "c")


def plot_eligibility_heatmap(ax: plt.Axes, eligible: pd.DataFrame) -> None:
    heat = (
        eligible.pivot_table(
            index="site_id",
            columns="layer",
            values="clean_stageII_48h_events",
            aggfunc="sum",
            observed=False,
        )
        .reindex(columns=LAYER_ORDER)
        .sort_index()
    )
    arr = heat.to_numpy(dtype=float)
    class_arr = np.zeros_like(arr, dtype=float)
    class_arr[(arr > 0) & (arr < MIN_LOCAL_SEGMENT_EVENTS)] = 1
    class_arr[(arr >= MIN_LOCAL_SEGMENT_EVENTS) & (arr < STRONG_SUPPORT_EVENTS)] = 2
    class_arr[arr >= STRONG_SUPPORT_EVENTS] = 3

    cmap = ListedColormap(["#F2F2F2", "#DDE8F1", "#5B8DB8", "#234E70"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(class_arr, aspect="auto", cmap=cmap, norm=norm)
    y_labels = [str(int(s)) if i % 3 == 0 else "" for i, s in enumerate(heat.index)]
    ax.set_yticks(np.arange(len(heat.index)), y_labels, fontsize=5.8)
    ax.set_xticks(np.arange(len(LAYER_ORDER)), [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax.set_xlabel("Sensor depth")
    ax.set_ylabel("FAWN station")
    ax.set_title("Clean stage-II-like events available for local CSR", loc="left", fontsize=8)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            n = int(arr[i, j])
            if n >= MIN_LOCAL_SEGMENT_EVENTS:
                color = "white" if n >= STRONG_SUPPORT_EVENTS else COLORS["neutral_dark"]
                ax.text(j, i, str(n), ha="center", va="center", fontsize=4.7, color=color)
            elif n == 0:
                ax.text(j, i, "-", ha="center", va="center", fontsize=4.4, color="#AAAAAA")

    legend_items = [
        ("none", "#F2F2F2"),
        (f"1-{MIN_LOCAL_SEGMENT_EVENTS - 1}", "#DDE8F1"),
        (f"{MIN_LOCAL_SEGMENT_EVENTS}-{STRONG_SUPPORT_EVENTS - 1}", "#5B8DB8"),
        (f">={STRONG_SUPPORT_EVENTS}", "#234E70"),
    ]
    handles = [patches.Patch(facecolor=color, edgecolor="none", label=label) for label, color in legend_items]
    ax.legend(
        handles=handles,
        title="events",
        fontsize=5.8,
        title_fontsize=6.1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    add_panel_label(ax, "d")


def write_report(events: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    regime_counts = tables["experiment2_regime_composition_by_layer"]
    eligibility = tables["experiment2_csr_eligibility_by_layer"]
    total = len(events)
    overall = (
        events.groupby("regime_proxy", observed=False)
        .size()
        .reset_index(name="events")
        .query("events > 0")
    )
    overall["fraction"] = overall["events"] / total
    clean_stage2 = int(events["clean_stageII_48h"].sum())
    stage2 = int((events["regime_proxy"].astype(str) == "stage-II-like").sum())
    early = int((events["regime_proxy"].astype(str) == "early-transient-heavy").sum())

    lines = [
        "# Experiment 2: Loss-Function Regime Diagnosis",
        "",
        "Prepared: 2026-06-05",
        "",
        "## Purpose",
        "",
        "This experiment diagnoses whether detected SMDEs behave as one uniform drydown population or as a mixture of loss-function regimes. It uses McColl-style loss-function logic as a conceptual frame, but applies event-level high-frequency FAWN diagnostics rather than assuming all events are stage-II drydowns.",
        "",
        "## Core result",
        "",
        f"- Detected SMDEs evaluated: {total:,}.",
        f"- Stage-II-like events by diagnostic proxy: {stage2:,} ({stage2 / total:.1%}).",
        f"- Early-transient-heavy events: {early:,} ({early / total:.1%}).",
        f"- Clean rainfall-associated and stage-II-like construction subset: {clean_stage2:,} ({clean_stage2 / total:.1%}).",
        "",
        "## Overall regime composition",
        "",
        "| Regime proxy | Events | Fraction |",
        "|---|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        regime = REGIME_LABELS[str(row.regime_proxy)]
        lines.append(f"| {regime} | {row.events:,} | {row.fraction:.1%} |")

    lines.extend(
        [
            "",
            "## Layer-level regime composition",
            "",
            "| Layer | Stage-II-like | Stage-I-like | Early transient | Mixed/uncertain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    pivot = (
        regime_counts.pivot(index="layer", columns="regime_proxy", values="fraction_of_layer")
        .reindex(LAYER_ORDER)
        .reindex(columns=REGIME_ORDER)
        .fillna(0)
    )
    for layer in LAYER_ORDER:
        vals = pivot.loc[layer]
        lines.append(
            f"| {LAYER_LABELS[layer]} | {vals['stage-II-like']:.1%} | {vals['stage-I-like']:.1%} | {vals['early-transient-heavy']:.1%} | {vals['mixed_or_uncertain']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## CSR eligibility summary",
            "",
            f"A station-layer is counted as eligible when it has at least {MIN_LOCAL_SEGMENT_EVENTS} clean stage-II-like events. Strong support is counted at at least {STRONG_SUPPORT_EVENTS} events.",
            "",
            "| Layer | Eligible station-layers | Strong station-layers | Clean stage-II-like events |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in eligibility.itertuples(index=False):
        lines.append(
            f"| {row.layer_label} | {row.eligible_station_layers:,} | {row.strong_station_layers:,} | {row.total_clean_stageII_48h_events:,} |"
        )

    lines.extend(
        [
            "",
            "## Figure files",
            "",
            "- `fig_experiment2_loss_function_regime_diagnosis.svg`",
            "- `fig_experiment2_loss_function_regime_diagnosis.pdf`",
            "- `fig_experiment2_loss_function_regime_diagnosis.tiff`",
            "- `fig_experiment2_loss_function_regime_diagnosis.png`",
            "",
            "## Draft figure legend",
            "",
            "Figure X. Loss-function regime diagnosis for detected FAWN SMDEs. (a) Conceptual three-regime soil moisture loss-function frame plotted as loss rate versus storage state; during a drydown, event time moves from wet to dry. (b) Empirical binned loss-storage relation for the 4-inch layer, with median loss rate and interquartile ranges by diagnostic regime proxy. A positive loss-storage slope means faster loss at wetter states and slower loss as storage declines. (c) Depth-wise composition of detected SMDEs by diagnostic regime proxy. Percent labels show the stage-II-like and early-transient-heavy fractions. (d) Station-layer availability of clean rainfall-associated stage-II-like events for local segmented CSR construction; numbers show station-layers with at least the minimum local event count.",
            "",
            "## Interpretation boundary",
            "",
            "The regime labels are diagnostic proxies, not direct physical partitioning of drainage, runoff, evaporation, transpiration, and redistribution. Stage-I-like behavior is especially conservative here and should be treated as a plausible component of the mixed/high-moisture response unless richer atmospheric, vegetation, or management data are added.",
        ]
    )
    (OUT / "experiment2_loss_function_regime_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_figure() -> None:
    ensure_out()
    events = read_events()
    binned = read_binned()
    tables = build_source_tables(events, binned)

    fig = plt.figure(figsize=(7.8, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.05], height_ratios=[0.95, 1.1], wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_conceptual_loss_function(ax_a)
    plot_empirical_loss_storage(ax_b, tables["experiment2_empirical_loss_storage_4in"])
    plot_regime_composition(ax_c, tables["experiment2_regime_composition_by_layer"])
    plot_eligibility_heatmap(ax_d, tables["experiment2_csr_eligibility_by_station_layer"])

    fig.suptitle(
        "Loss-function regime diagnosis separates clean CSR-supporting drydowns from mixed SMDEs",
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
        "Diagnostic proxies use early drop share, trimmed exponential fit, and post-3 h loss-storage relation.",
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["neutral_mid"],
    )
    fig.subplots_adjust(top=0.89)
    save_pub(fig, "fig_experiment2_loss_function_regime_diagnosis")
    plt.close(fig)
    write_report(events, tables)


if __name__ == "__main__":
    build_figure()


