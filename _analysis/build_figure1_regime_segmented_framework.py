from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
OUT = ANALYSIS / "figure1_regime_segmented_framework"
SOURCE = OUT / "source_data"
SUBMISSION_FIGURES = ROOT / "figures"

EVENTS_FILE = ANALYSIS / "fawn_full_smde_audit" / "full_smde_event_audit.csv"
SEGMENTS_FILE = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data" / "adaptive_segment_diagnostics.csv"
POINTS_FILE = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data" / "adaptive_segment_points_all.parquet"

EXAMPLE_EVENT_ID = "S430_moisture_4in_0320"

REGIME_ORDER = ["early_transient", "stageI_like", "stageII_like"]
REGIME_LABELS = {
    "early_transient": "Early transient",
    "stageI_like": "Stage I-like",
    "stageII_like": "Stage II-like",
}
REGIME_TEXT = {
    "early_transient": "wet-end redistribution,\ndrainage, runoff-related adjustment",
    "stageI_like": "approximately storage-\ninvariant loss",
    "stageII_like": "storage-limited\nwater loss",
}
REGIME_COLORS = {
    "early_transient": "#C78D4B",
    "stageI_like": "#4E7E9E",
    "stageII_like": "#6FA38A",
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
    "grid": "#E6EBEF",
    "accent": "#A64B56",
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
    shutil.copyfile(OUT / f"{stem}.pdf", SUBMISSION_FIGURES / "Figure_2_regime_segmentation_concept.pdf")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.09, y: float = 1.06) -> None:
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


def plot_loss_concept(ax: plt.Axes) -> None:
    x = np.linspace(0.02, 1.0, 400)
    y = np.piecewise(
        x,
        [x <= 0.42, (x > 0.42) & (x <= 0.74), x > 0.74],
        [
            lambda z: 0.08 + 0.95 * (z / 0.42) ** 1.25,
            lambda z: 1.02 + 0.02 * np.sin((z - 0.42) / 0.32 * np.pi),
            lambda z: 1.02 + 4.7 * (z - 0.74) ** 1.85,
        ],
    )
    spans = [
        (0.02, 0.42, "stageII_like"),
        (0.42, 0.74, "stageI_like"),
        (0.74, 1.0, "early_transient"),
    ]
    for x0, x1, regime in spans:
        ax.axvspan(x0, x1, color=REGIME_COLORS[regime], alpha=0.16, lw=0)
    ax.plot(x, y, color=COLORS["neutral_dark"], lw=1.8)
    ax.plot(x[x <= 0.42], y[x <= 0.42], color=REGIME_COLORS["stageII_like"], lw=2.4)
    ax.plot(x[(x > 0.42) & (x <= 0.74)], y[(x > 0.42) & (x <= 0.74)], color=REGIME_COLORS["stageI_like"], lw=2.4)
    ax.plot(x[x > 0.74], y[x > 0.74], color=REGIME_COLORS["early_transient"], lw=2.4)

    labels = [
        ("Stage II-like\nstorage-limited", 0.22, 0.62, "stageII_like"),
        ("Stage I-like\nnear-constant", 0.58, 1.24, "stageI_like"),
        ("Early transient\nwet-end loss", 0.88, 1.62, "early_transient"),
    ]
    for text, xx, yy, regime in labels:
        ax.text(xx, yy, text, ha="center", va="center", fontsize=6.7, color=REGIME_COLORS[regime])

    ax.annotate(
        "event time moves\nwet -> dry",
        xy=(0.36, 0.18),
        xytext=(0.68, 0.18),
        arrowprops=dict(arrowstyle="->", color=COLORS["neutral_mid"], lw=0.8),
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["neutral_mid"],
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.9)
    ax.set_xticks([0.1, 0.5, 0.9], ["dry", "mid", "wet"])
    ax.set_yticks([])
    ax.set_xlabel("Soil water state, $S$ or normalized $x$")
    ax.set_ylabel("Loss rate, $L=-dS/dt$")
    ax.set_title("Loss-function regimes define the modeling states", loc="left", fontsize=8, pad=6)
    add_panel_label(ax, "a")


def load_example() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])
    segments = pd.read_csv(SEGMENTS_FILE)
    points = pd.read_parquet(POINTS_FILE)
    event = events.loc[events["event_id"] == EXAMPLE_EVENT_ID].iloc[0]
    seg = segments.loc[segments["event_id"] == EXAMPLE_EVENT_ID].sort_values("adaptive_segment_order").copy()
    pts = points.loc[points["event_id"] == EXAMPLE_EVENT_ID].sort_values("t_h").copy()
    seg.to_csv(SOURCE / "figure1_example_adaptive_segments.csv", index=False)
    pts.to_csv(SOURCE / "figure1_example_points.csv", index=False)
    return seg, pts, event


def plot_adaptive_example(ax: plt.Axes, seg: pd.DataFrame, pts: pd.DataFrame, event: pd.Series) -> None:
    for row in seg.itertuples(index=False):
        regime = str(row.segment_regime)
        color = REGIME_COLORS.get(regime, "#BFC3C7")
        ax.axvspan(float(row.start_t_h), float(row.end_t_h), color=color, alpha=0.18, lw=0)
    ax.plot(pts["t_h"], pts["moisture_mm"], color=COLORS["neutral_dark"], lw=1.15, alpha=0.82)
    ax.scatter(pts["t_h"], pts["moisture_mm"], s=5.5, color=COLORS["neutral_dark"], alpha=0.45, linewidths=0)
    for row in seg.itertuples(index=False):
        regime = str(row.segment_regime)
        color = REGIME_COLORS.get(regime, "#BFC3C7")
        order = int(row.adaptive_segment_order)
        use = pts[pts["adaptive_segment_order"].eq(order)].sort_values("t_h").copy()
        if len(use) < 2:
            use = pts[pts["t_h"].between(float(row.start_t_h), float(row.end_t_h), inclusive="both")].sort_values("t_h").copy()
        if len(use) < 2:
            continue
        x = use["t_h"].to_numpy(float)
        y = use["moisture_mm"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.array([float(x.min()), float(x.max())])
        y_fit = intercept + slope * x_fit
        ax.plot(x_fit, y_fit, color=color, lw=2.15, solid_capstyle="round", zorder=6)
        label_x = float(np.mean(x_fit))
        label_y = float(np.mean(y_fit))
        loss_rate = max(0.0, -float(slope))
        ax.text(
            label_x,
            label_y,
            f"L={loss_rate:.2f} mm h$^{{-1}}$",
            ha="center",
            va="center",
            fontsize=5.7,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "none", "alpha": 0.84},
            zorder=8,
        )
    for boundary in seg["end_t_h"].iloc[:-1]:
        ax.axvline(float(boundary), color=COLORS["neutral_mid"], lw=0.9, ls="--")
    y_min = float(pts["moisture_mm"].min())
    y_max = float(pts["moisture_mm"].max())
    pad = max(0.08, (y_max - y_min) * 0.14)
    ax.set_ylim(y_min - pad, y_max + pad)
    start = pd.Timestamp(event["start"])
    ax.text(
        0.02,
        0.06,
        f"FAWN {int(event['site_id'])}, 4 in; start {start:%Y-%m-%d %H:%M}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=COLORS["neutral_mid"],
    )
    ax.set_xlabel("Elapsed time since SMDE start (h)")
    ax.set_ylabel("Soil water amount, $S(t)$ (mm)")
    ax.set_title("Observed soil moisture still declines, but each segment has a different loss rate", loc="left", fontsize=8, pad=7)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    handles = [
        patches.Patch(color=REGIME_COLORS[regime], alpha=0.40, label=REGIME_LABELS[regime])
        for regime in REGIME_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.0, handlelength=0.9, borderaxespad=0.25)
    add_panel_label(ax, "b")


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.30, height: float = 0.16) -> None:
    x, y = xy
    rect = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        fc=color,
        ec="none",
        alpha=0.20,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=6.7, transform=ax.transAxes)


def plot_workflow(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_title("Operational workflow", loc="left", fontsize=8, pad=6)
    boxes = [
        ((0.02, 0.66), "SM-only\nSMDE detection", "#5B8DB8"),
        ((0.36, 0.66), "independent\nrainfall audit", "#355C7D"),
        ((0.70, 0.66), "adaptive\nregime diagnosis", "#6FA38A"),
        ((0.18, 0.34), "CSR response\nlibraries", "#4E7E9E"),
        ((0.54, 0.34), "calibrated\nrecent-loss forecast", "#C78D4B"),
    ]
    for xy, text, color in boxes:
        draw_box(ax, xy, text, color)
    arrows = [
        ((0.32, 0.74), (0.36, 0.74)),
        ((0.66, 0.74), (0.70, 0.74)),
        ((0.80, 0.66), (0.34, 0.50)),
        ((0.48, 0.42), (0.54, 0.42)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", color=COLORS["neutral_mid"], lw=1.0),
        )
    ax.text(
        0.02,
        0.12,
        "CSR summarizes regime-consistent responses.\nForecasting uses calibrated recent loss at the origin.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=COLORS["neutral_mid"],
    )
    add_panel_label(ax, "c", x=-0.06)


def plot_definitions(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_title("Definitions used throughout the manuscript", loc="left", fontsize=8, pad=6)
    rows = [
        ("Layer storage", r"$S(t)=\Delta z\,\theta(t)$"),
        ("Loss rate", r"$L_i=-(S_{i+1}-S_i)/\Delta t_i$"),
        ("Storage coordinate", r"$x_i=(S_{mid,i}-S_{min})/(S_{max}-S_{min})$"),
        ("Adaptive segmentation", r"$\min_{\tau,K}\sum_k SSE_k+\lambda K\log n$"),
        ("CSR registration", r"$\delta_{ij}=\arg\min_\delta(D_S+\lambda D_L)$"),
        ("Forecast operator", r"$\hat S(t+h)=S(t)-\alpha_{h,r,l,s}L_{recent}(t)h$"),
    ]
    y0 = 0.86
    row_h = 0.125
    for i, (label, formula) in enumerate(rows):
        y = y0 - i * row_h
        ax.add_patch(
            patches.Rectangle(
                (0.02, y - 0.060),
                0.96,
                0.087,
                fc="#F7F8FA" if i % 2 == 0 else "white",
                ec="#D9DEE3",
                lw=0.5,
                transform=ax.transAxes,
            )
        )
        ax.text(0.05, y - 0.018, label, transform=ax.transAxes, fontsize=6.1, fontweight="bold", va="center")
        ax.text(0.41, y - 0.018, formula, transform=ax.transAxes, fontsize=6.15, va="center")
    ax.text(
        0.02,
        0.04,
        "Regime labels are diagnostic states inferred from soil moisture behavior; they are not direct flux partitions.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=COLORS["neutral_mid"],
    )
    add_panel_label(ax, "d", x=-0.06)


def write_report(seg: pd.DataFrame, event: pd.Series) -> None:
    lines = [
        "# Figure 1 framework report",
        "",
        "Figure 2 was rebuilt as a two-panel concept figure. Workflow and notation panels were removed from the figure and should be described in Section 3.2.",
        "",
        f"Example event: {EXAMPLE_EVENT_ID}",
        f"Site: {int(event['site_id'])}",
        f"Layer: {event['layer']}",
        f"Start: {pd.Timestamp(event['start'])}",
        f"End: {pd.Timestamp(event['end'])}",
        "",
        "Adaptive segments:",
        "",
        seg[
            [
                "adaptive_segment_order",
                "start_t_h",
                "end_t_h",
                "segment_regime",
                "segment_drop_mm",
                "segment_loss_storage_corr",
            ]
        ].to_markdown(index=False),
        "",
        "Manuscript use: define loss-function states and show that within-event boundaries are adaptive.",
    ]
    (OUT / "figure1_regime_segmented_framework_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_figure() -> None:
    ensure_out()
    seg, pts, event = load_example()

    fig = plt.figure(figsize=(7.9, 3.85), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.10], wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_loss_concept(ax_a)
    plot_adaptive_example(ax_b, seg, pts, event)

    fig.suptitle(
        "SMDE regime diagnosis supports calibrated recent-loss forecasting",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )
    fig.text(
        0.01,
        0.943,
        "Regimes are inferred within events before regime-specific CSR libraries and recent-loss forecasts are built.",
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["neutral_mid"],
    )
    fig.subplots_adjust(left=0.070, right=0.985, top=0.835, bottom=0.145)
    save_pub(fig, "fig_regime_segmented_framework")
    plt.close(fig)
    write_report(seg, event)


if __name__ == "__main__":
    build_figure()
