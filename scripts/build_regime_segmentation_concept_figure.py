from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_analysis" / "journal_hydrology_submission" / "conceptual_figure"
EMPIRICAL = ROOT / "_analysis" / "experiment2_loss_function_regime_diagnosis" / "source_data" / "experiment2_empirical_loss_storage_4in.csv"
REP_POINTS = ROOT / "_analysis" / "experiment3_localized_segmented_csr" / "source_data" / "experiment3_representative_location_aligned_points.csv"
OUT.mkdir(parents=True, exist_ok=True)


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


COL = {
    "ink": "#303030",
    "muted": "#6F7378",
    "early": "#C78D4B",
    "stage1": "#4F7B9C",
    "stage2": "#6FA38A",
    "mixed": "#9A9A9A",
    "fill_early": "#F4E3D0",
    "fill_stage1": "#DCE8F4",
    "fill_stage2": "#E1EEE6",
    "grid": "#E7EBEF",
    "accent": "#A64B56",
}

REGIME_COLORS = {
    "Stage-II-like": COL["stage2"],
    "Stage-I-like": COL["stage1"],
    "Early transient": COL["early"],
    "Mixed/uncertain": COL["mixed"],
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COL["ink"],
    )


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COL["grid"], lw=0.6, zorder=0)
    ax.tick_params(labelsize=7)


def plot_canonical_loss_function(ax: plt.Axes) -> None:
    theta_w = 0.08
    theta_star = 0.60
    theta_fc = 0.72
    emax = 1.0

    x2 = np.linspace(theta_w, theta_star, 140)
    l2 = emax * ((x2 - theta_w) / (theta_star - theta_w)) ** 1.05
    x1 = np.linspace(theta_star, theta_fc, 40)
    l1 = np.full_like(x1, emax)
    xw = np.linspace(theta_fc, 1.0, 120)
    lw = emax + 1.15 * ((xw - theta_fc) / (1.0 - theta_fc)) ** 2.7

    ax.axvspan(theta_w, theta_star, color=COL["fill_stage2"], lw=0)
    ax.axvspan(theta_star, theta_fc, color=COL["fill_stage1"], lw=0)
    ax.axvspan(theta_fc, 1.0, color=COL["fill_early"], lw=0)
    ax.plot(x2, l2, color=COL["stage2"], lw=2.2)
    ax.plot(x1, l1, color=COL["stage1"], lw=2.2)
    ax.plot(xw, lw, color=COL["early"], lw=2.2)
    ax.axhline(emax, color=COL["muted"], lw=0.9, ls="--")
    for xpos, lab in [(theta_w, r"$\theta_w$"), (theta_star, r"$\theta^*$"), (theta_fc, r"$\theta_{fc}$")]:
        ax.axvline(xpos, ymin=0, ymax=0.82, color=COL["muted"], lw=0.8, ls="--")
        ax.text(
            xpos,
            0.055,
            lab,
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4),
        )

    ax.annotate("Stage II\nwater-limited", xy=(0.31, 1.55), ha="center", va="center", color=COL["stage2"], fontsize=7)
    ax.annotate("Stage I\nET-limited", xy=(0.66, 1.55), ha="center", va="center", color=COL["stage1"], fontsize=7)
    ax.annotate("Wet transient\ndrainage/runoff", xy=(0.86, 1.55), ha="center", va="center", color=COL["early"], fontsize=7)
    ax.text(0.015, emax + 0.03, r"$E_{max}$", ha="left", va="bottom", fontsize=7, color=COL["muted"])
    ax.annotate(
        "drydown time",
        xy=(0.74, 0.10),
        xytext=(0.46, 0.10),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<-", color=COL["muted"], lw=0.8),
        fontsize=6.8,
        color=COL["muted"],
        ha="center",
        va="center",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 2.28)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel(r"Soil water state, $\theta$ or $S$")
    ax.set_ylabel(r"Loss rate, $L(\theta)$")
    ax.set_title("Canonical three-regime loss function", loc="left", fontsize=8, pad=7)
    style_axis(ax)


def plot_fawn_empirical_loss(ax: plt.Axes) -> None:
    data = pd.read_csv(EMPIRICAL)
    order = ["Stage-II-like", "Stage-I-like", "Early transient", "Mixed/uncertain"]
    for label in order:
        d = data[data["regime_label"] == label].sort_values("storage_norm")
        if d.empty:
            continue
        color = REGIME_COLORS[label]
        ax.plot(d["storage_norm"], d["loss_mm_h_median"], color=color, lw=1.9, label=label)
        ax.fill_between(
            d["storage_norm"].to_numpy(float),
            d["loss_mm_h_q25"].to_numpy(float),
            d["loss_mm_h_q75"].to_numpy(float),
            color=color,
            alpha=0.13,
            lw=0,
        )
    ax.annotate(
        "event time",
        xy=(0.78, 0.10),
        xytext=(0.52, 0.10),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<-", color=COL["muted"], lw=0.8),
        fontsize=6.8,
        color=COL["muted"],
        ha="center",
        va="center",
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(0.88, data["loss_mm_h_q75"].quantile(0.98) * 1.10))
    ax.set_xlabel(r"Normalized event storage, $x$ (dry 0 $\rightarrow$ wet 1)")
    ax.set_ylabel(r"Median loss rate (mm h$^{-1}$)")
    ax.set_title("Empirical FAWN loss-storage relation, 4 in layer", loc="left", fontsize=8, pad=7)
    ax.legend(loc="upper left", fontsize=6.4, handlelength=1.8, ncol=1)
    style_axis(ax)


def plot_representative_drydown(ax: plt.Axes) -> None:
    points = pd.read_csv(REP_POINTS)
    event_id = points.groupby("event_id")["t_h"].max().sort_values(ascending=False).index[0]
    d = points[points["event_id"] == event_id].sort_values("t_h").copy()
    threshold = d["end_mm"].iloc[0] + 0.25 * (d["start_mm"].iloc[0] - d["end_mm"].iloc[0])
    late_rows = d[d["event_storage_norm"] < 0.25]
    t_cross = float(late_rows["t_h"].iloc[0]) if not late_rows.empty else float(d["t_h"].max())

    ax.axvspan(0, 3, color=COL["fill_early"], lw=0)
    ax.axvspan(3, t_cross, color=COL["fill_stage1"], lw=0)
    ax.axvspan(t_cross, d["t_h"].max(), color=COL["fill_stage2"], lw=0)
    ax.plot(d["t_h"], d["moisture_mm"], color=COL["ink"], lw=2.0)
    ax.scatter(d["t_h"], d["moisture_mm"], s=7, color=COL["ink"], alpha=0.55, zorder=3)
    ax.axvline(3, color=COL["early"], lw=1.0, ls="--")
    ax.axhline(threshold, color=COL["stage2"], lw=1.0, ls=":")
    ax.text(1.75, d["moisture_mm"].max() + 0.40, "first 3 h", ha="center", va="bottom", color=COL["early"], fontsize=6.8)
    ax.text((3 + t_cross) / 2, d["moisture_mm"].max() + 0.40, r"post-3 h, $x_s\geq0.25$", ha="center", va="bottom", color=COL["stage1"], fontsize=6.8)
    ax.text((t_cross + d["t_h"].max()) / 2, d["moisture_mm"].max() + 0.40, r"$x_s<0.25$", ha="center", va="bottom", color=COL["stage2"], fontsize=6.8)
    ax.text(d["t_h"].max() * 0.98, threshold + 0.10, r"$x_s=0.25$", ha="right", va="bottom", fontsize=6.8, color=COL["stage2"])
    ax.set_xlim(-0.25, d["t_h"].max())
    ax.set_ylim(d["moisture_mm"].min() - 0.35, d["moisture_mm"].max() + 0.90)
    ax.set_xlabel("Elapsed time since SMDE start (h)")
    ax.set_ylabel("Soil water amount, S(t) (mm)")
    ax.set_title("Representative FAWN drydown segmented for CSR", loc="left", fontsize=8, pad=7)
    style_axis(ax)


def plot_operational_rules(ax: plt.Axes) -> None:
    ax.axis("off")
    rows = [
        ("Loss rate", r"$L_i=-(S_{i+1}-S_i)/\Delta t_i$", COL["ink"]),
        ("Storage coordinate", r"$x=(S_{mid}-S_{min})/(S_{max}-S_{min})$", COL["ink"]),
        ("Stage-II-like", r"$R^2_{trim3}\geq0.70$ and $r(L,x)_{post3}\geq0.20$", COL["stage2"]),
        ("Stage-I-like", r"$|r(L,x)_{post3}|<0.20$ and $CV(L)_{post3}<0.60$", COL["stage1"]),
        ("Early transient-heavy", r"$E_3\geq0.40$", COL["early"]),
    ]
    ax.text(0.02, 0.93, "Operational translation used in this study", ha="left", va="center", fontsize=8, fontweight="bold", color=COL["ink"])
    y = 0.80
    for label, definition, color in rows:
        ax.add_patch(
            plt.Rectangle((0.02, y - 0.055), 0.96, 0.095, transform=ax.transAxes, facecolor="white", edgecolor="#E1E4E8", lw=0.6)
        )
        ax.text(0.05, y, label, ha="left", va="center", fontsize=6.7, color=color, fontweight="bold")
        ax.text(0.42, y, definition, ha="left", va="center", fontsize=6.45, color=COL["ink"])
        y -= 0.13
    ax.text(
        0.02,
        0.05,
        "Labels diagnose soil-moisture behavior for event filtering; they do not partition measured fluxes.",
        ha="left",
        va="bottom",
        fontsize=7,
        color=COL["muted"],
    )
    ax.set_title("Diagnostic equations and regime labels", loc="left", fontsize=8, pad=7)


def main() -> None:
    fig = plt.figure(figsize=(7.4, 6.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        top=0.86,
        bottom=0.09,
        hspace=0.58,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_canonical_loss_function(ax_a)
    plot_fawn_empirical_loss(ax_b)
    plot_representative_drydown(ax_c)
    plot_operational_rules(ax_d)
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        add_panel_label(ax, label)

    fig.suptitle(
        "Three-regime loss-function framework used to diagnose FAWN drydowns",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(
        0.08,
        0.945,
        "Conceptual loss functions define the regimes; FAWN 15-min observations provide the empirical loss-storage relation used for event filtering.",
        ha="left",
        va="top",
        fontsize=7.3,
        color=COL["muted"],
    )
    save_pub(fig, "fig_regime_segmentation_concept")


if __name__ == "__main__":
    main()
