from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures"
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
    "mid": "#355C7D",
    "late": "#6FA38A",
    "uncertain": "#9A9A9A",
    "fill_early": "#F4E3D0",
    "fill_mid": "#DCE8F4",
    "fill_late": "#E1EEE6",
    "fill_gray": "#ECEFF2",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.07,
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


def plot_segmentation(ax: plt.Axes) -> None:
    t = np.linspace(0, 10, 400)
    s = 4.2 + 5.8 * np.exp(-t / 4.2) - 0.35 * (1 - np.exp(-t / 1.1))
    s0 = float(s[0])
    send = float(s[-1])
    x_storage = (s - send) / (s0 - send)
    threshold = send + 0.25 * (s0 - send)
    cross_idx = np.argmin(np.abs(x_storage - 0.25))
    t_cross = float(t[cross_idx])

    ax.axvspan(0, 3, color=COL["fill_early"], lw=0, zorder=0)
    ax.axvspan(3, t_cross, color=COL["fill_mid"], lw=0, zorder=0)
    ax.axvspan(t_cross, 10, color=COL["fill_late"], lw=0, zorder=0)
    ax.plot(t, s, color=COL["ink"], lw=2.1)
    ax.axvline(3, color=COL["early"], lw=1.2, ls="--")
    ax.axhline(threshold, color=COL["late"], lw=1.0, ls=":")
    ax.scatter([0, 3, t_cross, 10], [s[0], np.interp(3, t, s), threshold, send], s=16, color=COL["ink"], zorder=3)

    top_y = s0 + 0.78
    ax.text(1.5, top_y, "0-3 h", ha="center", va="center", color=COL["early"], fontsize=7.2, fontweight="bold")
    ax.text((3 + t_cross) / 2, top_y, r"post-3 h, $x_s\geq0.25$", ha="center", va="center", color=COL["mid"], fontsize=7.2, fontweight="bold")
    ax.text((t_cross + 10) / 2, top_y, r"post-3 h, $x_s<0.25$", ha="center", va="center", color=COL["late"], fontsize=7.2, fontweight="bold")
    ax.text(3.05, send + 0.12, "t = 3 h", ha="left", va="bottom", color=COL["early"], fontsize=6.7)
    ax.text(9.85, threshold + 0.10, r"$x_s=0.25$", ha="right", va="bottom", color=COL["late"], fontsize=6.9)
    ax.set_xlim(0, 10)
    ax.set_ylim(send - 0.2, s0 + 1.15)
    ax.set_xlabel("Elapsed time since SMDE start, t (h)")
    ax.set_ylabel("Soil water amount, S(t)")
    ax.set_title("SMDE segmentation used for CSR", loc="left", fontsize=8, pad=8)


def plot_loss_regimes(ax: plt.Axes) -> None:
    x = np.linspace(0, 1, 400)
    stage2 = 0.12 + 0.95 * x**1.45
    stage1 = np.full_like(x, 0.78)
    early = 0.50 + 1.35 * x**3.0

    ax.axvspan(0.00, 0.42, color=COL["fill_late"], lw=0)
    ax.axvspan(0.42, 0.76, color=COL["fill_mid"], lw=0)
    ax.axvspan(0.76, 1.00, color=COL["fill_early"], lw=0)
    ax.plot(x, stage2, color=COL["late"], lw=2.2, label="Stage-II-like")
    ax.plot(x, stage1, color=COL["mid"], lw=1.8, label="Stage-I-like")
    ax.plot(x, early, color=COL["early"], lw=1.8, label="Early transient-heavy")
    ax.fill_between(x, stage2 - 0.05, stage2 + 0.05, color=COL["late"], alpha=0.08, lw=0)
    ax.fill_between(x, early - 0.08, early + 0.08, color=COL["early"], alpha=0.08, lw=0)

    ax.text(0.21, 1.77, "storage-limited", ha="center", va="center", color=COL["late"], fontsize=6.8)
    ax.text(0.59, 1.77, "storage-invariant", ha="center", va="center", color=COL["mid"], fontsize=6.8)
    ax.text(0.88, 1.77, "wet transient", ha="center", va="center", color=COL["early"], fontsize=6.8)
    ax.text(1.025, stage2[-1], "Stage-II", ha="left", va="center", color=COL["late"], fontsize=6.8, clip_on=False)
    ax.text(1.025, stage1[-1], "Stage-I", ha="left", va="center", color=COL["mid"], fontsize=6.8, clip_on=False)
    ax.text(1.025, early[-1] - 0.02, "early", ha="left", va="center", color=COL["early"], fontsize=6.8, clip_on=False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.85)
    ax.annotate(
        "event time",
        xy=(0.82, 0.13),
        xytext=(0.56, 0.13),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<-", color=COL["muted"], lw=0.8),
        fontsize=6.5,
        color=COL["muted"],
        ha="center",
        va="center",
    )
    ax.set_xlabel(r"Normalized event storage, $x_s$ (dry 0 $\rightarrow$ wet 1)")
    ax.set_ylabel(r"Loss rate, $L=-dS/dt$")
    ax.set_title("Diagnostic loss rate versus storage", loc="left", fontsize=8, pad=8)


def plot_rules(ax: plt.Axes) -> None:
    ax.axis("off")
    rules = [
        ("Storage coordinate", r"$x_s(t)=\{S(t)-S_{end}\}/\{S_{start}-S_{end}\}$", COL["ink"]),
        ("Loss rate", r"$L_i=-(S_{i+1}-S_i)/\Delta t_i$", COL["ink"]),
        ("Stage-II-like", r"$R^2_{\mathrm{trim3}}\geq0.70$ and $r(L,x)_{\mathrm{post3}}\geq0.20$", COL["late"]),
        ("Stage-I-like", r"$|r(L,x)_{\mathrm{post3}}|<0.20$ and $CV(L)_{\mathrm{post3}}<0.60$", COL["mid"]),
        ("Early transient-heavy", r"$E_3\geq0.40$", COL["early"]),
        ("Mixed/uncertain", "events not meeting the above diagnostic rules", COL["muted"]),
    ]
    ax.text(0.02, 0.93, "Quantity or class", ha="left", va="center", fontsize=7.2, color=COL["muted"])
    ax.text(0.30, 0.93, "Operational definition", ha="left", va="center", fontsize=7.2, color=COL["muted"])
    y = 0.81
    for label, eq, color in rules:
        ax.add_patch(
            plt.Rectangle((0.015, y - 0.065), 0.97, 0.105, transform=ax.transAxes, facecolor="white", edgecolor="#E1E4E8", lw=0.6)
        )
        ax.text(0.035, y, label, ha="left", va="center", fontsize=7.2, color=color, fontweight="bold")
        ax.text(0.30, y, eq, ha="left", va="center", fontsize=7.2, color=COL["ink"])
        y -= 0.125
    ax.text(
        0.02,
        0.035,
        "Regime labels are diagnostic proxies for event filtering, not direct flux partitioning.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=COL["muted"],
    )
    ax.set_title("Event-level diagnostic rules", loc="left", fontsize=8)


def main() -> None:
    fig = plt.figure(figsize=(7.4, 5.05), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.92],
        width_ratios=[1.04, 1.0],
        left=0.075,
        right=0.985,
        top=0.82,
        bottom=0.11,
        hspace=0.78,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_segmentation(ax_a)
    plot_loss_regimes(ax_b)
    plot_rules(ax_c)
    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")

    fig.suptitle(
        "Regime diagnosis and segmentation logic for localized segmented CSR",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(0.075, 0.935, "Diagnostic labels are used for event filtering and CSR segmentation.", ha="left", va="top", fontsize=7.2, color=COL["muted"])
    save_pub(fig, "fig_regime_segmentation_concept")


if __name__ == "__main__":
    main()


