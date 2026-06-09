from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
OUT = ANALYSIS / "figure2_response_centric_theory"
SOURCE = OUT / "source_data"

INTEGRATED_TABLE = ANALYSIS / "response_centric_2x2" / "SMDE_CSR_response_centric_2x2_integrated_compact_table.csv"
ONFARM_SUPPORT = ANALYSIS / "response_centric_2x2" / "SMDE_CSR_onfarm_input_class_forecast_support_by_split.csv"

COLORS = {
    "fawn": "#4C78A8",
    "onfarm": "#D55E00",
    "static": "#7A7A7A",
    "response": "#1B9E77",
    "threshold": "#B7B7B7",
    "light_blue": "#EAF2FA",
    "light_green": "#EAF6EF",
    "light_orange": "#FFF1E8",
    "grid": "#DDE3E8",
    "text": "#222222",
    "muted": "#646464",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
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


def panel_label(ax: plt.Axes, label: str, x: float = -0.035, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["text"],
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "none",
    fontsize: float = 7.0,
    weight: str = "normal",
) -> None:
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        fc=facecolor,
        ec=edgecolor,
        lw=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["text"],
    )


def axis_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9, shrinkA=0, shrinkB=0),
    )


def plot_property_mini(ax: plt.Axes, origin: tuple[float, float], width: float, height: float) -> None:
    x0, y0 = origin
    # Mini coordinate system in axes coordinates.
    ax.plot([x0, x0], [y0, y0 + height], color=COLORS["muted"], lw=0.8, transform=ax.transAxes, clip_on=False)
    ax.plot([x0, x0 + width], [y0, y0], color=COLORS["muted"], lw=0.8, transform=ax.transAxes, clip_on=False)
    for frac, label in [(0.72, "S_upper"), (0.28, "S_lower")]:
        yy = y0 + height * frac
        ax.plot([x0, x0 + width], [yy, yy], color=COLORS["threshold"], lw=1.0, ls="--", transform=ax.transAxes)
        ax.text(x0 + width + 0.01, yy, label, ha="left", va="center", fontsize=5.8, color=COLORS["muted"], transform=ax.transAxes)
    t = np.linspace(0, 1, 80)
    for offset, alpha in [(0.00, 0.55), (0.08, 0.35), (-0.06, 0.35)]:
        y = 0.82 - 0.55 * (1 - np.exp(-2.5 * t)) + offset * np.sin(2 * np.pi * t)
        ax.plot(x0 + width * t, y0 + height * y, color=COLORS["static"], lw=1.0, alpha=alpha, transform=ax.transAxes)


def plot_response_mini(ax: plt.Axes, origin: tuple[float, float], width: float, height: float) -> None:
    x0, y0 = origin
    ax.plot([x0, x0], [y0, y0 + height], color=COLORS["muted"], lw=0.8, transform=ax.transAxes, clip_on=False)
    ax.plot([x0, x0 + width], [y0, y0], color=COLORS["muted"], lw=0.8, transform=ax.transAxes, clip_on=False)
    t = np.linspace(0, 1, 120)
    curves = [
        (0.87 - 0.70 * (1 - np.exp(-4.2 * t)), "#C78D4B", "rain"),
        (0.83 - 0.50 * t - 0.05 * np.sin(1.3 * np.pi * t), COLORS["response"], "irrig."),
        (0.90 - 0.62 * (t**1.6), COLORS["fawn"], "mixed"),
    ]
    for y, color, label in curves:
        ax.plot(x0 + width * t, y0 + height * y, color=color, lw=1.4, alpha=0.9, transform=ax.transAxes)
    ax.scatter([x0 + width * 0.05, x0 + width * 0.05], [y0 + height * 0.86, y0 + height * 0.78], s=10, color=["#4C78A8", COLORS["onfarm"]], transform=ax.transAxes, zorder=5)
    ax.text(x0 + width * 0.02, y0 + height * 1.03, "water input", ha="left", va="bottom", fontsize=5.8, color=COLORS["muted"], transform=ax.transAxes)


def plot_framework_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.025, y=1.02)
    ax.text(
        0.03,
        0.98,
        "Representation shift under agricultural local variability",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["text"],
    )

    add_box(
        ax,
        (0.04, 0.74),
        0.38,
        0.12,
        "Static/property-centric operator\n$\\hat{S}_{static}(t+h)=F_h(S(t),\\ell(t),q)$",
        facecolor="#F1F1F1",
        edgecolor="#D3D3D3",
        fontsize=6.8,
        weight="bold",
    )
    add_box(
        ax,
        (0.58, 0.74),
        0.38,
        0.12,
        "Response-centric operator\n$\\hat{S}_{response}(t+h)=G_h(S(t),\\ell(t),q,c_e)$",
        facecolor=COLORS["light_green"],
        edgecolor="#C8DED0",
        fontsize=6.8,
        weight="bold",
    )
    axis_arrow(ax, (0.45, 0.79), (0.55, 0.79), color=COLORS["response"])
    ax.text(0.50, 0.84, "add response context $c_e$", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color=COLORS["response"])

    plot_property_mini(ax, (0.07, 0.46), 0.30, 0.18)
    plot_response_mini(ax, (0.61, 0.46), 0.30, 0.18)

    add_box(
        ax,
        (0.06, 0.25),
        0.34,
        0.11,
        "local variability\nresidual around $F_h$",
        facecolor="#FAFAFA",
        edgecolor="#D8D8D8",
        fontsize=6.7,
    )
    add_box(
        ax,
        (0.60, 0.25),
        0.34,
        0.11,
        "$c_e$ = input class + regime\n+ local response memory",
        facecolor=COLORS["light_orange"],
        edgecolor="#F4C7A6",
        fontsize=6.4,
    )
    axis_arrow(ax, (0.23, 0.43), (0.23, 0.37), color=COLORS["muted"])
    axis_arrow(ax, (0.77, 0.43), (0.77, 0.37), color=COLORS["muted"])

    ax.text(
        0.50,
        0.09,
        "testable prediction: response gains stay near zero in FAWN but increase on-farm if $c_e$ contains reusable morphology",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        color=COLORS["muted"],
    )


def load_main_table() -> pd.DataFrame:
    table = pd.read_csv(INTEGRATED_TABLE)
    main = table[table["evidence_tier"].eq("1_main_conservative_static_recession")].copy()
    main = main.sort_values(["data_setting", "horizon_h"]).reset_index(drop=True)
    main.to_csv(SOURCE / "figure2_main_2x2_values.csv", index=False)
    support = pd.read_csv(ONFARM_SUPPORT)
    support.to_csv(SOURCE / "figure2_onfarm_input_support.csv", index=False)
    return main


def plot_delta_r2(ax: plt.Axes, table: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.11, y=1.05)
    style = {
        "FAWN rainfall-only": ("FAWN rainfall-only", COLORS["fawn"], "o"),
        "On-farm managed": ("On-farm managed", COLORS["onfarm"], "s"),
    }
    for setting, group in table.groupby("data_setting", sort=False):
        label, color, marker = style.get(setting, (setting, COLORS["text"], "o"))
        group = group.sort_values("horizon_h")
        ax.plot(group["horizon_h"], group["delta_r2"], color=color, marker=marker, ms=4.2, lw=1.7, label=label)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xscale("log", base=2)
    horizons = sorted(table["horizon_h"].unique())
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(h)) for h in horizons])
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Response minus static R2")
    ax.set_title("Response value increases in the managed setting", loc="left", fontsize=8.0, pad=7)
    ax.grid(True, color=COLORS["grid"], lw=0.55)
    ax.legend(loc="upper left", fontsize=6.3)


def plot_rmse_ratio(ax: plt.Axes, table: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.11, y=1.05)
    style = {
        "FAWN rainfall-only": ("FAWN rainfall-only", COLORS["fawn"], "o"),
        "On-farm managed": ("On-farm managed", COLORS["onfarm"], "s"),
    }
    for setting, group in table.groupby("data_setting", sort=False):
        label, color, marker = style.get(setting, (setting, COLORS["text"], "o"))
        group = group.sort_values("horizon_h")
        ax.plot(
            group["horizon_h"],
            group["rmse_ratio_response_over_static"],
            color=color,
            marker=marker,
            ms=4.2,
            lw=1.7,
            label=label,
        )
    ax.axhline(1, color="#555555", lw=0.8)
    ax.set_xscale("log", base=2)
    horizons = sorted(table["horizon_h"].unique())
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(h)) for h in horizons])
    ax.set_ylim(0.34, 1.08)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Response/static RMSE ratio")
    ax.set_title("Error reduction is larger on farm", loc="left", fontsize=8.0, pad=7)
    ax.grid(True, color=COLORS["grid"], lw=0.55)
    ax.text(
        0.98,
        0.14,
        "values < 1 favor\nresponse-centric",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
    )


def write_report(table: pd.DataFrame) -> None:
    onfarm_24 = table[(table["data_setting"].eq("On-farm managed")) & (table["horizon_h"].eq(24))].iloc[0]
    lines = [
        "# Figure 2 response-centric theory draft",
        "",
        "Figure contract:",
        "",
        "- Core conclusion: under agricultural local variability, the transferable forecast object shifts from fixed soil moisture state to local event-response morphology.",
        "- Panel a now states the forecast representation hypothesis explicitly: `S_hat_static(t+h) = F_h(S(t), l(t), q)` versus `S_hat_response(t+h) = G_h(S(t), l(t), q, c_e)`.",
        "- The response-context term `c_e` is defined visually as input class, diagnostic regime, and local response memory.",
        "- Archetype: schematic-led composite with quantitative 2 x 2 evidence panels.",
        "- Main evidence table: `SMDE_CSR_response_centric_2x2_integrated_compact_table.csv`, evidence tier `1_main_conservative_static_recession`.",
        "- Main 24 h on-farm result: delta R2 "
        f"{float(onfarm_24['delta_r2']):.6f}; response/static RMSE ratio {float(onfarm_24['rmse_ratio_response_over_static']):.3f}.",
        "- Caution: on-farm RMSE uses source soil moisture units; cross-setting interpretation emphasizes R2/CCC and within-setting ratios.",
        "- Caution: input-class performance is not separately claimed; current on-farm test support is rain-only dominated.",
    ]
    (OUT / "figure2_response_centric_theory_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs() -> None:
    ensure_out()
    table = load_main_table()
    fig = plt.figure(figsize=(7.25, 4.85))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.18, 1.0],
        left=0.07,
        right=0.985,
        top=0.94,
        bottom=0.16,
        hspace=0.34,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    plot_framework_panel(ax_a)
    plot_delta_r2(ax_b, table)
    plot_rmse_ratio(ax_c, table)

    fig.text(
        0.07,
        0.035,
        "Main 2 x 2 uses the conservative static-recession baseline. FAWN RMSE is in mm; on-farm RMSE uses source soil moisture units.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    save_pub(fig, "fig2_response_centric_theory_draft")
    plt.close(fig)
    write_report(table)
    print(
        {
            "figure_png": str(OUT / "fig2_response_centric_theory_draft.png"),
            "figure_pdf": str(OUT / "fig2_response_centric_theory_draft.pdf"),
            "source_rows": int(len(table)),
        }
    )


if __name__ == "__main__":
    write_outputs()
