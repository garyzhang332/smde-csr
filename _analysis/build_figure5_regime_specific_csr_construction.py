from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
EXP3_SOURCE = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data"
OUT = ANALYSIS / "experiment3_regime_curve_construction_figure"
SOURCE = OUT / "source_data"
SUBMISSION_FIGS = ROOT / "figures"
INLINE_FIGS = ROOT / "figures" / "inline_jpg"

SITE_ID = 230
LAYER = "moisture_4in"

REGIME_ORDER = ["early_transient", "stageI_like", "stageII_like"]
REGIME_LABELS = {
    "early_transient": "Early transient",
    "stageI_like": "Stage I-like",
    "stageII_like": "Stage II-like",
}
REGIME_COLORS = {
    "early_transient": "#C78D4B",
    "stageI_like": "#4E7E9E",
    "stageII_like": "#6FA38A",
}
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
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
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
    }
)


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIGS.mkdir(parents=True, exist_ok=True)
    INLINE_FIGS.mkdir(parents=True, exist_ok=True)


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(SUBMISSION_FIGS / "Figure_6_regime_specific_CSR_curve_construction.pdf", bbox_inches="tight")
    fig.savefig(
        INLINE_FIGS / "fig_regime_specific_csr_curve_construction.jpg",
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(EXP3_SOURCE / "adaptive_regime_csr_points.parquet")
    aligned = pd.read_parquet(EXP3_SOURCE / "adaptive_regime_aligned_points.parquet")
    curves = pd.read_csv(EXP3_SOURCE / "adaptive_regime_curves_calibrated.csv")
    edges = pd.read_csv(EXP3_SOURCE / "adaptive_regime_registration_edges.csv")
    summary = pd.read_csv(EXP3_SOURCE / "adaptive_regime_construction_pool_summary.csv")
    for frame in [raw, aligned, curves, edges, summary]:
        if "layer" in frame.columns:
            frame["layer"] = frame["layer"].astype(str)
        if "segment" in frame.columns:
            frame["segment"] = frame["segment"].astype(str)
    return raw, aligned, curves, edges, summary


def local_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if "site_id" not in frame.columns:
        return frame[frame["layer"].eq(LAYER)].copy()
    return frame[frame["site_id"].eq(SITE_ID) & frame["layer"].eq(LAYER)].copy()


def sample_ids(ids: np.ndarray, max_n: int) -> np.ndarray:
    ids = np.asarray(sorted(pd.Series(ids).dropna().astype(str).unique()))
    if len(ids) <= max_n:
        return ids
    idx = np.linspace(0, len(ids) - 1, max_n).round().astype(int)
    return ids[idx]


def binned_band(data: pd.DataFrame, x_col: str, y_col: str, bins: int = 45) -> pd.DataFrame:
    use = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if use.empty or use[x_col].max() <= use[x_col].min():
        return pd.DataFrame()
    edges = np.linspace(float(use[x_col].min()), float(use[x_col].max()), bins + 1)
    use["bin"] = pd.cut(use[x_col], edges, include_lowest=True, duplicates="drop")
    out = (
        use.groupby("bin", observed=True)
        .agg(
            x=(x_col, "median"),
            q10=(y_col, lambda s: float(np.nanquantile(s, 0.10))),
            q50=(y_col, "median"),
            q90=(y_col, lambda s: float(np.nanquantile(s, 0.90))),
            n=(y_col, "size"),
        )
        .dropna()
        .reset_index(drop=True)
    )
    return out[out["n"].ge(max(3, int(out["n"].median() * 0.20)))]


def nice_xlim(max_value: float) -> tuple[float, float]:
    if not np.isfinite(max_value) or max_value <= 0:
        return 0.0, 1.0
    if max_value <= 12:
        step = 2.0
    elif max_value <= 30:
        step = 5.0
    else:
        step = 10.0
    return 0.0, float(np.ceil(max_value / step) * step)


def draw_raw_pool(
    ax: plt.Axes,
    data: pd.DataFrame,
    regime: str,
    y_limits: tuple[float, float],
    x_limits: tuple[float, float],
) -> None:
    color = REGIME_COLORS[regime]
    use = data[data["segment"].eq(regime)].copy()
    if regime == "stageI_like":
        event_summary = (
            use.sort_values("t_h")
            .groupby("event_id", observed=True)
            .agg(
                duration=("t_h", lambda s: float(s.max() - s.min())),
                drop=("moisture_mm", lambda s: float(s.iloc[0] - s.iloc[-1])),
                storage_range=("moisture_mm", lambda s: float(s.max() - s.min())),
            )
        )
        keep_ids = event_summary[
            event_summary["drop"].ge(max(0.10, float(event_summary["drop"].quantile(0.30))))
            & event_summary["storage_range"].ge(max(0.12, float(event_summary["storage_range"].quantile(0.30))))
            & event_summary["duration"].ge(max(1.0, float(event_summary["duration"].quantile(0.10))))
        ].index.to_numpy()
        use = use[use["event_id"].astype(str).isin(pd.Series(keep_ids).astype(str))].copy()
    keep = sample_ids(use["event_id"].to_numpy(), 85)
    display = use[use["event_id"].astype(str).isin(keep)].copy()
    for _, seg in display.groupby("event_id", observed=True, sort=False):
        seg = seg.sort_values("t_h")
        ax.plot(seg["t_h"], seg["moisture_mm"], color=color, alpha=0.42, lw=0.78)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True, color="#E7ECF0", lw=0.45)
    ax.set_title(REGIME_LABELS[regime], color=color, fontweight="bold", fontsize=7.4, pad=6)
    ax.text(
        0.03,
        0.07,
        f"{use['event_id'].nunique():,} segments\n{len(use):,} observations",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#555555",
    )


def draw_registered_curve(
    ax: plt.Axes,
    aligned: pd.DataFrame,
    curves: pd.DataFrame,
    edges: pd.DataFrame,
    regime: str,
    y_limits: tuple[float, float],
    x_limits: tuple[float, float],
) -> None:
    color = REGIME_COLORS[regime]
    use = aligned[aligned["segment"].eq(regime)].copy()
    curve = curves[curves["segment"].eq(regime)].sort_values("csr_x_h").copy()
    keep = sample_ids(use["event_id"].to_numpy(), 70)
    display = use[use["event_id"].astype(str).isin(keep)].copy()
    bg_alpha = 0.34 if regime == "stageI_like" else 0.30
    bg_lw = 0.76 if regime == "stageI_like" else 0.68
    for _, seg in display.groupby("event_id", observed=True, sort=False):
        seg = seg.sort_values("csr_x_h")
        ax.plot(seg["csr_x_h"], seg["moisture_mm"], color=color, alpha=bg_alpha, lw=bg_lw, zorder=1)

    guide = binned_band(use, "csr_x_h", "moisture_mm", bins=60)
    if not guide.empty:
        x = guide["x"].to_numpy(float)
        y = guide["q50"].to_numpy(float)
        n = guide["n"].to_numpy(float) if "n" in guide.columns else np.ones_like(y)
        if len(y) >= 4:
            degree = 2 if regime == "early_transient" else 3
            degree = min(degree, len(y) - 1)
            weights = np.sqrt(np.maximum(n, 1.0))
            coeff = np.polyfit(x, y, degree, w=weights)
            x_plot = np.linspace(float(x.min()), float(x.max()), 140)
            y_plot = np.polyval(coeff, x_plot)
            y_plot = pd.Series(y_plot).rolling(7, center=True, min_periods=1).mean().to_numpy(float)
            y_plot = np.minimum.accumulate(y_plot)
        else:
            x_plot = x
            y_plot = y
        ax.plot(
            x_plot,
            y_plot,
            color=color,
            lw=2.25,
            zorder=6,
            path_effects=[pe.Stroke(linewidth=3.20, foreground="white", alpha=0.85), pe.Normal()],
        )
    elif not curve.empty:
        ax.plot(
            curve["csr_x_h"],
            curve["csr_mm"],
            color=color,
            lw=2.25,
            zorder=6,
            path_effects=[pe.Stroke(linewidth=3.20, foreground="white", alpha=0.85), pe.Normal()],
        )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True, color="#E7ECF0", lw=0.45)
    edge_count = int(len(edges[edges["segment"].eq(regime)])) if not edges.empty else 0
    ax.text(
        0.03,
        0.07,
        f"{use['event_id'].nunique():,} aligned segments\n{edge_count:,} pairwise links",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#555555",
    )


def build_summary(
    raw: pd.DataFrame,
    aligned: pd.DataFrame,
    curves: pd.DataFrame,
    edges: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    local_raw = local_filter(raw)
    local_aligned = local_filter(aligned)
    local_curves = local_filter(curves)
    local_edges = local_filter(edges)
    network = summary[summary["layer"].eq(LAYER)]
    for regime in REGIME_ORDER:
        rows.append(
            {
                "site_id": SITE_ID,
                "layer": LAYER,
                "regime": regime,
                "raw_segments_site_layer": int(local_raw[local_raw["segment"].eq(regime)]["event_id"].nunique()),
                "raw_points_site_layer": int(len(local_raw[local_raw["segment"].eq(regime)])),
                "aligned_segments_site_layer": int(local_aligned[local_aligned["segment"].eq(regime)]["event_id"].nunique()),
                "aligned_points_site_layer": int(len(local_aligned[local_aligned["segment"].eq(regime)])),
                "curve_points_site_layer": int(len(local_curves[local_curves["segment"].eq(regime)])),
                "pairwise_links_site_layer": int(len(local_edges[local_edges["segment"].eq(regime)])),
                "network_segments_layer": int(network[network["segment"].eq(regime)]["segments"].sum()),
                "network_points_layer": int(network[network["segment"].eq(regime)]["points"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_out()
    raw, aligned, curves, edges, summary = load_inputs()
    raw_local = local_filter(raw)
    aligned_local = local_filter(aligned)
    curves_local = local_filter(curves)
    edges_local = local_filter(edges)
    source_summary = build_summary(raw, aligned, curves, edges, summary)
    source_summary.to_csv(SOURCE / "figure5_regime_specific_csr_construction_summary.csv", index=False)

    y_values = pd.concat(
        [
            raw_local[raw_local["segment"].isin(REGIME_ORDER)]["moisture_mm"],
            aligned_local[aligned_local["segment"].isin(REGIME_ORDER)]["moisture_mm"],
        ],
        ignore_index=True,
    ).to_numpy(float)
    y_min = float(np.nanquantile(y_values, 0.005))
    y_max = float(np.nanquantile(y_values, 0.995))
    y_pad = max((y_max - y_min) * 0.08, 0.4)
    y_limits = (max(0.0, y_min - y_pad), y_max + y_pad)
    raw_x_max = float(raw_local[raw_local["segment"].isin(REGIME_ORDER)]["t_h"].max())
    registered_x_max = float(aligned_local[aligned_local["segment"].isin(REGIME_ORDER)]["csr_x_h"].max())
    x_limits = nice_xlim(max(raw_x_max, registered_x_max))

    fig = plt.figure(figsize=(7.25, 5.25))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.775,
        wspace=0.22,
        hspace=0.72,
    )
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for i, regime in enumerate(REGIME_ORDER):
        draw_raw_pool(axes_top[i], raw_local, regime, y_limits, x_limits)
        draw_registered_curve(axes_bottom[i], aligned_local, curves_local, edges_local, regime, y_limits, x_limits)
        if i > 0:
            axes_top[i].set_yticklabels([])
            axes_bottom[i].set_yticklabels([])
        else:
            axes_top[i].set_ylabel("Soil water amount (mm)")
            axes_bottom[i].set_ylabel("Soil water amount (mm)")
        axes_top[i].set_xlabel("Time within parent SMDE (h)")
        axes_bottom[i].set_xlabel("Registered CSR coordinate (h)")

    fig.suptitle(
        "Regime-specific CSR curve construction from many SMDE segments",
        x=0.02,
        y=0.955,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.910,
        f"Representative local library: FAWN {SITE_ID}, {LAYER_LABELS[LAYER]}. Top row keeps segments in original SMDE time; bottom row shows hydrologically registered training segments and fitted CSR curves.",
        ha="left",
        va="top",
        fontsize=6.7,
        color="#555555",
    )
    fig.text(
        0.075,
        0.840,
        "a  Raw adaptive segments retained at their within-SMDE timing",
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.818,
        "Segment starts are diagnosed regime-transition times, not reset origins.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#666666",
    )
    fig.text(
        0.075,
        0.395,
        "b  Hydrologically registered observations and fitted regime-specific CSR curves",
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
    )
    save_pub(fig, "fig_regime_specific_csr_curve_construction")
    plt.close(fig)
    print(f"Wrote regime-specific CSR construction figure to {OUT}")


if __name__ == "__main__":
    main()
