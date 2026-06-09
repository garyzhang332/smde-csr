from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
EXP = ANALYSIS / "experiment4c_all_train_rate_forecast" / "source_data"
OUT = ANALYSIS / "forecast_process_schematic"
SOURCE = OUT / "source_data"
SUBMISSION_FIGURES = ROOT / "figures"

PREDICTIONS = EXP / "external_2026_all_train_forecast_predictions.parquet"
POINTS = EXP / "external_2026_detected_smde_points.parquet"
REGIME_PREDICTIONS = EXP / "external_2026_online_regime_predictions_all_train.csv"

EXAMPLE_ORIGIN_ID = "S335_2026_moisture_4in_0080|7.50"
MAIN_MODEL = "local_regime_shrink_rate"

REGIME_ORDER = ["early_transient", "stageI_like", "stageII_like"]
REGIME_LABELS = {
    "early_transient": "Early transient",
    "stageI_like": "Stage I-like",
    "stageII_like": "Stage II-like",
}
PROB_COLS = {
    "early_transient": "prob_early_transient",
    "stageI_like": "prob_stageI_like",
    "stageII_like": "prob_stageII_like",
}
REGIME_COLORS = {
    "early_transient": "#C78D4B",
    "stageI_like": "#4E7E9E",
    "stageII_like": "#6FA38A",
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#626A70",
    "grid": "#E6EBEF",
    "forecast": "#C67B2E",
    "history": "#202020",
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
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    shutil.copyfile(OUT / f"{stem}.pdf", SUBMISSION_FIGURES / "Figure_3_forecast_process_schematic.pdf")


def layer_label(layer: str) -> str:
    return LAYER_LABELS.get(str(layer), str(layer))


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
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


def load_example() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pred_cols = [
        "model",
        "origin_id",
        "site_id",
        "layer",
        "event_id",
        "start",
        "origin_t_h",
        "horizon_h",
        "s0_mm",
        "pred_mm",
        "target_mm",
        "start_mm",
        "drop_sofar_mm",
        "recent_loss_1h",
        "predicted_regime",
        "analog_level",
    ]
    point_cols = ["site_id", "event_id", "layer", "t_h", "moisture_mm", "start", "rain_before_48", "rain_during"]
    predictions = pd.read_parquet(PREDICTIONS, columns=pred_cols)
    points = pd.read_parquet(POINTS, columns=point_cols)
    regime = pd.read_csv(REGIME_PREDICTIONS)

    rows = predictions[
        predictions["origin_id"].eq(EXAMPLE_ORIGIN_ID) & predictions["model"].eq(MAIN_MODEL)
    ].sort_values("horizon_h")
    if rows.empty:
        raise RuntimeError(f"Example origin not found for {EXAMPLE_ORIGIN_ID}.")

    row0 = rows.iloc[0].copy()
    probs = regime[regime["origin_id"].eq(EXAMPLE_ORIGIN_ID)]
    if probs.empty:
        raise RuntimeError(f"Regime probabilities not found for {EXAMPLE_ORIGIN_ID}.")
    for col in probs.columns:
        row0[col] = probs.iloc[0][col]

    event = points[
        points["event_id"].eq(row0["event_id"]) & points["layer"].astype(str).eq(str(row0["layer"]))
    ].sort_values("t_h")
    history = event[event["t_h"] <= float(row0["origin_t_h"]) + 1e-9].copy()
    if history.empty:
        raise RuntimeError("No pre-origin history was available for the selected example.")

    rows.to_csv(SOURCE / "figure6_forecast_process_predictions.csv", index=False)
    history.to_csv(SOURCE / "figure6_forecast_process_pre_origin_history.csv", index=False)
    pd.DataFrame([row0.to_dict()]).to_csv(SOURCE / "figure6_forecast_process_origin_metadata.csv", index=False)
    return rows, history, row0


def expanded_ylim(values: list[np.ndarray], min_span: float = 1.2) -> tuple[float, float]:
    merged = np.concatenate([v[np.isfinite(v)] for v in values if len(v)])
    low = float(np.nanmin(merged))
    high = float(np.nanmax(merged))
    span = max(high - low, min_span)
    center = (high + low) / 2
    return center - span * 0.62, center + span * 0.62


def plot_origin_history(ax: plt.Axes, rows: pd.DataFrame, history: pd.DataFrame, meta: pd.Series) -> None:
    origin_t = float(meta["origin_t_h"])
    selected_color = REGIME_COLORS.get(str(meta["predicted_regime"]), COLORS["forecast"])
    x = history["t_h"].to_numpy(float) - origin_t
    y = history["moisture_mm"].to_numpy(float)
    ax.plot(x, y, color=COLORS["history"], lw=1.35, zorder=3)
    recent = history[history["t_h"].between(origin_t - 1.0, origin_t + 1e-9)].copy()
    if len(recent) >= 2:
        ax.plot(recent["t_h"].to_numpy(float) - origin_t, recent["moisture_mm"], color=selected_color, lw=2.2, zorder=4)
    ax.scatter([0], [float(meta["s0_mm"])], s=34, marker="D", color=selected_color, edgecolor="white", lw=0.5, zorder=6)
    ax.axvline(0, color="#444444", lw=0.85, ls="--", zorder=1)
    ax.text(0.02, 0.93, "forecast origin", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=COLORS["neutral_mid"])
    ax.text(
        0.04,
        0.12,
        f"recent 1 h loss = {float(meta['recent_loss_1h']):.3f} mm h$^{{-1}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=selected_color,
    )
    ax.set_xlim(min(-8.0, float(x.min()) - 0.15), 0.45)
    ax.set_ylim(*expanded_ylim([y], min_span=1.1))
    ax.set_xlabel("Time relative to origin (h)")
    ax.set_ylabel("S (mm)")
    ax.grid(True, color=COLORS["grid"], lw=0.50)
    ax.set_title("Origin information", loc="left", fontsize=8.0, fontweight="bold")
    add_panel_label(ax, "a", y=1.14)


def plot_regime_evidence(ax: plt.Axes, meta: pd.Series) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "b", x=0.0, y=1.00)
    ax.text(0.08, 1.00, "Online regime evidence", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold")

    left = 0.08
    y0 = 0.62
    width_total = 0.84
    height = 0.16
    cursor = left
    for regime in REGIME_ORDER:
        prob = float(meta.get(PROB_COLS[regime], 0.0))
        width = width_total * prob
        ax.add_patch(
            mpl.patches.Rectangle(
                (cursor, y0),
                width,
                height,
                transform=ax.transAxes,
                facecolor=REGIME_COLORS[regime],
                edgecolor="white",
                lw=0.7,
            )
        )
        if width > 0.08:
            ax.text(
                cursor + width / 2,
                y0 + height / 2,
                f"{prob:.0%}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if regime == "stageII_like" else COLORS["neutral_dark"],
                fontweight="bold",
            )
        cursor += width

    legend_y = 0.47
    for i, regime in enumerate(REGIME_ORDER):
        yy = legend_y - i * 0.11
        ax.scatter([0.09], [yy], transform=ax.transAxes, s=30, color=REGIME_COLORS[regime])
        ax.text(0.14, yy, REGIME_LABELS[regime], transform=ax.transAxes, ha="left", va="center", fontsize=6.0)

    selected = str(meta["predicted_regime"])
    ax.text(
        0.55,
        0.43,
        f"selected regime: {REGIME_LABELS.get(selected, selected)}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=REGIME_COLORS.get(selected, COLORS["neutral_dark"]),
    )
    ax.text(
        0.55,
        0.29,
        f"elapsed = {float(meta['origin_t_h']):.1f} h; drop so far = {float(meta['drop_sofar_mm']):.2f} mm",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        color=COLORS["neutral_mid"],
    )
    ax.text(
        0.55,
        0.15,
        "evidence only; no future soil moisture",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=COLORS["neutral_mid"],
    )


def plot_forecast(ax: plt.Axes, rows: pd.DataFrame, history: pd.DataFrame, meta: pd.Series) -> None:
    pred_x = np.r_[0.0, rows["horizon_h"].to_numpy(float)]
    pred_y = np.r_[float(meta["s0_mm"]), rows["pred_mm"].to_numpy(float)]
    selected_color = REGIME_COLORS.get(str(meta["predicted_regime"]), COLORS["forecast"])
    ax.plot(pred_x, pred_y, color=selected_color, lw=1.85, zorder=3)
    ax.scatter(pred_x[1:], pred_y[1:], s=24, color=selected_color, edgecolor="white", lw=0.45, zorder=5)
    ax.scatter([0], [float(meta["s0_mm"])], s=34, marker="D", color=selected_color, edgecolor="white", lw=0.5, zorder=6)
    ax.set_xlim(0, 25)
    ax.set_ylim(*expanded_ylim([history["moisture_mm"].to_numpy(float), pred_y], min_span=1.6))
    ax.grid(True, color=COLORS["grid"], lw=0.50)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Predicted S (mm)")
    ax.set_title("Calibrated forecast output", loc="left", fontsize=8.0, fontweight="bold")
    ax.text(
        0.34,
        0.92,
        r"$L_{recent}(t)=\frac{S(t-1h)-S(t)}{1h}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=COLORS["neutral_dark"],
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.86},
    )
    ax.text(
        0.34,
        0.80,
        r"$r_t=\arg\max_r\,p(r\mid x_t)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=COLORS["neutral_dark"],
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.86},
    )
    ax.text(
        0.34,
        0.68,
        r"$\hat{S}(t+h)=S(t)-\alpha_{h,r_t,l,s}L_{recent}(t)h$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=COLORS["neutral_dark"],
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.86},
    )
    ax.text(
        0.04,
        0.80,
        str(rows["analog_level"].iloc[-1]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=COLORS["neutral_mid"],
    )
    add_panel_label(ax, "c")


def main() -> None:
    ensure_out()
    rows, history, meta = load_example()

    fig = plt.figure(figsize=(7.25, 4.45))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.10, 1.0],
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.80,
        hspace=0.55,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_origin_history(ax_a, rows, history, meta)
    plot_regime_evidence(ax_b, meta)
    plot_forecast(ax_c, rows, history, meta)

    fig.suptitle(
        "Forecast process: origin-state information is converted into a calibrated recent-loss prediction",
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )
    fig.text(
        0.02,
        0.905,
        f"Example origin: FAWN {int(meta['site_id'])}, {layer_label(str(meta['layer']))}, {pd.to_datetime(meta['start']).strftime('%Y-%m-%d')}; observed future soil moisture is intentionally not plotted.",
        ha="left",
        va="top",
        fontsize=6.8,
        color=COLORS["neutral_mid"],
    )
    save_pub(fig, "fig6_forecast_process_schematic")
    plt.close(fig)
    print(OUT / "fig6_forecast_process_schematic.pdf")


if __name__ == "__main__":
    main()
