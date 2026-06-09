from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
EXP = ANALYSIS / "experiment4c_all_train_rate_forecast" / "source_data"
OUT = ANALYSIS / "experiment4c_all_train_rate_forecast_figure"
SOURCE = OUT / "source_data"
SUBMISSION_FIGURES = ROOT / "figures"

PREDICTIONS = EXP / "external_2026_all_train_forecast_predictions.parquet"
POINTS = EXP / "external_2026_detected_smde_points.parquet"
METRICS = EXP / "external_2026_all_train_forecast_metrics.csv"
COEFFICIENTS = EXP / "all_train_rate_calibration_coefficients.csv"

MAIN_MODEL = "local_regime_shrink_rate"
MAIN_LABEL = "Local-regime shrink"
EXAMPLE_SITE = 405
EXAMPLE_ORIGINS = {
    "moisture_4in": "S405_2026_moisture_4in_0059|1.00",
    "moisture_8in": "S405_2026_moisture_8in_0066|1.00",
    "moisture_12in": "S405_2026_moisture_12in_0057|5.50",
    "moisture_16in": "S405_2026_moisture_16in_0058|7.00",
    "moisture_20in": "S405_2026_moisture_20in_0077|31.50",
}

LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
REGIME_LABELS = {
    "early_transient": "Early transient",
    "stageI_like": "Stage I-like",
    "stageII_like": "Stage II-like",
    "mixed_uncertain": "Mixed/uncertain",
}
REGIME_ORDER = ["early_transient", "stageI_like", "stageII_like"]
REGIME_COLORS = {
    "early_transient": "#C78D4B",
    "stageI_like": "#4E7E9E",
    "stageII_like": "#6FA38A",
    "mixed_uncertain": "#9EA3A8",
}
MODEL_STYLE = {
    "local_regime_shrink_rate": ("#C67B2E", MAIN_LABEL),
    "horizon_calibrated_rate": ("#7A7A7A", "Horizon calibrated"),
    "layer_regime_calibrated_rate": ("#8A9C78", "Layer-regime calibrated"),
    "regime_calibrated_rate": ("#6FA38A", "Regime calibrated"),
    "persistence": ("#4E7E9E", "Persistence"),
    "recent_slope": ("#A65D75", "Recent slope"),
    "nonregime_analog": ("#9083B8", "Non-regime analog"),
    "regime_mixture_analog": ("#7E8E9A", "Regime-mixture analog"),
    "online_regime_analog": ("#6D759B", "Online-regime analog"),
    "registered_csr_operator": ("#B23A48", "Registered CSR"),
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#6F7378",
    "grid": "#E6EBEF",
    "forecast": "#C67B2E",
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
    shutil.copyfile(OUT / f"{stem}.pdf", SUBMISSION_FIGURES / "Figure_7_calibrated_recent_loss_forecast_validation.pdf")


def layer_label(layer: str) -> str:
    return LAYER_LABELS.get(str(layer), str(layer))


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_cols = [
        "model",
        "site_id",
        "layer",
        "event_id",
        "origin_id",
        "start",
        "origin_t_h",
        "horizon_h",
        "target_t_h",
        "s0_mm",
        "target_mm",
        "pred_mm",
        "recent_loss_1h",
        "predicted_regime",
    ]
    point_cols = ["site_id", "event_id", "layer", "t_h", "moisture_mm", "start", "end"]
    predictions = pd.read_parquet(PREDICTIONS, columns=pred_cols)
    points = pd.read_parquet(POINTS, columns=point_cols)
    metrics = pd.read_csv(METRICS)
    coefs = pd.read_csv(COEFFICIENTS)
    for frame in [predictions, points]:
        frame["layer"] = frame["layer"].astype(str)
    predictions["start"] = pd.to_datetime(predictions["start"], errors="coerce")
    points["start"] = pd.to_datetime(points["start"], errors="coerce")
    return predictions, points, metrics, coefs


def expanded_ylim(values: list[np.ndarray], min_span: float = 3.0) -> tuple[float, float]:
    arr = np.concatenate([v[np.isfinite(v)] for v in values if len(v)]) if values else np.array([])
    if arr.size == 0:
        return 0.0, min_span
    ymin = float(np.nanmin(arr))
    ymax = float(np.nanmax(arr))
    span = max(ymax - ymin, min_span)
    center = 0.5 * (ymin + ymax)
    ymin = max(0.0, center - span / 2 - 0.12 * span)
    ymax = center + span / 2 + 0.12 * span
    return ymin, ymax


def plot_multidepth_examples(fig: plt.Figure, grid, predictions: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    axes = [fig.add_subplot(grid[i, 0]) for i in range(len(LAYER_ORDER))]
    model_pred = predictions[predictions["model"].eq(MAIN_MODEL)].copy()
    selected = []
    for ax, layer in zip(axes, LAYER_ORDER):
        origin_id = EXAMPLE_ORIGINS[layer]
        rows = model_pred[model_pred["origin_id"].eq(origin_id)].sort_values("horizon_h").copy()
        if rows.empty:
            ax.text(0.5, 0.5, f"No example for {layer_label(layer)}", ha="center", va="center", transform=ax.transAxes)
            continue
        row0 = rows.iloc[0]
        event = points[
            points["event_id"].eq(row0["event_id"]) & points["layer"].eq(layer)
        ].sort_values("t_h").copy()
        origin_t = float(row0["origin_t_h"])
        max_h = 24.0
        origin_trace = (
            model_pred[
                model_pred["event_id"].eq(row0["event_id"])
                & model_pred["layer"].eq(layer)
                & model_pred["origin_t_h"].between(origin_t, origin_t + max_h, inclusive="both")
            ][["origin_id", "origin_t_h", "predicted_regime"]]
            .drop_duplicates("origin_id")
            .sort_values("origin_t_h")
        )
        if not origin_trace.empty:
            rel_t = origin_trace["origin_t_h"].to_numpy(float) - origin_t
            regimes = origin_trace["predicted_regime"].astype(str).to_numpy()
            diffs = np.diff(rel_t)
            step = float(np.nanmedian(diffs)) if len(diffs) and np.isfinite(diffs).any() else 0.5
            step = min(max(step, 0.25), 1.0)
            start = max(0.0, float(rel_t[0]))
            current = str(regimes[0])
            for idx in range(1, len(rel_t)):
                if str(regimes[idx]) == current:
                    continue
                end = min(max_h, max(start, float(rel_t[idx])))
                ax.axvspan(start, end, color=REGIME_COLORS.get(current, REGIME_COLORS["mixed_uncertain"]), alpha=0.105, lw=0, zorder=0)
                start = min(max_h, float(rel_t[idx]))
                current = str(regimes[idx])
            end = min(max_h, float(rel_t[-1]) + step)
            ax.axvspan(start, end, color=REGIME_COLORS.get(current, REGIME_COLORS["mixed_uncertain"]), alpha=0.105, lw=0, zorder=0)
        obs = event[event["t_h"].between(origin_t, origin_t + max_h + 1e-9)].copy()
        obs_x = obs["t_h"].to_numpy(float) - origin_t
        obs_y = obs["moisture_mm"].to_numpy(float)
        pred_x = np.r_[0.0, rows["horizon_h"].to_numpy(float)]
        pred_y = np.r_[float(row0["s0_mm"]), rows["pred_mm"].to_numpy(float)]
        target_x = rows["horizon_h"].to_numpy(float)
        target_y = rows["target_mm"].to_numpy(float)
        forecast_y = rows["pred_mm"].to_numpy(float)
        ax.plot(obs_x, obs_y, color="#222222", lw=1.20, label="Observed SMDE", zorder=2)
        regime = str(row0["predicted_regime"])
        regime_color = REGIME_COLORS.get(regime, COLORS["forecast"])
        ax.plot(pred_x, pred_y, color=regime_color, lw=1.65, label="Forecast", zorder=3)
        ax.scatter(target_x, target_y, s=14, facecolor="white", edgecolor="#222222", lw=0.65, zorder=5)
        ax.scatter(target_x, forecast_y, s=14, color=regime_color, edgecolor="white", lw=0.35, zorder=6)
        ax.scatter([0], [float(row0["s0_mm"])], s=20, marker="D", color=regime_color, edgecolor="white", lw=0.35, zorder=7)
        residual = rows["pred_mm"].to_numpy(float) - rows["target_mm"].to_numpy(float)
        rmse = float(np.sqrt(np.mean(residual**2)))
        ax.text(
            0.98,
            0.78,
            f"RMSE = {rmse:.2f} mm",
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=5.8,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#D8DEE3", "lw": 0.45, "alpha": 0.9},
        )
        ax.text(
            0.50,
            0.86,
            layer_label(layer),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=5.9,
            fontweight="bold",
            color=COLORS["neutral_mid"],
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.78},
        )
        ax.set_xlim(0, 25)
        ax.set_ylim(*expanded_ylim([obs_y, pred_y, target_y], min_span=3.4))
        ax.grid(True, color=COLORS["grid"], lw=0.50)
        ax.set_ylabel("S (mm)", fontsize=6.1)
        ax.tick_params(labelsize=5.7)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time since forecast origin (h)")
        selected.append(
            {
                "site_id": EXAMPLE_SITE,
                "layer": layer,
                "event_id": row0["event_id"],
                "origin_id": origin_id,
                "start": row0["start"],
                "origin_t_h": origin_t,
                "online_regime": regime,
                "window_regimes": ">".join(origin_trace["predicted_regime"].dropna().astype(str).drop_duplicates().tolist())
                if not origin_trace.empty
                else regime,
                "target_drop_24h_mm": float(row0["s0_mm"]) - float(rows.loc[rows["horizon_h"].eq(24), "target_mm"].iloc[0])
                if rows["horizon_h"].eq(24).any()
                else np.nan,
                "rmse_mm": rmse,
            }
        )
    axes[0].set_title(
        f"a  2026 external validation examples, FAWN {EXAMPLE_SITE}",
        loc="left",
        fontsize=8.0,
        fontweight="bold",
        pad=4,
    )
    top_pos = axes[0].get_position()
    legend_handles = [
        Line2D([0], [0], color="#222222", lw=1.20, label="Observed S"),
        Line2D([0], [0], color=REGIME_COLORS["early_transient"], lw=1.65, label="Early transient"),
        Line2D([0], [0], color=REGIME_COLORS["stageI_like"], lw=1.65, label="Stage I-like"),
        Line2D([0], [0], color=REGIME_COLORS["stageII_like"], lw=1.65, label="Stage II-like"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(top_pos.x1, min(0.940, top_pos.y1 + 0.078)),
        ncol=4,
        fontsize=5.25,
        handlelength=1.15,
        columnspacing=0.55,
        handletextpad=0.35,
        borderpad=0.18,
        labelspacing=0.25,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="#D8DEE3",
    )
    selected_df = pd.DataFrame(selected)
    selected_df.to_csv(SOURCE / "figure7_panel_a_selected_2026_examples.csv", index=False)
    return selected_df


def station_layer_rmse(predictions: pd.DataFrame) -> pd.DataFrame:
    model_pred = predictions[predictions["model"].eq(MAIN_MODEL)].dropna(subset=["pred_mm", "target_mm"]).copy()
    rows = []
    for (site_id, layer), group in model_pred.groupby(["site_id", "layer"], observed=True):
        if layer not in LAYER_ORDER:
            continue
        residual = group["pred_mm"].to_numpy(float) - group["target_mm"].to_numpy(float)
        rows.append(
            {
                "site_id": int(site_id),
                "layer": str(layer),
                "forecasts": int(len(group)),
                "events": int(group["event_id"].nunique()),
                "rmse_mm": float(np.sqrt(np.mean(residual**2))),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["forecasts"].ge(50)].copy()
    complete = out.groupby("site_id").filter(lambda g: set(g["layer"]) == set(LAYER_ORDER))
    complete = complete.sort_values(["site_id", "layer"]).reset_index(drop=True)
    complete.to_csv(SOURCE / "figure7_panel_b_station_layer_rmse.csv", index=False)
    return complete


def plot_heatmap(ax: plt.Axes, site_layer: pd.DataFrame) -> None:
    sites = sorted(site_layer["site_id"].dropna().astype(int).unique())
    matrix = np.full((len(sites), len(LAYER_ORDER)), np.nan)
    for i, site_id in enumerate(sites):
        for j, layer in enumerate(LAYER_ORDER):
            cell = site_layer[(site_layer["site_id"].eq(site_id)) & (site_layer["layer"].eq(layer))]
            if not cell.empty:
                matrix[i, j] = float(cell["rmse_mm"].iloc[0])
    vmax = max(0.65, float(np.nanpercentile(matrix, 95))) if np.isfinite(matrix).any() else 1.0
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad("#EFEFEF")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(f"b  2026 station-layer RMSE\n{MAIN_LABEL} forecast", loc="left", fontsize=8.2, fontweight="bold")
    ax.set_xticks(np.arange(len(LAYER_ORDER)))
    ax.set_xticklabels([layer_label(l) for l in LAYER_ORDER])
    ax.set_yticks(np.arange(len(sites)))
    ax.set_yticklabels([str(s) for s in sites], fontsize=5.1)
    ax.set_xlabel("Sensor depth")
    ax.set_ylabel("FAWN station")
    ax.set_xticks(np.arange(-0.5, len(LAYER_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sites), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.50)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.075)
    cbar.ax.set_title("RMSE\n(mm)", fontsize=5.8, pad=4)
    for i, site_id in enumerate(sites):
        for j, layer in enumerate(LAYER_ORDER):
            value = matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if value > 0.62 * vmax else "#1F1F1F"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.0, color=text_color)


def plot_model_comparison(ax: plt.Axes, metrics: pd.DataFrame) -> pd.DataFrame:
    models = [
        "local_regime_shrink_rate",
        "horizon_calibrated_rate",
        "layer_regime_calibrated_rate",
        "regime_calibrated_rate",
        "persistence",
        "recent_slope",
        "nonregime_analog",
        "regime_mixture_analog",
        "registered_csr_operator",
    ]
    sub = metrics[(metrics["comparison_set"].eq("all_common_origin")) & (metrics["model"].isin(models))].copy()
    horizons = [1, 3, 6, 12, 24]
    pivot = sub.pivot(index="model", columns="horizon_h", values="rmse_mm").reindex(index=models, columns=horizons)
    matrix = pivot.to_numpy(dtype=float)
    cmap = mpl.colormaps["YlOrBr"].copy()
    cmap.set_bad("#F1F1F1")
    norm = mpl.colors.LogNorm(vmin=0.045, vmax=6.0)
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=norm)
    ax.set_title("c  External-validation model comparison", loc="left", fontsize=8.0, fontweight="bold")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(horizons)), [str(h) for h in horizons])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_STYLE[m][1] for m in models], fontsize=5.9)
    for label, model in zip(ax.get_yticklabels(), models):
        if model == MAIN_MODEL:
            label.set_fontweight("bold")
            label.set_color(COLORS["neutral_dark"])
    ax.set_xticks(np.arange(-0.5, len(horizons), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.add_patch(
        patches.Rectangle(
            (-0.5, -0.5),
            len(horizons),
            1.0,
            fill=False,
            edgecolor=COLORS["neutral_dark"],
            linewidth=1.15,
            zorder=8,
            clip_on=False,
        )
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if val > 1.0 else "#222222",
                    fontweight="bold" if models[i] == MAIN_MODEL else "normal",
                )
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.018)
    cbar.set_label("RMSE (mm, log color)")
    sub.to_csv(SOURCE / "figure7_panel_c_model_comparison.csv", index=False)
    return sub


def write_report(selected: pd.DataFrame, comparison: pd.DataFrame, site_layer: pd.DataFrame) -> None:
    core = comparison[comparison["model"].eq(MAIN_MODEL)].sort_values("horizon_h")
    report = [
        "# Figure 7 source report",
        "",
        f"Main model: `{MAIN_MODEL}`.",
        "",
        "The figure uses the all-2023-2025 training and 2026 external-validation experiment.",
        "",
        "## Main model horizon metrics",
        "",
        core[["horizon_h", "forecasts", "sites", "events", "rmse_mm", "mae_mm", "bias_mm", "ccc"]].to_markdown(index=False),
        "",
        "## Panel a selected examples",
        "",
        selected.to_markdown(index=False) if not selected.empty else "_No examples selected._",
        "",
        "## Panel b station-layer summary",
        "",
        f"Complete station-layer cells shown: {len(site_layer)}",
    ]
    (OUT / "figure7_calibrated_recent_loss_forecast_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_out()
    predictions, points, metrics, coefs = load_inputs()

    fig = plt.figure(figsize=(7.55, 9.35))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[5.95, 2.60],
        width_ratios=[1.46, 1.08],
        left=0.075,
        right=0.975,
        bottom=0.065,
        top=0.905,
        hspace=0.24,
        wspace=0.32,
    )
    a_grid = gs[0, 0].subgridspec(len(LAYER_ORDER), 1, hspace=0.18)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    selected = plot_multidepth_examples(fig, a_grid, predictions, points)
    site_layer = station_layer_rmse(predictions)
    plot_heatmap(ax_b, site_layer)
    comparison = plot_model_comparison(ax_c, metrics)

    fig.suptitle(
        "Calibrated recent-loss forecasting from regime-diagnosed SMDEs",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )
    fig.text(
        0.02,
        0.955,
        "All 2023-2025 clean rainfall-associated SMDE origins train the damping coefficients; 2026 SMDEs are used only for external scoring.",
        ha="left",
        va="top",
        fontsize=7.1,
        color=COLORS["neutral_mid"],
    )
    save_pub(fig, "fig7_calibrated_recent_loss_forecast_validation")
    plt.close(fig)
    write_report(selected, comparison, site_layer)
    print(OUT / "fig7_calibrated_recent_loss_forecast_validation.pdf")


if __name__ == "__main__":
    main()
