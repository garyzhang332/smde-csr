from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fawn_segmented_csr import (
    anchor_interpolator,
    align_by_state_stitch,
    concordance_correlation_coefficient,
    curve_predict,
    fit_segment_curve,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT
SEGCSR = ANALYSIS / "fawn_segmented_csr"
OUT = ANALYSIS / "experiment3_localized_segmented_csr"
SOURCE = OUT / "source_data"

MAIN_SUBSET = "clean_stageII_48h"
RANDOM_SEED = 20260605
TEST_FRACTION = 0.20
MIN_TRAIN_EVENTS = 8
MIN_TEST_EVENTS = 2
MIN_TRAIN_POINTS = 40
MIN_TEST_POINTS = 8

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
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
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


def label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "layer" in out.columns:
        out["layer_label"] = out["layer"].map(LAYER_LABELS)
    if "segment" in out.columns:
        out["segment_label"] = out["segment"].map(SEGMENT_LABELS)
        out["segment_short"] = out["segment"].map(SEGMENT_SHORT)
    return out


def deterministic_seed(site_id: int, layer: str, segment: str) -> int:
    text = f"{site_id}-{layer}-{segment}-{RANDOM_SEED}"
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % (2**32 - 1)
    return value


def split_event_ids(event_ids: np.ndarray, site_id: int, layer: str, segment: str) -> tuple[set[str], set[str]] | None:
    event_ids = np.array(sorted(event_ids), dtype=object)
    n_events = len(event_ids)
    if n_events < MIN_TRAIN_EVENTS + MIN_TEST_EVENTS:
        return None
    proposed_test = max(MIN_TEST_EVENTS, int(np.ceil(TEST_FRACTION * n_events)))
    max_test = n_events - MIN_TRAIN_EVENTS
    if max_test < MIN_TEST_EVENTS:
        return None
    n_test = min(proposed_test, max_test)
    rng = np.random.default_rng(deterministic_seed(site_id, layer, segment))
    shuffled = event_ids.copy()
    rng.shuffle(shuffled)
    test_ids = set(str(x) for x in shuffled[:n_test])
    train_ids = set(str(x) for x in shuffled[n_test:])
    return train_ids, test_ids


def align_test_to_training(test_segment: pd.DataFrame, train_aligned: pd.DataFrame) -> pd.DataFrame:
    interp = anchor_interpolator(train_aligned)
    if interp is None:
        return pd.DataFrame()
    anchor_y, anchor_x = interp
    parts = []
    train_max = float(train_aligned["csr_x_h"].max())
    for event_id, event in test_segment.groupby("event_id", sort=False):
        event = event.sort_values("segment_t_h").copy()
        if len(event) < 4:
            continue
        t = event["segment_t_h"].to_numpy(dtype=float)
        y = event["moisture_mm"].to_numpy(dtype=float)
        overlap = (y >= anchor_y.min()) & (y <= anchor_y.max())
        if overlap.sum() >= 3:
            matched_x = np.interp(y[overlap], anchor_y, anchor_x)
            offset = float(np.nanmedian(matched_x - t[overlap]))
        elif np.nanmax(y) < anchor_y.min():
            step = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.25
            offset = train_max + step
        else:
            offset = 0.0
        event["csr_x_h"] = event["segment_t_h"] + offset
        parts.append(event)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 3:
        return {
            "ccc": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "rmse_mm": np.nan,
            "mae_mm": np.nan,
            "bias_mm": np.nan,
            "nrmse_pct_range": np.nan,
        }
    residual = pred - obs
    sse = float(np.sum(residual**2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    value_range = float(np.nanmax(obs) - np.nanmin(obs))
    pearson_r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    return {
        "ccc": concordance_correlation_coefficient(obs, pred),
        "r2": 1.0 - sse / sst if sst > 0 else np.nan,
        "pearson_r": pearson_r,
        "rmse_mm": float(np.sqrt(np.nanmean(residual**2))),
        "mae_mm": float(np.nanmean(np.abs(residual))),
        "bias_mm": float(np.nanmean(residual)),
        "nrmse_pct_range": float(np.sqrt(np.nanmean(residual**2)) / value_range * 100.0) if value_range > 0 else np.nan,
    }


def dynamic_mae(data: pd.DataFrame) -> float:
    errors = []
    for _, event in data.sort_values(["event_id", "t_h"]).groupby("event_id"):
        if len(event) < 2:
            continue
        obs = event["moisture_mm"].to_numpy(dtype=float)
        pred = event["pred_mm"].to_numpy(dtype=float)
        errors.extend(((pred[:-1] - pred[1:]) - (obs[:-1] - obs[1:])).tolist())
    return float(np.nanmean(np.abs(errors))) if errors else np.nan


def run_split_validation(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local_rows = []
    prediction_parts = []
    curve_parts = []
    skipped_rows = []

    grouped = points.groupby(["site_id", "layer", "segment"], sort=True)
    for (site_id, layer, segment), group in grouped:
        group = group.sort_values(["event_id", "segment_t_h"]).copy()
        split = split_event_ids(group["event_id"].unique(), int(site_id), str(layer), str(segment))
        if split is None:
            skipped_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "reason": "too_few_events_for_80_20_with_min_train_test",
                    "events": int(group["event_id"].nunique()),
                    "points": int(len(group)),
                }
            )
            continue
        train_ids, test_ids = split
        train = group[group["event_id"].astype(str).isin(train_ids)].copy()
        test = group[group["event_id"].astype(str).isin(test_ids)].copy()
        if train["event_id"].nunique() < MIN_TRAIN_EVENTS or len(train) < MIN_TRAIN_POINTS:
            skipped_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "reason": "too_few_training_points",
                    "events": int(group["event_id"].nunique()),
                    "points": int(len(group)),
                }
            )
            continue
        if test["event_id"].nunique() < MIN_TEST_EVENTS or len(test) < MIN_TEST_POINTS:
            skipped_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "reason": "too_few_test_points",
                    "events": int(group["event_id"].nunique()),
                    "points": int(len(group)),
                }
            )
            continue

        train_aligned = align_by_state_stitch(train)
        if train_aligned.empty or train_aligned["event_id"].nunique() < MIN_TRAIN_EVENTS:
            skipped_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "reason": "training_alignment_failed",
                    "events": int(group["event_id"].nunique()),
                    "points": int(len(group)),
                }
            )
            continue
        curve, _ = fit_segment_curve(train_aligned)
        if curve.empty:
            skipped_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "reason": "training_curve_failed",
                    "events": int(group["event_id"].nunique()),
                    "points": int(len(group)),
                }
            )
            continue

        test_aligned = align_test_to_training(test, train_aligned)
        if test_aligned.empty:
            skipped_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "reason": "test_alignment_failed",
                    "events": int(group["event_id"].nunique()),
                    "points": int(len(group)),
                }
            )
            continue
        test_aligned["pred_mm"] = curve_predict(curve, test_aligned["csr_x_h"].to_numpy(dtype=float))
        test_aligned["split"] = "test"
        test_aligned["site_id"] = int(site_id)
        test_aligned["layer"] = layer
        test_aligned["segment"] = segment

        obs = test_aligned["moisture_mm"].to_numpy(dtype=float)
        pred = test_aligned["pred_mm"].to_numpy(dtype=float)
        metric = metrics(obs, pred)
        local_rows.append(
            {
                "site_id": int(site_id),
                "layer": layer,
                "segment": segment,
                "train_events": int(len(train_ids)),
                "test_events": int(test_aligned["event_id"].nunique()),
                "train_points": int(len(train_aligned)),
                "test_points": int(len(test_aligned)),
                "test_fraction_events": float(test_aligned["event_id"].nunique() / group["event_id"].nunique()),
                "dynamic_mae_mm_step": dynamic_mae(test_aligned),
                **metric,
            }
        )

        curve = curve.copy()
        curve["site_id"] = int(site_id)
        curve["layer"] = layer
        curve["segment"] = segment
        curve["train_events"] = int(len(train_ids))
        curve["test_events"] = int(test_aligned["event_id"].nunique())

        prediction_parts.append(
            test_aligned[
                [
                    "site_id",
                    "event_id",
                    "layer",
                    "segment",
                    "t_h",
                    "segment_t_h",
                    "event_storage_norm",
                    "csr_x_h",
                    "moisture_mm",
                    "pred_mm",
                    "split",
                ]
            ]
        )
        curve_parts.append(curve)

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    local_metrics = pd.DataFrame(local_rows)
    skipped = pd.DataFrame(skipped_rows)
    return predictions, local_metrics, curves, skipped


def summarize_pooled(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for keys, group in predictions.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        obs = group["moisture_mm"].to_numpy(dtype=float)
        pred = group["pred_mm"].to_numpy(dtype=float)
        row.update(
            {
                "sites": int(group["site_id"].nunique()),
                "test_events": int(group["event_id"].nunique()),
                "test_points": int(len(group)),
                "dynamic_mae_mm_step": dynamic_mae(group),
                **metrics(obs, pred),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_local(local_metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if local_metrics.empty:
        return pd.DataFrame()
    summary = (
        local_metrics.groupby(by, dropna=False)
        .agg(
            local_models=("site_id", "count"),
            median_train_events=("train_events", "median"),
            median_test_events=("test_events", "median"),
            total_test_points=("test_points", "sum"),
            median_ccc=("ccc", "median"),
            median_r2=("r2", "median"),
            median_rmse_mm=("rmse_mm", "median"),
            median_mae_mm=("mae_mm", "median"),
            median_bias_mm=("bias_mm", "median"),
            median_nrmse_pct_range=("nrmse_pct_range", "median"),
        )
        .reset_index()
    )
    return summary


def build_accuracy_tables(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predictions, local_metrics, curves, skipped = run_split_validation(points)
    pooled_layer = summarize_pooled(predictions, ["layer"])
    pooled_layer_segment = summarize_pooled(predictions, ["layer", "segment"])
    local_layer = summarize_local(local_metrics, ["layer"])
    local_layer_segment = summarize_local(local_metrics, ["layer", "segment"])

    tables = {
        "experiment3_split_validation_predictions": label_columns(predictions),
        "experiment3_split_validation_train_curves": label_columns(curves),
        "experiment3_split_validation_local_metrics": label_columns(local_metrics),
        "experiment3_split_validation_skipped_models": label_columns(skipped),
        "experiment3_split_validation_pooled_by_layer": label_columns(pooled_layer),
        "experiment3_split_validation_pooled_by_layer_segment": label_columns(pooled_layer_segment),
        "experiment3_split_validation_local_summary_by_layer": label_columns(local_layer),
        "experiment3_split_validation_local_summary_by_layer_segment": label_columns(local_layer_segment),
    }

    for name, table in tables.items():
        if name.endswith("predictions"):
            table.to_parquet(SOURCE / f"{name}.parquet", index=False)
            compact = table.sample(min(5000, len(table)), random_state=RANDOM_SEED) if len(table) > 5000 else table
            compact.to_csv(SOURCE / f"{name}_preview.csv", index=False)
        elif name.endswith("train_curves"):
            table.to_csv(SOURCE / f"{name}.csv", index=False)
        else:
            table.to_csv(SOURCE / f"{name}.csv", index=False)

    return tables


def plot_accuracy_figure(tables: dict[str, pd.DataFrame]) -> None:
    pooled = tables["experiment3_split_validation_pooled_by_layer_segment"].copy()
    preds = tables["experiment3_split_validation_predictions"].copy()
    if pooled.empty or preds.empty:
        return

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.35)
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_ccc = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, :])

    x = np.arange(len(LAYER_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(SEGMENT_ORDER))
    for offset, segment in zip(offsets, SEGMENT_ORDER):
        panel = pooled[pooled["segment"] == segment].set_index("layer").reindex(LAYER_ORDER)
        ax_rmse.bar(
            x + offset,
            panel["rmse_mm"],
            width=width,
            color=SEGMENT_COLORS[segment],
            alpha=0.92,
            label=SEGMENT_SHORT[segment],
        )
        ax_ccc.plot(
            x + offset,
            panel["ccc"],
            marker="o",
            ms=4.0,
            lw=1.2,
            color=SEGMENT_COLORS[segment],
            label=SEGMENT_SHORT[segment],
        )

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_rmse.set_ylabel("Held-out RMSE (mm)")
    ax_rmse.set_title("80/20 held-out error by depth")
    ax_rmse.grid(axis="y", color=COLORS["neutral_light"], lw=0.6)
    ax_rmse.legend(fontsize=6.2, loc="upper right", handlelength=1.2)
    add_panel_label(ax_rmse, "a")

    ax_ccc.set_xticks(x)
    ax_ccc.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_ccc.set_ylim(0, 1.02)
    ax_ccc.set_ylabel("CCC")
    ax_ccc.set_title("CCC by depth and segment")
    ax_ccc.grid(axis="y", color=COLORS["neutral_light"], lw=0.6)
    add_panel_label(ax_ccc, "b")

    scatter = preds[preds["layer"].isin(["moisture_4in", "moisture_8in"])].copy()
    if len(scatter) > 7000:
        scatter = scatter.sample(7000, random_state=RANDOM_SEED)
    layer_colors = {"moisture_4in": "#355C7D", "moisture_8in": "#6FA38A"}
    for layer, panel in scatter.groupby("layer"):
        ax_scatter.scatter(
            panel["moisture_mm"],
            panel["pred_mm"],
            s=6,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            color=layer_colors[layer],
            label=LAYER_LABELS[layer],
        )
    lo = float(np.nanmin([scatter["moisture_mm"].min(), scatter["pred_mm"].min()]))
    hi = float(np.nanmax([scatter["moisture_mm"].max(), scatter["pred_mm"].max()]))
    pad = 0.05 * (hi - lo)
    ax_scatter.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS["accent"], lw=1.1)
    ax_scatter.text(
        hi - 0.12 * (hi - lo),
        hi - 0.03 * (hi - lo),
        "1:1 line",
        color=COLORS["accent"],
        fontsize=6.8,
        rotation=32,
        ha="left",
        va="bottom",
    )
    ax_scatter.set_xlim(lo - pad, hi + pad)
    ax_scatter.set_ylim(lo - pad, hi + pad)
    ax_scatter.set_xlabel("Observed soil water amount (mm)")
    ax_scatter.set_ylabel("Predicted soil water amount (mm)")
    ax_scatter.set_title("Held-out predicted versus observed soil water amount")
    ax_scatter.grid(color=COLORS["neutral_light"], lw=0.6)
    ax_scatter.legend(fontsize=6.4, loc="upper left")
    add_panel_label(ax_scatter, "c", x=-0.055, y=1.04)

    save_pub(fig, "fig_experiment3_split_validation_accuracy")
    plt.close(fig)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def write_accuracy_report(tables: dict[str, pd.DataFrame]) -> None:
    pooled_layer = tables["experiment3_split_validation_pooled_by_layer"].copy()
    pooled_segment = tables["experiment3_split_validation_pooled_by_layer_segment"].copy()
    local_summary = tables["experiment3_split_validation_local_summary_by_layer_segment"].copy()
    skipped = tables["experiment3_split_validation_skipped_models"].copy()

    pooled_layer["layer"] = pd.Categorical(pooled_layer["layer"], categories=LAYER_ORDER, ordered=True)
    pooled_segment["layer"] = pd.Categorical(pooled_segment["layer"], categories=LAYER_ORDER, ordered=True)
    pooled_segment["segment"] = pd.Categorical(pooled_segment["segment"], categories=SEGMENT_ORDER, ordered=True)
    local_summary["layer"] = pd.Categorical(local_summary["layer"], categories=LAYER_ORDER, ordered=True)
    local_summary["segment"] = pd.Categorical(local_summary["segment"], categories=SEGMENT_ORDER, ordered=True)

    lines = [
        "# Experiment 3 addendum: 80/20 split validation accuracy",
        "",
        "Prepared: 2026-06-05",
        "",
        "This addendum estimates held-out accuracy for localized segmented CSR using an event-level 80/20 split within each eligible location-layer-segment. Training events build the local state-stitched CSR curve; held-out events are aligned to the training anchor and predicted without contributing to the smoother.",
        "",
        "## Pooled held-out accuracy by layer",
        "",
        "| Layer | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm | nRMSE % range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_layer.sort_values("layer").itertuples(index=False):
        lines.append(
            f"| {row.layer_label} | {int(row.sites):,} | {int(row.test_events):,} | {int(row.test_points):,} | "
            f"{fmt(row.ccc)} | {fmt(row.r2)} | {fmt(row.rmse_mm)} | {fmt(row.mae_mm)} | "
            f"{fmt(row.bias_mm)} | {fmt(row.nrmse_pct_range, 1)} |"
        )

    lines.extend(
        [
            "",
            "## Pooled held-out accuracy by layer and segment",
            "",
            "| Layer | Segment | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pooled_segment.sort_values(["layer", "segment"]).itertuples(index=False):
        lines.append(
            f"| {row.layer_label} | {row.segment_label} | {int(row.sites):,} | {int(row.test_events):,} | "
            f"{int(row.test_points):,} | {fmt(row.ccc)} | {fmt(row.r2)} | {fmt(row.rmse_mm)} | "
            f"{fmt(row.mae_mm)} | {fmt(row.bias_mm)} |"
        )

    lines.extend(
        [
            "",
            "## Local-model median held-out accuracy",
            "",
            "| Layer | Segment | Valid local models | Median test events | Median CCC | Median R2 | Median RMSE mm | Median MAE mm |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in local_summary.sort_values(["layer", "segment"]).itertuples(index=False):
        lines.append(
            f"| {row.layer_label} | {row.segment_label} | {int(row.local_models):,} | "
            f"{fmt(row.median_test_events, 1)} | {fmt(row.median_ccc)} | {fmt(row.median_r2)} | "
            f"{fmt(row.median_rmse_mm)} | {fmt(row.median_mae_mm)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The 4 in layer has the strongest held-out support because it contributes the largest number of valid local validation models and test events.",
            "- The 8 in layer is still useful as secondary evidence, but its validation sample is much smaller.",
            "- Sparse 12-20 in validation rows should be treated as exploratory; skipped rows mostly reflect insufficient events for a strict event-level 80/20 split after preserving at least eight training events.",
            "- CCC is reported alongside RMSE, MAE, bias, and R2 to keep consistency with the earlier CSR diagnostics.",
            "",
            f"Skipped local validation rows: {len(skipped):,}.",
            "",
            "## Output files",
            "",
            "- `fig_experiment3_split_validation_accuracy.svg/pdf/png/tiff`",
            "- `source_data/experiment3_split_validation_pooled_by_layer.csv`",
            "- `source_data/experiment3_split_validation_pooled_by_layer_segment.csv`",
            "- `source_data/experiment3_split_validation_local_metrics.csv`",
            "- `source_data/experiment3_split_validation_local_summary_by_layer_segment.csv`",
            "- `source_data/experiment3_split_validation_predictions.parquet`",
            "",
        ]
    )

    (OUT / "experiment3_split_validation_accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")


def append_to_main_report(tables: dict[str, pd.DataFrame]) -> None:
    report = OUT / "experiment3_localized_segmented_csr_report.md"
    if not report.exists():
        return
    pooled_layer = tables["experiment3_split_validation_pooled_by_layer"].copy()
    pooled_layer["layer"] = pd.Categorical(pooled_layer["layer"], categories=LAYER_ORDER, ordered=True)
    validation_lines = [
        "",
        "## 80/20 split validation accuracy",
        "",
        "Held-out accuracy was evaluated with an event-level 80/20 split within each eligible location-layer-segment. Training events build the local state-stitched CSR curve; held-out events are predicted from the training curve.",
        "",
        "| Layer | Sites | Test events | Test points | CCC | R2 | RMSE mm | MAE mm | Bias mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_layer.sort_values("layer").itertuples(index=False):
        validation_lines.append(
            f"| {row.layer_label} | {int(row.sites):,} | {int(row.test_events):,} | {int(row.test_points):,} | "
            f"{fmt(row.ccc)} | {fmt(row.r2)} | {fmt(row.rmse_mm)} | {fmt(row.mae_mm)} | {fmt(row.bias_mm)} |"
        )
    validation_lines.extend(
        [
            "",
            "This split-validation result supports using 4 in as the primary localized segmented CSR layer and 8 in as secondary support; deeper layers remain exploratory because few local validation models meet the event-level split threshold.",
            "",
        ]
    )
    text = report.read_text(encoding="utf-8")
    marker = "## 80/20 split validation accuracy"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    report.write_text(text.rstrip() + "\n" + "\n".join(validation_lines), encoding="utf-8")


def main() -> None:
    ensure_out()
    points = pd.read_parquet(SEGCSR / "segmented_csr_aligned_points.parquet")
    points = points[
        (points["subset"] == MAIN_SUBSET)
        & (points["segment"].isin(SEGMENT_ORDER))
        & (points["layer"].isin(LAYER_ORDER))
    ].copy()
    needed = ["site_id", "event_id", "layer", "segment", "t_h", "segment_t_h", "event_storage_norm", "moisture_mm"]
    points = points[needed].dropna(subset=["site_id", "event_id", "layer", "segment", "segment_t_h", "moisture_mm"])
    tables = build_accuracy_tables(points)
    plot_accuracy_figure(tables)
    write_accuracy_report(tables)
    append_to_main_report(tables)
    print(f"Experiment 3 split-validation outputs written to {OUT}")
    print(tables["experiment3_split_validation_pooled_by_layer"].to_string(index=False))


if __name__ == "__main__":
    main()


