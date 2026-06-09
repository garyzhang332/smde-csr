from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from fawn_full_smde_audit import MOISTURE_LAYERS, OUT_DIR as AUDIT_DIR, prepare_soil


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("SMDE_SEGMENTED_CSR_OUT_DIR", ROOT / "_analysis" / "fawn_segmented_csr"))
EVENT_AUDIT = AUDIT_DIR / "full_smde_event_audit.csv"

LOWESS_BINS = 260
CURVE_GRID_POINTS = 220
SITE_BALANCE_CAP = 30
MIN_LOCAL_SEGMENT_EVENTS = 8
MIN_LOCAL_SEGMENT_POINTS = 40
RANDOM_SEED = 20260605
MAIN_SUBSET = "clean_stageII_48h"
PRIMARY_LAYERS = ["moisture_4in", "moisture_8in"]


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    label: str
    exclude_site: int | None = None
    site_cap: int | None = None


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    label: str
    coord_mode: str


SUBSETS = [
    SubsetSpec("all_events", "All detected events"),
    SubsetSpec("clean_48h", "Clean 48h events"),
    SubsetSpec("clean_stageII_48h", "Clean 48h + stage-II-like events"),
    SubsetSpec("clean_stageII_48h_no405", "Clean 48h + stage-II-like, no site 405", exclude_site=405),
    SubsetSpec(
        "clean_stageII_48h_site_balanced",
        f"Clean 48h + stage-II-like, site-balanced cap {SITE_BALANCE_CAP}",
        site_cap=SITE_BALANCE_CAP,
    ),
]

SEGMENTS = [
    SegmentSpec("early_0_3h", "Early transient, 0-3 h", "state_stitch"),
    SegmentSpec("post3_mid_storage", "Post-3 h mid-storage stage-II-like segment", "state_stitch"),
    SegmentSpec("post3_late_storage", "Post-3 h low-storage tail", "state_stitch"),
]
SEGMENT_ORDER = [segment.name for segment in SEGMENTS]


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENT_AUDIT, parse_dates=["start", "end"])
    events["clean_48h"] = events["associated_48h"] & ~events["interrupted_by_rain"]
    events["clean_stageII_48h"] = events["clean_48h"] & (events["regime_proxy"] == "stage-II-like")
    return events


def select_events(events: pd.DataFrame, spec: SubsetSpec) -> pd.DataFrame:
    if spec.name == "all_events":
        selected = events.copy()
    elif spec.name == "clean_48h":
        selected = events[events["clean_48h"]].copy()
    else:
        selected = events[events["clean_stageII_48h"]].copy()

    if spec.exclude_site is not None:
        selected = selected[selected["site_id"] != spec.exclude_site].copy()

    if spec.site_cap is not None and not selected.empty:
        parts = []
        for _, group in selected.groupby(["layer", "site_id"], sort=False):
            if len(group) > spec.site_cap:
                group = group.sample(n=spec.site_cap, random_state=RANDOM_SEED)
            parts.append(group)
        selected = pd.concat(parts, ignore_index=True)

    return selected.sort_values(["layer", "site_id", "start", "event_id"]).reset_index(drop=True)


def build_event_points(events: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for site_id in sorted(events["site_id"].unique()):
        soil = prepare_soil(int(site_id))
        if soil.empty:
            continue
        soil["UTC"] = pd.to_datetime(soil["UTC"], errors="coerce")
        soil = soil.dropna(subset=["UTC"]).sort_values("UTC").drop_duplicates("UTC").set_index("UTC")
        site_events = events[events["site_id"] == site_id]
        for row in site_events.itertuples(index=False):
            layer = row.layer
            if layer not in soil.columns:
                continue
            segment = soil.loc[pd.Timestamp(row.start) : pd.Timestamp(row.end), [layer]].dropna()
            if len(segment) < 2:
                continue
            t_h = (segment.index - segment.index[0]).total_seconds().to_numpy() / 3600.0
            frames.append(
                pd.DataFrame(
                    {
                        "site_id": int(site_id),
                        "event_id": row.event_id,
                        "layer": layer,
                        "t_h": t_h,
                        "moisture_mm": segment[layer].to_numpy(dtype=float),
                        "start_mm": float(row.start_mm),
                        "end_mm": float(row.end_mm),
                        "total_drop_mm": float(row.total_drop_mm),
                        "duration_h": float(row.duration_h),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_subset_points(all_points: pd.DataFrame, selected_events: pd.DataFrame) -> pd.DataFrame:
    if selected_events.empty:
        return pd.DataFrame()
    points = all_points[all_points["event_id"].isin(selected_events["event_id"])].copy()
    if points.empty:
        return points

    points = points.sort_values(["event_id", "t_h"])
    counts = points.groupby("event_id")["moisture_mm"].transform("count")
    points = points[counts >= 4].copy()
    denom = points["total_drop_mm"].where(points["total_drop_mm"] > 0)
    points["event_storage_norm"] = ((points["moisture_mm"] - points["end_mm"]) / denom).clip(0, 1)
    return assign_segments(points)


def assign_segments(points: pd.DataFrame) -> pd.DataFrame:
    points = points.copy()
    points["segment"] = np.select(
        [
            points["t_h"] < 3.0,
            (points["t_h"] >= 3.0) & (points["event_storage_norm"] >= 0.25),
            (points["t_h"] >= 3.0) & (points["event_storage_norm"] < 0.25),
        ],
        SEGMENT_ORDER,
        default="unclassified",
    )
    return points[points["segment"].isin(SEGMENT_ORDER)].copy()


def prepare_segment_points(points: pd.DataFrame, segment_name: str) -> pd.DataFrame:
    segment = points[points["segment"] == segment_name].copy()
    if segment.empty:
        return segment
    first_t = segment.groupby("event_id")["t_h"].transform("min")
    segment["segment_t_h"] = segment["t_h"] - first_t
    counts = segment.groupby("event_id")["moisture_mm"].transform("count")
    segment = segment[counts >= 4].copy()
    return segment


def anchor_interpolator(aligned: pd.DataFrame, bins: int = 200) -> tuple[np.ndarray, np.ndarray] | None:
    if aligned.empty:
        return None
    y = aligned["moisture_mm"].to_numpy(dtype=float)
    x = aligned["csr_x_h"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.nanmax(y[ok]) <= np.nanmin(y[ok]):
        return None
    frame = pd.DataFrame({"x": x[ok], "y": y[ok]})
    edges = np.linspace(frame["y"].min(), frame["y"].max(), min(bins, len(frame)) + 1)
    frame["bin"] = pd.cut(frame["y"], edges, include_lowest=True, duplicates="drop")
    med = frame.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median")).dropna()
    med = med.sort_values("y").drop_duplicates("y")
    if len(med) < 4:
        return None
    return med["y"].to_numpy(dtype=float), med["x"].to_numpy(dtype=float)


def align_by_phase_time(segment: pd.DataFrame) -> pd.DataFrame:
    aligned = segment.copy()
    aligned["csr_x_h"] = aligned["segment_t_h"]
    return aligned


def align_by_state_stitch(segment: pd.DataFrame) -> pd.DataFrame:
    event_index = (
        segment.sort_values(["event_id", "segment_t_h"])
        .groupby("event_id")
        .agg(start_mm=("moisture_mm", "first"), end_mm=("moisture_mm", "last"), n=("moisture_mm", "size"))
        .reset_index()
    )
    event_index = event_index[event_index["n"] >= 4].sort_values(["start_mm", "end_mm"], ascending=[False, False])

    aligned_parts: list[pd.DataFrame] = []
    interp: tuple[np.ndarray, np.ndarray] | None = None
    aligned_so_far = pd.DataFrame()

    for idx, event_id in enumerate(event_index["event_id"], start=1):
        event = segment[segment["event_id"] == event_id].sort_values("segment_t_h").copy()
        if len(event) < 4:
            continue
        t = event["segment_t_h"].to_numpy(dtype=float)
        y = event["moisture_mm"].to_numpy(dtype=float)

        if not aligned_parts:
            offset = 0.0
        else:
            if interp is None or idx % 25 == 0:
                aligned_so_far = pd.concat(aligned_parts, ignore_index=True)
                interp = anchor_interpolator(aligned_so_far)
            if interp is None:
                offset = float(aligned_so_far["csr_x_h"].max())
            else:
                anchor_y, anchor_x = interp
                overlap = (y >= anchor_y.min()) & (y <= anchor_y.max())
                if overlap.sum() >= 3:
                    matched_x = np.interp(y[overlap], anchor_y, anchor_x)
                    offset = float(np.nanmedian(matched_x - t[overlap]))
                elif np.nanmax(y) < anchor_y.min():
                    aligned_so_far = pd.concat(aligned_parts, ignore_index=True)
                    step = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.25
                    offset = float(aligned_so_far["csr_x_h"].max() + step)
                else:
                    offset = 0.0

        event["csr_x_h"] = event["segment_t_h"] + offset
        aligned_parts.append(event)

    aligned = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else pd.DataFrame()
    if not aligned.empty:
        aligned["csr_x_h"] = aligned["csr_x_h"] - aligned["csr_x_h"].min()
    return aligned


def align_segment(segment: pd.DataFrame, segment_spec: SegmentSpec) -> pd.DataFrame:
    if segment_spec.coord_mode == "phase_time":
        return align_by_phase_time(segment)
    if segment_spec.coord_mode == "state_stitch":
        return align_by_state_stitch(segment)
    raise ValueError(f"Unknown coordinate mode: {segment_spec.coord_mode}")


def binned_points(aligned: pd.DataFrame, bins: int = LOWESS_BINS) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame()
    x = aligned["csr_x_h"].to_numpy(dtype=float)
    if np.nanmax(x) <= np.nanmin(x):
        return pd.DataFrame()
    edges = np.linspace(np.nanmin(x), np.nanmax(x), min(bins, len(aligned)) + 1)
    frame = aligned.copy()
    frame["bin"] = pd.cut(frame["csr_x_h"], edges, include_lowest=True, duplicates="drop")
    binned = (
        frame.groupby("bin", observed=True)
        .agg(
            csr_x_h=("csr_x_h", "median"),
            moisture_mm=("moisture_mm", "median"),
            n_points=("moisture_mm", "size"),
            q25_mm=("moisture_mm", lambda x: float(np.nanquantile(x, 0.25))),
            q75_mm=("moisture_mm", lambda x: float(np.nanquantile(x, 0.75))),
        )
        .dropna()
        .reset_index(drop=True)
        .sort_values("csr_x_h")
    )
    return binned


def fit_segment_curve(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    binned = binned_points(aligned)
    if len(binned) < 8:
        return pd.DataFrame(), binned
    x = binned["csr_x_h"].to_numpy(dtype=float)
    y = binned["moisture_mm"].to_numpy(dtype=float)
    frac = min(0.5, max(0.12, 40.0 / len(binned)))
    fitted = lowess(y, x, frac=frac, it=1, return_sorted=True)
    grid = np.linspace(x.min(), x.max(), CURVE_GRID_POINTS)
    raw = np.interp(grid, fitted[:, 0], fitted[:, 1])
    monotone = np.minimum.accumulate(raw)
    curve = pd.DataFrame({"csr_x_h": grid, "csr_mm": monotone, "raw_smooth_mm": raw})
    return curve, binned


def curve_predict(curve: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    return np.interp(x, curve["csr_x_h"].to_numpy(dtype=float), curve["csr_mm"].to_numpy(dtype=float))


def concordance_correlation_coefficient(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 3:
        return np.nan
    mean_o = float(np.mean(obs))
    mean_p = float(np.mean(pred))
    var_o = float(np.var(obs))
    var_p = float(np.var(pred))
    cov = float(np.mean((obs - mean_o) * (pred - mean_p)))
    denom = var_o + var_p + (mean_o - mean_p) ** 2
    return 2.0 * cov / denom if denom > 0 else np.nan


def summarize_predictions(predictions: pd.DataFrame, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    layer_rows = []
    segment_rows = []

    for layer, layer_data in predictions.groupby("layer"):
        obs = layer_data["moisture_mm"].to_numpy(dtype=float)
        pred = layer_data["pred_mm"].to_numpy(dtype=float)
        residual = pred - obs
        layer_rows.append(
            {
                "subset": subset,
                "layer": layer,
                "events": int(layer_data["event_id"].nunique()),
                "sites": int(layer_data["site_id"].nunique()),
                "points": int(len(layer_data)),
                "ccc": concordance_correlation_coefficient(obs, pred),
                "rmse_mm": float(np.sqrt(np.nanmean(residual**2))),
                "mae_mm": float(np.nanmean(np.abs(residual))),
                "bias_mm": float(np.nanmean(residual)),
            }
        )

        for segment, segment_data in layer_data.groupby("segment"):
            obs_segment = segment_data["moisture_mm"].to_numpy(dtype=float)
            pred_segment = segment_data["pred_mm"].to_numpy(dtype=float)
            residual_segment = pred_segment - obs_segment
            dynamic_errors = []
            for _, event in segment_data.sort_values(["event_id", "t_h"]).groupby("event_id"):
                if len(event) < 2:
                    continue
                obs_event = event["moisture_mm"].to_numpy(dtype=float)
                pred_event = event["pred_mm"].to_numpy(dtype=float)
                dynamic_errors.extend(((pred_event[:-1] - pred_event[1:]) - (obs_event[:-1] - obs_event[1:])).tolist())
            segment_rows.append(
                {
                    "subset": subset,
                    "layer": layer,
                    "segment": segment,
                    "events": int(segment_data["event_id"].nunique()),
                    "sites": int(segment_data["site_id"].nunique()),
                    "points": int(len(segment_data)),
                    "ccc": concordance_correlation_coefficient(obs_segment, pred_segment),
                    "rmse_mm": float(np.sqrt(np.nanmean(residual_segment**2))),
                    "mae_mm": float(np.nanmean(np.abs(residual_segment))),
                    "bias_mm": float(np.nanmean(residual_segment)),
                    "dynamic_mae_mm_step": float(np.nanmean(np.abs(dynamic_errors))) if dynamic_errors else np.nan,
                }
            )

    return pd.DataFrame(layer_rows), pd.DataFrame(segment_rows)


def curve_distance_to_main(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = curves[curves["subset"] == MAIN_SUBSET]
    for (site_id, layer, segment), main_curve in main.groupby(["site_id", "layer", "segment"]):
        if main_curve.empty:
            continue
        for subset in sorted(curves["subset"].unique()):
            other = curves[
                (curves["site_id"] == site_id)
                & (curves["layer"] == layer)
                & (curves["segment"] == segment)
                & (curves["subset"] == subset)
            ]
            if other.empty:
                continue
            lo = max(float(main_curve["csr_x_h"].min()), float(other["csr_x_h"].min()))
            hi = min(float(main_curve["csr_x_h"].max()), float(other["csr_x_h"].max()))
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, 180)
            main_y = np.interp(grid, main_curve["csr_x_h"], main_curve["csr_mm"])
            other_y = np.interp(grid, other["csr_x_h"], other["csr_mm"])
            diff = other_y - main_y
            rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "segment": segment,
                    "subset": subset,
                    "main_subset": MAIN_SUBSET,
                    "curve_rmse_to_main_mm": float(np.sqrt(np.mean(diff**2))),
                    "curve_mae_to_main_mm": float(np.mean(np.abs(diff))),
                    "curve_bias_to_main_mm": float(np.mean(diff)),
                    "curve_max_abs_to_main_mm": float(np.max(np.abs(diff))),
                    "overlap_min_h": lo,
                    "overlap_max_h": hi,
                }
            )
    return pd.DataFrame(rows)


def local_segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (subset, site_id, layer, segment), data in predictions.groupby(["subset", "site_id", "layer", "segment"]):
        obs = data["moisture_mm"].to_numpy(dtype=float)
        pred = data["pred_mm"].to_numpy(dtype=float)
        residual = pred - obs
        rows.append(
            {
                "subset": subset,
                "site_id": int(site_id),
                "layer": layer,
                "segment": segment,
                "events": int(data["event_id"].nunique()),
                "points": int(len(data)),
                "ccc": concordance_correlation_coefficient(obs, pred),
                "rmse_mm": float(np.sqrt(np.nanmean(residual**2))),
                "mae_mm": float(np.nanmean(np.abs(residual))),
                "bias_mm": float(np.nanmean(residual)),
            }
        )
    return pd.DataFrame(rows)


def summarize_local_metrics(local_metrics: pd.DataFrame) -> pd.DataFrame:
    if local_metrics.empty:
        return pd.DataFrame()
    return (
        local_metrics.groupby(["subset", "layer", "segment"], dropna=False)
        .agg(
            local_models=("site_id", "count"),
            median_events_per_model=("events", "median"),
            total_points=("points", "sum"),
            median_rmse_mm=("rmse_mm", "median"),
            mean_rmse_mm=("rmse_mm", "mean"),
            median_mae_mm=("mae_mm", "median"),
            mean_mae_mm=("mae_mm", "mean"),
            median_bias_mm=("bias_mm", "median"),
            mean_bias_mm=("bias_mm", "mean"),
        )
        .reset_index()
    )


def distance_summary(distance: pd.DataFrame) -> pd.DataFrame:
    if distance.empty:
        return pd.DataFrame()
    return (
        distance.groupby(["layer", "segment", "subset"], dropna=False)
        .agg(
            local_model_pairs=("site_id", "count"),
            median_curve_rmse_to_main_mm=("curve_rmse_to_main_mm", "median"),
            mean_curve_rmse_to_main_mm=("curve_rmse_to_main_mm", "mean"),
            median_curve_bias_to_main_mm=("curve_bias_to_main_mm", "median"),
            median_curve_max_abs_to_main_mm=("curve_max_abs_to_main_mm", "median"),
        )
        .reset_index()
    )


def fit_segmented_csr_for_subset(
    all_points: pd.DataFrame, selected_events: pd.DataFrame, spec: SubsetSpec
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = prepare_subset_points(all_points, selected_events)
    aligned_outputs = []
    curve_outputs = []
    binned_outputs = []
    prediction_outputs = []

    if points.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for (site_id, layer), site_layer_points in points.groupby(["site_id", "layer"]):
        if site_layer_points.empty:
            continue
        for segment_spec in SEGMENTS:
            segment_points = prepare_segment_points(site_layer_points, segment_spec.name)
            if (
                segment_points["event_id"].nunique() < MIN_LOCAL_SEGMENT_EVENTS
                or len(segment_points) < MIN_LOCAL_SEGMENT_POINTS
            ):
                continue

            aligned = align_segment(segment_points, segment_spec)
            if aligned.empty or aligned["event_id"].nunique() < MIN_LOCAL_SEGMENT_EVENTS:
                continue

            curve, binned = fit_segment_curve(aligned)
            if curve.empty:
                continue

            prediction = aligned[
                ["site_id", "event_id", "layer", "segment", "t_h", "segment_t_h", "event_storage_norm", "moisture_mm", "csr_x_h"]
            ].copy()
            prediction["pred_mm"] = curve_predict(curve, prediction["csr_x_h"].to_numpy(dtype=float))
            prediction["subset"] = spec.name

            aligned["subset"] = spec.name
            aligned["segment"] = segment_spec.name
            curve["subset"] = spec.name
            curve["subset_label"] = spec.label
            curve["site_id"] = int(site_id)
            curve["layer"] = layer
            curve["segment"] = segment_spec.name
            curve["segment_label"] = segment_spec.label
            curve["coord_mode"] = segment_spec.coord_mode
            binned["subset"] = spec.name
            binned["site_id"] = int(site_id)
            binned["layer"] = layer
            binned["segment"] = segment_spec.name
            binned["segment_label"] = segment_spec.label
            binned["coord_mode"] = segment_spec.coord_mode

            aligned_outputs.append(aligned)
            curve_outputs.append(curve)
            binned_outputs.append(binned)
            prediction_outputs.append(prediction)

    aligned_all = pd.concat(aligned_outputs, ignore_index=True) if aligned_outputs else pd.DataFrame()
    curves = pd.concat(curve_outputs, ignore_index=True) if curve_outputs else pd.DataFrame()
    binned_all = pd.concat(binned_outputs, ignore_index=True) if binned_outputs else pd.DataFrame()
    predictions = pd.concat(prediction_outputs, ignore_index=True) if prediction_outputs else pd.DataFrame()
    return aligned_all, curves, binned_all, predictions


def plot_main_segmented_curves(curves: pd.DataFrame, binned: pd.DataFrame) -> None:
    data = curves[(curves["subset"] == MAIN_SUBSET) & (curves["layer"].isin(PRIMARY_LAYERS))].copy()
    support = binned[(binned["subset"] == MAIN_SUBSET) & (binned["layer"].isin(PRIMARY_LAYERS))].copy()
    if data.empty:
        return
    reps = (
        data.groupby(["layer", "site_id"])["segment"]
        .nunique()
        .reset_index(name="segments")
        .sort_values(["layer", "segments", "site_id"], ascending=[True, False, True])
        .groupby("layer")
        .head(1)
    )
    fig, axes = plt.subplots(len(PRIMARY_LAYERS), len(SEGMENT_ORDER), figsize=(16, 8.5), sharex=False, sharey=False)
    for i, layer in enumerate(PRIMARY_LAYERS):
        rep_rows = reps[reps["layer"] == layer]
        if rep_rows.empty:
            for j in range(len(SEGMENT_ORDER)):
                axes[i, j].set_visible(False)
            continue
        site_id = int(rep_rows["site_id"].iloc[0])
        for j, segment in enumerate(SEGMENT_ORDER):
            ax = axes[i, j]
            curve = data[(data["layer"] == layer) & (data["site_id"] == site_id) & (data["segment"] == segment)]
            points = support[(support["layer"] == layer) & (support["site_id"] == site_id) & (support["segment"] == segment)]
            if curve.empty:
                ax.set_visible(False)
                continue
            if not points.empty:
                ax.fill_between(points["csr_x_h"], points["q25_mm"], points["q75_mm"], alpha=0.22, color="#8ab6d6", label="Binned IQR")
                ax.scatter(points["csr_x_h"], points["moisture_mm"], s=8, alpha=0.35, color="#376996", label="Binned median")
            ax.plot(curve["csr_x_h"], curve["csr_mm"], color="#b23a48", linewidth=2.3, label="Segmented CSR")
            ax.set_title(f"Site {site_id}, {layer}: {segment}")
            ax.set_xlabel("Segment coordinate (h)")
            ax.set_ylabel("Soil water amount (mm)")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_segmented_csr_main_primary_layers.png", dpi=300)
    plt.close(fig)


def plot_segmented_sensitivity(curves: pd.DataFrame) -> None:
    data = curves[curves["layer"].isin(PRIMARY_LAYERS)].copy()
    data = data[data["subset"].isin(["clean_48h", MAIN_SUBSET, "clean_stageII_48h_no405", "clean_stageII_48h_site_balanced"])]
    if data.empty:
        return
    main = data[data["subset"] == MAIN_SUBSET]
    reps = (
        main.groupby(["layer", "site_id"])["segment"]
        .nunique()
        .reset_index(name="segments")
        .sort_values(["layer", "segments", "site_id"], ascending=[True, False, True])
        .groupby("layer")
        .head(1)
    )
    fig, axes = plt.subplots(len(PRIMARY_LAYERS), len(SEGMENT_ORDER), figsize=(16, 8.5), sharex=False, sharey=False)
    for i, layer in enumerate(PRIMARY_LAYERS):
        rep_rows = reps[reps["layer"] == layer]
        if rep_rows.empty:
            for j in range(len(SEGMENT_ORDER)):
                axes[i, j].set_visible(False)
            continue
        site_id = int(rep_rows["site_id"].iloc[0])
        for j, segment in enumerate(SEGMENT_ORDER):
            ax = axes[i, j]
            panel = data[(data["layer"] == layer) & (data["site_id"] == site_id) & (data["segment"] == segment)]
            if panel.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(data=panel, x="csr_x_h", y="csr_mm", hue="subset", ax=ax, linewidth=2)
            ax.set_title(f"Site {site_id}, {layer}: {segment}")
            ax.set_xlabel("Segment coordinate (h)")
            ax.set_ylabel("Soil water amount (mm)")
            ax.legend(title="Subset", fontsize=7, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_segmented_csr_sensitivity_primary_layers.png", dpi=300)
    plt.close(fig)


def plot_segment_metrics(local_summary: pd.DataFrame) -> None:
    data = local_summary[
        (local_summary["subset"] == MAIN_SUBSET) & (local_summary["layer"].isin(PRIMARY_LAYERS))
    ].copy()
    if data.empty:
        return
    data["segment"] = pd.Categorical(data["segment"], categories=SEGMENT_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    sns.barplot(data=data, x="segment", y="median_rmse_mm", hue="layer", ax=axes[0])
    axes[0].set_title("Median local segmented CSR RMSE")
    axes[0].set_xlabel("Segment")
    axes[0].set_ylabel("RMSE (mm)")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=data, x="segment", y="local_models", hue="layer", ax=axes[1])
    axes[1].set_title("Number of fitted local models")
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("Location-layer models")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_segmented_csr_main_segment_metrics.png", dpi=300)
    plt.close(fig)


def plot_curve_distance(distance_summary_df: pd.DataFrame) -> None:
    data = distance_summary_df[
        (distance_summary_df["subset"] != MAIN_SUBSET) & (distance_summary_df["layer"].isin(PRIMARY_LAYERS))
    ].copy()
    if data.empty:
        return
    data["segment"] = pd.Categorical(data["segment"], categories=SEGMENT_ORDER, ordered=True)
    fig, axes = plt.subplots(1, len(PRIMARY_LAYERS), figsize=(16, 5.8), sharey=True)
    for i, layer in enumerate(PRIMARY_LAYERS):
        ax = axes[i]
        panel = data[data["layer"] == layer]
        if panel.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=panel, x="segment", y="median_curve_rmse_to_main_mm", hue="subset", ax=ax)
        ax.set_title(layer)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Median local curve RMSE to main (mm)" if i == 0 else "")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Subset", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_segmented_csr_curve_distance_to_main.png", dpi=300)
    plt.close(fig)


def plot_location_layer_heatmaps(local_metrics: pd.DataFrame) -> None:
    data = local_metrics[
        (local_metrics["subset"] == MAIN_SUBSET)
        & (local_metrics["layer"].isin(PRIMARY_LAYERS))
        & (local_metrics["segment"].isin(["early_0_3h", "post3_mid_storage", "post3_late_storage"]))
    ].copy()
    if data.empty:
        return
    data["site_id"] = data["site_id"].astype(int)
    fig, axes = plt.subplots(1, len(SEGMENT_ORDER), figsize=(15, 9), sharey=True)
    vmax = float(data["rmse_mm"].quantile(0.95))
    for j, segment in enumerate(SEGMENT_ORDER):
        ax = axes[j]
        panel = data[data["segment"] == segment]
        if panel.empty:
            ax.set_visible(False)
            continue
        heat = panel.pivot_table(index="site_id", columns="layer", values="rmse_mm", aggfunc="median")
        heat = heat.reindex(columns=PRIMARY_LAYERS)
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap="viridis_r",
            vmin=0,
            vmax=vmax,
            cbar=j == len(SEGMENT_ORDER) - 1,
            ax=ax,
        )
        ax.set_title(segment)
        ax.set_xlabel("Layer")
        ax.set_ylabel("FAWN site ID" if j == 0 else "")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_segmented_csr_location_layer_rmse_heatmaps.png", dpi=300)
    plt.close(fig)


def write_key_findings(
    layer_metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    local_summary: pd.DataFrame,
    distance_summary_df: pd.DataFrame,
) -> None:
    main_layer = layer_metrics[(layer_metrics["subset"] == MAIN_SUBSET) & (layer_metrics["layer"].isin(PRIMARY_LAYERS))]
    main_segment = segment_metrics[(segment_metrics["subset"] == MAIN_SUBSET) & (segment_metrics["layer"].isin(PRIMARY_LAYERS))]
    main_local = local_summary[
        (local_summary["subset"] == MAIN_SUBSET) & (local_summary["layer"].isin(PRIMARY_LAYERS))
    ].copy()

    lines = [
        "# Segmented CSR construction results",
        "",
        "Prepared: 2026-06-05",
        "",
        "This analysis replaces a single pooled CSR curve with localized segmented CSR.",
        "Each location-layer-segment is fitted separately. Full-station results are summarized",
        "after all local segmented CSR models are built. LOWESS is used only as a within-segment",
        "empirical smoother, not as a pooled curve across locations, layers, or regimes.",
        "",
        "## Segment definitions",
        "",
        "| Segment | Coordinate | Interpretation |",
        "|---|---|---|",
        "| early_0_3h | local moisture-state stitching | rapid early redistribution, drainage, or sensor/layer adjustment after wetting |",
        "| post3_mid_storage | moisture-state stitching | main post-transient storage-dependent drydown segment |",
        "| post3_late_storage | moisture-state stitching | low-storage tail where drying may slow or become noise-limited |",
        "",
        "The manuscript-safe main subset is `clean_stageII_48h`: rainfall-associated within 48 h,",
        "not interrupted by rain during the event, and classified as stage-II-like by the diagnostic proxy.",
        "",
        "## Full-station summary of local segmented CSR models",
        "",
    ]

    if not main_local.empty:
        lines.extend(
            [
                "| Layer | Segment | Local models | Median events/model | Total points | Median RMSE mm | Median MAE mm | Median bias mm |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in main_local.sort_values(["layer", "segment"]).itertuples(index=False):
            lines.append(
                f"| {row.layer} | {row.segment} | {row.local_models:,} | "
                f"{row.median_events_per_model:.1f} | {row.total_points:,} | "
                f"{row.median_rmse_mm:.3f} | {row.median_mae_mm:.3f} | {row.median_bias_mm:.3f} |"
            )
        lines.append("")

    lines.extend(["## Pooled diagnostic metrics from local predictions", ""])
    if not main_layer.empty:
        lines.extend(
            [
                "| Layer | Events | Sites | Points | CCC | RMSE mm | MAE mm | Bias mm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in main_layer.itertuples(index=False):
            lines.append(
                f"| {row.layer} | {row.events:,} | {row.sites:,} | {row.points:,} | "
                f"{row.ccc:.3f} | {row.rmse_mm:.3f} | {row.mae_mm:.3f} | {row.bias_mm:.3f} |"
            )
        lines.append("")

    lines.extend(["## Segment-level pooled diagnostics", ""])
    if not main_segment.empty:
        lines.extend(
            [
                "| Layer | Segment | Events | Sites | Points | RMSE mm | MAE mm | Dynamic MAE mm/step |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in main_segment.sort_values(["layer", "segment"]).itertuples(index=False):
            lines.append(
                f"| {row.layer} | {row.segment} | {row.events:,} | {row.sites:,} | {row.points:,} | "
                f"{row.rmse_mm:.3f} | {row.mae_mm:.3f} | {row.dynamic_mae_mm_step:.3f} |"
            )
        lines.append("")

    primary_distance = distance_summary_df[
        (distance_summary_df["subset"].isin(["clean_48h", "clean_stageII_48h_no405", "clean_stageII_48h_site_balanced"]))
        & (distance_summary_df["layer"].isin(PRIMARY_LAYERS))
    ].copy()
    if not primary_distance.empty:
        lines.extend(
            [
                "## Sensitivity against main segmented CSR",
                "",
                "| Layer | Segment | Subset | Local model pairs | Median curve RMSE to main mm | Median curve bias to main mm |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in primary_distance.sort_values(["layer", "segment", "subset"]).itertuples(index=False):
            lines.append(
                f"| {row.layer} | {row.segment} | {row.subset} | "
                f"{row.local_model_pairs:,} | {row.median_curve_rmse_to_main_mm:.3f} | "
                f"{row.median_curve_bias_to_main_mm:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Manuscript interpretation",
            "",
            "The localized segmented CSR framing avoids two averaging problems: mixing regimes into one",
            "curve and mixing stations with different soil-water ranges into one pooled curve. Each local",
            "location-layer model has its own early transient, post-3h mid-storage, and post-3h low-storage",
            "segments. Full-station tables summarize how many local models can be fitted and how stable each",
            "segment is across the network.",
            "",
            "## Output files",
            "",
            "- `segmented_csr_curves.csv`",
            "- `segmented_csr_aligned_points.parquet`",
            "- `segmented_csr_binned_points.csv`",
            "- `segmented_csr_predictions.parquet`",
            "- `segmented_csr_layer_metrics.csv`",
            "- `segmented_csr_segment_metrics.csv`",
            "- `segmented_csr_local_segment_metrics.csv`",
            "- `segmented_csr_local_summary_by_layer_segment.csv`",
            "- `segmented_csr_curve_distance_to_main.csv`",
            "- `segmented_csr_curve_distance_summary.csv`",
            "- `fig_segmented_csr_main_primary_layers.png`",
            "- `fig_segmented_csr_sensitivity_primary_layers.png`",
            "- `fig_segmented_csr_main_segment_metrics.png`",
            "- `fig_segmented_csr_curve_distance_to_main.png`",
            "- `fig_segmented_csr_location_layer_rmse_heatmaps.png`",
        ]
    )

    (OUT_DIR / "segmented_csr_key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out_dir()
    events = load_events()
    all_points = build_event_points(events)
    all_points.to_parquet(OUT_DIR / "segmented_csr_raw_event_points.parquet", index=False)

    aligned_outputs = []
    curve_outputs = []
    binned_outputs = []
    prediction_outputs = []
    subset_counts = []

    for spec in SUBSETS:
        selected = select_events(events, spec)
        subset_counts.append(
            {
                "subset": spec.name,
                "label": spec.label,
                "events_before_segment_filter": int(len(selected)),
                "sites_before_segment_filter": int(selected["site_id"].nunique()) if not selected.empty else 0,
            }
        )
        aligned, curves, binned, predictions = fit_segmented_csr_for_subset(all_points, selected, spec)
        if not aligned.empty:
            aligned_outputs.append(aligned)
        if not curves.empty:
            curve_outputs.append(curves)
        if not binned.empty:
            binned_outputs.append(binned)
        if not predictions.empty:
            prediction_outputs.append(predictions)

    aligned_all = pd.concat(aligned_outputs, ignore_index=True) if aligned_outputs else pd.DataFrame()
    curves_all = pd.concat(curve_outputs, ignore_index=True) if curve_outputs else pd.DataFrame()
    binned_all = pd.concat(binned_outputs, ignore_index=True) if binned_outputs else pd.DataFrame()
    predictions_all = pd.concat(prediction_outputs, ignore_index=True) if prediction_outputs else pd.DataFrame()
    subset_counts_df = pd.DataFrame(subset_counts)

    layer_metric_frames = []
    segment_metric_frames = []
    for subset in sorted(predictions_all["subset"].unique()):
        subset_predictions = predictions_all[predictions_all["subset"] == subset]
        layer_metrics, segment_metrics = summarize_predictions(subset_predictions, subset)
        layer_metric_frames.append(layer_metrics)
        segment_metric_frames.append(segment_metrics)
    layer_metrics_all = pd.concat(layer_metric_frames, ignore_index=True) if layer_metric_frames else pd.DataFrame()
    segment_metrics_all = pd.concat(segment_metric_frames, ignore_index=True) if segment_metric_frames else pd.DataFrame()
    local_metrics_all = local_segment_metrics(predictions_all)
    local_summary_all = summarize_local_metrics(local_metrics_all)
    distance = curve_distance_to_main(curves_all)
    distance_summary_all = distance_summary(distance)

    aligned_all.to_parquet(OUT_DIR / "segmented_csr_aligned_points.parquet", index=False)
    curves_all.to_csv(OUT_DIR / "segmented_csr_curves.csv", index=False)
    binned_all.to_csv(OUT_DIR / "segmented_csr_binned_points.csv", index=False)
    predictions_all.to_parquet(OUT_DIR / "segmented_csr_predictions.parquet", index=False)
    layer_metrics_all.to_csv(OUT_DIR / "segmented_csr_layer_metrics.csv", index=False)
    segment_metrics_all.to_csv(OUT_DIR / "segmented_csr_segment_metrics.csv", index=False)
    local_metrics_all.to_csv(OUT_DIR / "segmented_csr_local_segment_metrics.csv", index=False)
    local_summary_all.to_csv(OUT_DIR / "segmented_csr_local_summary_by_layer_segment.csv", index=False)
    distance.to_csv(OUT_DIR / "segmented_csr_curve_distance_to_main.csv", index=False)
    distance_summary_all.to_csv(OUT_DIR / "segmented_csr_curve_distance_summary.csv", index=False)
    subset_counts_df.to_csv(OUT_DIR / "segmented_csr_subset_counts.csv", index=False)

    plot_main_segmented_curves(curves_all, binned_all)
    plot_segmented_sensitivity(curves_all)
    plot_segment_metrics(local_summary_all)
    plot_curve_distance(distance_summary_all)
    plot_location_layer_heatmaps(local_metrics_all)
    write_key_findings(layer_metrics_all, segment_metrics_all, local_summary_all, distance_summary_all)

    manifest = {
        "main_subset": MAIN_SUBSET,
        "subsets": [spec.name for spec in SUBSETS],
        "segments": SEGMENT_ORDER,
        "layers": sorted(curves_all["layer"].unique().tolist()) if not curves_all.empty else [],
        "curves": int(len(curves_all)),
        "aligned_points": int(len(aligned_all)),
        "prediction_points": int(len(predictions_all)),
        "layer_metric_rows": int(len(layer_metrics_all)),
        "segment_metric_rows": int(len(segment_metrics_all)),
        "local_segment_metric_rows": int(len(local_metrics_all)),
        "local_summary_rows": int(len(local_summary_all)),
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
    }
    (OUT_DIR / "segmented_csr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("\nMain subset local summary:")
    print(local_summary_all[local_summary_all["subset"] == MAIN_SUBSET].to_string(index=False))


if __name__ == "__main__":
    main()
