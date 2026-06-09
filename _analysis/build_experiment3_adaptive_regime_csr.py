from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fawn_segmented_csr import (
    MIN_LOCAL_SEGMENT_EVENTS,
    MIN_LOCAL_SEGMENT_POINTS,
    concordance_correlation_coefficient,
    curve_predict,
    fit_segment_curve,
)
from hydro_csr_registration import (
    RegistrationConfig,
    align_to_registered_template,
    hydrologically_constrained_registration,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
EVENT_AUDIT = ANALYSIS / "fawn_full_smde_audit" / "full_smde_event_audit.csv"
RAW_POINTS = ANALYSIS / "fawn_segmented_csr" / "segmented_csr_raw_event_points.parquet"
OUT = ANALYSIS / "experiment3_adaptive_regime_csr"
SOURCE = OUT / "source_data"

RANDOM_SEED = 20260605
MAX_SEGMENTS = 3
MIN_SEGMENT_POINTS = 5
MAX_CANDIDATE_CUTS = 28
TEST_FRACTION = 0.20
MIN_TRAIN_EVENTS = 8
MIN_TEST_EVENTS = 2
MIN_TRAIN_POINTS = 40
MIN_TEST_POINTS = 8
REP_LAYER = "moisture_4in"
EXCLUDE_REP_SITE = 405
REGISTRATION_CFG = RegistrationConfig()

LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}

REGIME_ORDER = ["early_transient", "stageI_like", "stageII_like"]
REGIME_LABELS = {
    "early_transient": "Early transient",
    "stageI_like": "Stage I-like",
    "stageII_like": "Stage II-like",
}
REGIME_PROXY = {
    "early_transient": "early-transient-heavy",
    "stageI_like": "stage-I-like",
    "stageII_like": "stage-II-like",
}
REGIME_PROCESS = {
    "early_transient": "post-rain redistribution, drainage, and runoff-related adjustment",
    "stageI_like": "approximately storage-invariant atmospheric-demand-limited loss",
    "stageII_like": "storage-limited water loss",
}
REGIME_COLORS = {
    "early_transient": "#C78D4B",
    "stageI_like": "#4E7E9E",
    "stageII_like": "#6FA38A",
    "mixed_or_uncertain": "#BFC3C7",
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
    "grid": "#E6EBEF",
}


@dataclass(frozen=True)
class EventSegment:
    start_idx: int
    end_idx: int
    order: int


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


def layer_label(layer: str) -> str:
    return LAYER_LABELS.get(str(layer), str(layer))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENT_AUDIT, parse_dates=["start", "end"])
    events["clean_48h"] = events["associated_48h"] & ~events["interrupted_by_rain"]
    points = pd.read_parquet(RAW_POINTS)
    meta_cols = ["event_id", "site_id", "layer", "associated_48h", "interrupted_by_rain", "clean_48h", "mean_loss_mm_h"]
    points = points.merge(events[meta_cols], on=["event_id", "site_id", "layer"], how="left", validate="many_to_one")
    return events, points


def segment_sse(t: np.ndarray, y: np.ndarray, start: int, end: int) -> float:
    tt = t[start:end]
    yy = y[start:end]
    if len(tt) < 2 or np.nanmax(tt) <= np.nanmin(tt):
        return float(np.nanvar(yy) * len(yy))
    coef = np.polyfit(tt, yy, 1)
    fitted = coef[0] * tt + coef[1]
    return float(np.nansum((yy - fitted) ** 2))


def candidate_partitions(n: int, k: int, min_points: int) -> list[list[tuple[int, int]]]:
    if k == 1:
        return [[(0, n)]]
    cuts_range = np.arange(min_points, n - min_points + 1, dtype=int)
    if len(cuts_range) > MAX_CANDIDATE_CUTS:
        cuts_range = np.unique(np.linspace(min_points, n - min_points, MAX_CANDIDATE_CUTS).round().astype(int))
    partitions = []
    for cuts in combinations(cuts_range, k - 1):
        bounds = (0, *cuts, n)
        if all(bounds[i + 1] - bounds[i] >= min_points for i in range(k)):
            partitions.append([(bounds[i], bounds[i + 1]) for i in range(k)])
    return partitions


def adaptive_piecewise_segments(event: pd.DataFrame) -> list[EventSegment]:
    event = event.sort_values("t_h")
    n = len(event)
    min_points = min(MIN_SEGMENT_POINTS, max(3, n // 3))
    if n < min_points:
        return []
    t = event["t_h"].to_numpy(float)
    y = event["moisture_mm"].to_numpy(float)
    max_k = min(MAX_SEGMENTS, n // min_points)
    best_score = np.inf
    best_partition: list[tuple[int, int]] | None = None
    for k in range(1, max_k + 1):
        for partition in candidate_partitions(n, k, min_points):
            sse = sum(segment_sse(t, y, start, end) for start, end in partition)
            # A BIC-like penalty keeps the algorithm from splitting every small curve change.
            score = n * np.log(max(sse / max(n, 1), 1e-8)) + (2 * k + k - 1) * np.log(max(n, 2))
            if score < best_score:
                best_score = score
                best_partition = partition
    if best_partition is None:
        return [EventSegment(0, n, 0)]
    return [EventSegment(start, end, order) for order, (start, end) in enumerate(best_partition)]


def slope_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1])


def segment_diagnostics(segment: pd.DataFrame, event_mean_loss: float) -> dict[str, float]:
    segment = segment.sort_values("t_h")
    y = segment["moisture_mm"].to_numpy(float)
    t = segment["t_h"].to_numpy(float)
    denom = float(segment["total_drop_mm"].iloc[0])
    event_end = float(segment["end_mm"].iloc[0])
    x = ((y - event_end) / denom).clip(0, 1) if denom > 0 else np.full_like(y, np.nan)
    dy = np.diff(y)
    dt = np.diff(t)
    loss = np.full_like(dt, np.nan, dtype=float)
    ok_dt = dt > 0
    loss[ok_dt] = -dy[ok_dt] / dt[ok_dt]
    x_mid = (x[:-1] + x[1:]) / 2.0 if len(x) > 1 else np.array([])
    ok_loss = np.isfinite(loss) & (loss > 0) & np.isfinite(x_mid)
    loss_pos = loss[ok_loss]
    x_pos = x_mid[ok_loss]
    slope, corr = slope_corr(x_pos, loss_pos)
    mean_loss = float(np.nanmean(loss_pos)) if len(loss_pos) else np.nan
    cv = float(np.nanstd(loss_pos) / mean_loss) if len(loss_pos) >= 3 and np.isfinite(mean_loss) and mean_loss > 0 else np.nan
    time_slope, time_corr = slope_corr(t[:-1][ok_loss] if len(t) > 1 else np.array([]), loss_pos)
    drop = float(y[0] - y[-1]) if len(y) else np.nan
    return {
        "segment_points": int(len(segment)),
        "segment_intervals": int(len(loss_pos)),
        "segment_start_mm": float(y[0]),
        "segment_end_mm": float(y[-1]),
        "segment_drop_mm": drop,
        "segment_drop_share": float(drop / denom) if denom > 0 else np.nan,
        "segment_duration_h": float(t[-1] - t[0]) if len(t) else np.nan,
        "segment_storage_start": float(x[0]) if len(x) else np.nan,
        "segment_storage_end": float(x[-1]) if len(x) else np.nan,
        "segment_storage_range": float(np.nanmax(x) - np.nanmin(x)) if len(x) else np.nan,
        "segment_mean_loss_mm_h": mean_loss,
        "segment_loss_to_event_loss": float(mean_loss / event_mean_loss) if event_mean_loss > 0 and np.isfinite(mean_loss) else np.nan,
        "segment_loss_storage_slope": slope,
        "segment_loss_storage_corr": corr,
        "segment_loss_cv": cv,
        "segment_loss_time_slope": time_slope,
        "segment_loss_time_corr": time_corr,
    }


def classify_regime(row: pd.Series) -> str:
    order = int(row["adaptive_segment_order"])
    storage_start = row["segment_storage_start"]
    drop_share = row["segment_drop_share"]
    loss_ratio = row["segment_loss_to_event_loss"]
    time_corr = row["segment_loss_time_corr"]
    storage_corr = row["segment_loss_storage_corr"]
    cv = row["segment_loss_cv"]
    storage_range = row["segment_storage_range"]

    early_like = (
        order == 0
        and np.isfinite(storage_start)
        and storage_start >= 0.55
        and (
            (np.isfinite(drop_share) and drop_share >= 0.30)
            or (np.isfinite(loss_ratio) and loss_ratio >= 1.25)
            or (np.isfinite(time_corr) and time_corr <= -0.30)
        )
    )
    if early_like:
        return "early_transient"
    if np.isfinite(storage_corr) and storage_corr >= 0.20 and np.isfinite(storage_range) and storage_range >= 0.05:
        return "stageII_like"
    if np.isfinite(storage_corr) and abs(storage_corr) < 0.20 and np.isfinite(cv) and cv < 0.60:
        return "stageI_like"
    return "mixed_or_uncertain"


def build_adaptive_segments(raw_points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = raw_points[raw_points["clean_48h"]].copy()
    segment_rows = []
    point_parts = []
    for (site_id, layer, event_id), event in clean.groupby(["site_id", "layer", "event_id"], sort=True):
        event = event.sort_values("t_h").reset_index(drop=True)
        if len(event) < MIN_SEGMENT_POINTS:
            continue
        pieces = adaptive_piecewise_segments(event)
        if not pieces:
            continue
        event_mean_loss = float(event["mean_loss_mm_h"].iloc[0])
        for piece in pieces:
            seg = event.iloc[piece.start_idx : piece.end_idx].copy()
            if len(seg) < 3:
                continue
            diag = segment_diagnostics(seg, event_mean_loss)
            adaptive_segment_id = f"{event_id}_seg{piece.order + 1:02d}"
            segment_rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "event_id": event_id,
                    "adaptive_segment_id": adaptive_segment_id,
                    "adaptive_segment_order": int(piece.order),
                    "n_adaptive_segments": len(pieces),
                    "start_t_h": float(seg["t_h"].iloc[0]),
                    "end_t_h": float(seg["t_h"].iloc[-1]),
                    "total_drop_mm": float(seg["total_drop_mm"].iloc[0]),
                    "event_duration_h": float(seg["duration_h"].iloc[0]),
                    "event_mean_loss_mm_h": event_mean_loss,
                    **diag,
                }
            )
            seg["parent_event_id"] = event_id
            seg["adaptive_segment_id"] = adaptive_segment_id
            seg["adaptive_segment_order"] = int(piece.order)
            seg["n_adaptive_segments"] = len(pieces)
            seg["segment_t_h"] = seg["t_h"] - float(seg["t_h"].iloc[0])
            denom = seg["total_drop_mm"].where(seg["total_drop_mm"] > 0)
            seg["event_storage_norm"] = ((seg["moisture_mm"] - seg["end_mm"]) / denom).clip(0, 1)
            point_parts.append(seg)
    segments = pd.DataFrame(segment_rows)
    if segments.empty:
        return segments, pd.DataFrame()
    segments["segment_regime"] = segments.apply(classify_regime, axis=1)
    points = pd.concat(point_parts, ignore_index=True) if point_parts else pd.DataFrame()
    points = points.merge(
        segments[["adaptive_segment_id", "segment_regime"]],
        on="adaptive_segment_id",
        how="inner",
        validate="many_to_one",
    )
    return segments, points


def build_csr_points(points: pd.DataFrame) -> pd.DataFrame:
    use = points[points["segment_regime"].isin(REGIME_ORDER)].copy()
    if use.empty:
        return use
    use["segment"] = use["segment_regime"]
    use["segment_label"] = use["segment"].map(REGIME_LABELS)
    use["segment_short"] = use["segment"].map(REGIME_LABELS)
    use["dominant_process"] = use["segment"].map(REGIME_PROCESS)
    use["event_id_original"] = use["event_id"]
    use["event_id"] = use["adaptive_segment_id"]
    counts = use.groupby("event_id")["moisture_mm"].transform("count")
    return use[counts >= 4].copy()


def fit_adaptive_csr(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned_outputs = []
    curve_outputs = []
    binned_outputs = []
    prediction_outputs = []
    edge_outputs = []
    component_outputs = []
    for (site_id, layer, segment), group in points.groupby(["site_id", "layer", "segment"], sort=True):
        group = group.sort_values(["event_id", "segment_t_h"]).copy()
        if group["event_id"].nunique() < MIN_LOCAL_SEGMENT_EVENTS or len(group) < MIN_LOCAL_SEGMENT_POINTS:
            continue
        aligned, edges, components = hydrologically_constrained_registration(group, REGISTRATION_CFG)
        if aligned.empty or aligned["event_id"].nunique() < MIN_LOCAL_SEGMENT_EVENTS:
            continue
        curve, binned = fit_segment_curve(aligned)
        if curve.empty:
            continue
        prediction = aligned[
            [
                "site_id",
                "event_id",
                "event_id_original",
                "parent_event_id",
                "adaptive_segment_id",
                "adaptive_segment_order",
                "layer",
                "segment",
                "segment_regime",
                "t_h",
                "segment_t_h",
                "event_storage_norm",
                "moisture_mm",
                "csr_x_h",
                "loss_rate_mm_h",
                "registration_component",
            ]
        ].copy()
        prediction["pred_mm"] = curve_predict(curve, prediction["csr_x_h"].to_numpy(float))
        for frame in (aligned, curve, binned, prediction):
            frame["site_id"] = int(site_id)
            frame["layer"] = layer
            frame["segment"] = segment
            frame["segment_label"] = REGIME_LABELS[segment]
            frame["dominant_process"] = REGIME_PROCESS[segment]
            frame["registration_method"] = "hydrologically_constrained_pairwise"
        for frame in (edges, components):
            if not frame.empty:
                frame["site_id"] = int(site_id)
                frame["layer"] = layer
                frame["segment"] = segment
                frame["segment_label"] = REGIME_LABELS[segment]
                frame["dominant_process"] = REGIME_PROCESS[segment]
                frame["registration_method"] = "hydrologically_constrained_pairwise"
        aligned_outputs.append(aligned)
        curve_outputs.append(curve)
        binned_outputs.append(binned)
        prediction_outputs.append(prediction)
        if not edges.empty:
            edge_outputs.append(edges)
        if not components.empty:
            component_outputs.append(components)
    aligned = pd.concat(aligned_outputs, ignore_index=True) if aligned_outputs else pd.DataFrame()
    curves = pd.concat(curve_outputs, ignore_index=True) if curve_outputs else pd.DataFrame()
    binned = pd.concat(binned_outputs, ignore_index=True) if binned_outputs else pd.DataFrame()
    predictions = pd.concat(prediction_outputs, ignore_index=True) if prediction_outputs else pd.DataFrame()
    edges = pd.concat(edge_outputs, ignore_index=True) if edge_outputs else pd.DataFrame()
    components = pd.concat(component_outputs, ignore_index=True) if component_outputs else pd.DataFrame()
    return aligned, curves, binned, predictions, edges, components


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 3:
        return {"ccc": np.nan, "r2": np.nan, "rmse_mm": np.nan, "mae_mm": np.nan, "bias_mm": np.nan}
    residual = pred - obs
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    sse = float(np.sum(residual**2))
    return {
        "ccc": concordance_correlation_coefficient(obs, pred),
        "r2": 1.0 - sse / sst if sst > 0 else np.nan,
        "rmse_mm": float(np.sqrt(np.nanmean(residual**2))),
        "mae_mm": float(np.nanmean(np.abs(residual))),
        "bias_mm": float(np.nanmean(residual)),
    }


def local_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site_id, layer, segment), data in predictions.groupby(["site_id", "layer", "segment"], sort=True):
        rows.append(
            {
                "site_id": int(site_id),
                "layer": layer,
                "layer_label": layer_label(layer),
                "segment": segment,
                "segment_label": REGIME_LABELS[segment],
                "dominant_process": REGIME_PROCESS[segment],
                "events": int(data["event_id"].nunique()),
                "parent_events": int(data["parent_event_id"].nunique()),
                "points": int(len(data)),
                **metrics(data["moisture_mm"].to_numpy(float), data["pred_mm"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def summarize_local(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    summary = (
        local.groupby(["layer", "layer_label", "segment", "segment_label", "dominant_process"])
        .agg(
            local_models=("site_id", "count"),
            median_events_per_model=("events", "median"),
            total_points=("points", "sum"),
            median_ccc=("ccc", "median"),
            median_r2=("r2", "median"),
            median_rmse_mm=("rmse_mm", "median"),
            median_mae_mm=("mae_mm", "median"),
            median_bias_mm=("bias_mm", "median"),
        )
        .reset_index()
    )
    summary["layer"] = pd.Categorical(summary["layer"], categories=LAYER_ORDER, ordered=True)
    summary["segment"] = pd.Categorical(summary["segment"], categories=REGIME_ORDER, ordered=True)
    return summary.sort_values(["layer", "segment"]).reset_index(drop=True)


def transition_boundaries(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta = (
        points.sort_values(["parent_event_id", "adaptive_segment_order"])
        .groupby(["site_id", "layer", "parent_event_id", "adaptive_segment_id", "segment"], sort=True)
        .agg(
            order=("adaptive_segment_order", "first"),
            start_mm=("moisture_mm", "first"),
            end_mm=("moisture_mm", "last"),
        )
        .reset_index()
        .sort_values(["site_id", "layer", "parent_event_id", "order"])
    )
    for (site_id, layer, parent_event_id), event in meta.groupby(["site_id", "layer", "parent_event_id"], sort=False):
        event = event.sort_values("order").reset_index(drop=True)
        for i in range(len(event) - 1):
            from_seg = event.loc[i, "segment"]
            to_seg = event.loc[i + 1, "segment"]
            if from_seg not in REGIME_ORDER or to_seg not in REGIME_ORDER or from_seg == to_seg:
                continue
            rows.append(
                {
                    "site_id": int(site_id),
                    "layer": layer,
                    "parent_event_id": parent_event_id,
                    "transition": f"{from_seg}->{to_seg}",
                    "from_segment": from_seg,
                    "to_segment": to_seg,
                    "boundary_mm": float((event.loc[i, "end_mm"] + event.loc[i + 1, "start_mm"]) / 2.0),
                }
            )
    return pd.DataFrame(rows)


def calibrate_composite_curves(curves: pd.DataFrame, points: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return curves
    out = []
    segment_ranges = (
        points.groupby(["site_id", "layer", "segment", "event_id"], sort=False)
        .agg(start_mm=("moisture_mm", "first"), end_mm=("moisture_mm", "last"))
        .reset_index()
    )
    for (site_id, layer), group in curves.groupby(["site_id", "layer"], sort=True):
        offset = 0.0
        local_bounds = boundaries[(boundaries["site_id"] == site_id) & (boundaries["layer"] == layer)]
        local_ranges = segment_ranges[(segment_ranges["site_id"] == site_id) & (segment_ranges["layer"] == layer)]
        local_points = points[(points["site_id"] == site_id) & (points["layer"] == layer)]
        observed_min = float(local_points["moisture_mm"].min()) if not local_points.empty else -np.inf
        observed_max = float(local_points["moisture_mm"].max()) if not local_points.empty else np.inf
        b_early_stage1 = local_bounds[local_bounds["transition"] == "early_transient->stageI_like"]["boundary_mm"].median()
        b_stage1_stage2 = local_bounds[local_bounds["transition"] == "stageI_like->stageII_like"]["boundary_mm"].median()
        b_early_stage2 = local_bounds[local_bounds["transition"] == "early_transient->stageII_like"]["boundary_mm"].median()
        for segment in REGIME_ORDER:
            curve = group[group["segment"] == segment].sort_values("csr_x_h").copy()
            if curve.empty:
                continue
            ranges = local_ranges[local_ranges["segment"] == segment]
            raw_start = float(curve["csr_mm"].iloc[0])
            raw_end = float(curve["csr_mm"].iloc[-1])
            median_start = float(ranges["start_mm"].median()) if not ranges.empty else raw_start
            median_end = float(ranges["end_mm"].median()) if not ranges.empty else raw_end
            target_start = median_start
            target_end = median_end
            if segment == "early_transient":
                if np.isfinite(b_early_stage1) and target_start > b_early_stage1:
                    target_end = float(b_early_stage1)
                elif np.isfinite(b_early_stage2) and target_start > b_early_stage2:
                    target_end = float(b_early_stage2)
            elif segment == "stageI_like":
                if np.isfinite(b_early_stage1) and b_early_stage1 > target_end:
                    target_start = float(b_early_stage1)
                if np.isfinite(b_stage1_stage2) and target_start > b_stage1_stage2:
                    target_end = float(b_stage1_stage2)
            elif segment == "stageII_like":
                if np.isfinite(b_stage1_stage2) and b_stage1_stage2 > target_end:
                    target_start = float(b_stage1_stage2)
                elif np.isfinite(b_early_stage2) and b_early_stage2 > target_end:
                    target_start = float(b_early_stage2)
            if not (np.isfinite(target_start) and np.isfinite(target_end) and target_start > target_end):
                target_start = median_start
                target_end = median_end
            if not (np.isfinite(target_start) and np.isfinite(target_end) and target_start > target_end):
                target_start = raw_start
                target_end = raw_end
            if raw_start > raw_end and target_start > target_end:
                scale = (target_start - target_end) / (raw_start - raw_end)
                curve["csr_mm_calibrated"] = target_end + (curve["csr_mm"] - raw_end) * scale
            else:
                curve["csr_mm_calibrated"] = curve["csr_mm"] + (target_start - raw_start)
            curve["csr_mm_calibrated"] = curve["csr_mm_calibrated"].clip(observed_min, observed_max)
            local_x = curve["csr_x_h"].to_numpy(float)
            local_x = local_x - np.nanmin(local_x)
            curve["composite_x_h"] = local_x + offset
            curve["target_start_mm"] = target_start
            curve["target_end_mm"] = target_end
            offset += float(np.nanmax(local_x)) + 0.25
            out.append(curve)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_aligned_composite_coordinate(aligned: pd.DataFrame, calibrated_curves: pd.DataFrame) -> pd.DataFrame:
    if aligned.empty or calibrated_curves.empty:
        return aligned
    outputs = []
    offsets = (
        calibrated_curves.groupby(["site_id", "layer", "segment"], sort=False)
        .agg(offset=("composite_x_h", "min"))
        .reset_index()
    )
    for (site_id, layer, segment), group in aligned.groupby(["site_id", "layer", "segment"], sort=False):
        match = offsets[(offsets["site_id"] == site_id) & (offsets["layer"] == layer) & (offsets["segment"] == segment)]
        if match.empty:
            continue
        piece = group.copy()
        local_x = piece["csr_x_h"].to_numpy(float)
        piece["composite_x_h"] = local_x - np.nanmin(local_x) + float(match["offset"].iloc[0])
        outputs.append(piece)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def deterministic_seed(site_id: int, layer: str, segment: str) -> int:
    value = 0
    for ch in f"{site_id}-{layer}-{segment}-{RANDOM_SEED}":
        value = (value * 131 + ord(ch)) % (2**32 - 1)
    return value


def split_event_ids(event_ids: np.ndarray, site_id: int, layer: str, segment: str) -> tuple[set[str], set[str]] | None:
    event_ids = np.array(sorted(event_ids), dtype=object)
    n_events = len(event_ids)
    if n_events < MIN_TRAIN_EVENTS + MIN_TEST_EVENTS:
        return None
    n_test = min(max(MIN_TEST_EVENTS, int(np.ceil(TEST_FRACTION * n_events))), n_events - MIN_TRAIN_EVENTS)
    if n_test < MIN_TEST_EVENTS:
        return None
    rng = np.random.default_rng(deterministic_seed(site_id, layer, segment))
    shuffled = event_ids.copy()
    rng.shuffle(shuffled)
    return set(str(x) for x in shuffled[n_test:]), set(str(x) for x in shuffled[:n_test])


def split_validation(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_rows = []
    prediction_parts = []
    for (site_id, layer, segment), group in points.groupby(["site_id", "layer", "segment"], sort=True):
        group = group.sort_values(["event_id", "segment_t_h"]).copy()
        split = split_event_ids(group["event_id"].unique(), int(site_id), str(layer), str(segment))
        if split is None:
            continue
        train_ids, test_ids = split
        train = group[group["event_id"].astype(str).isin(train_ids)].copy()
        test = group[group["event_id"].astype(str).isin(test_ids)].copy()
        if train["event_id"].nunique() < MIN_TRAIN_EVENTS or len(train) < MIN_TRAIN_POINTS:
            continue
        if test["event_id"].nunique() < MIN_TEST_EVENTS or len(test) < MIN_TEST_POINTS:
            continue
        train_aligned, train_edges, train_components = hydrologically_constrained_registration(train, REGISTRATION_CFG)
        if train_aligned.empty:
            continue
        curve, _ = fit_segment_curve(train_aligned)
        if curve.empty:
            continue
        test_aligned = align_to_registered_template(test, train_aligned, REGISTRATION_CFG)
        if test_aligned.empty:
            continue
        test_aligned["pred_mm"] = curve_predict(curve, test_aligned["csr_x_h"].to_numpy(float))
        values = metrics(test_aligned["moisture_mm"].to_numpy(float), test_aligned["pred_mm"].to_numpy(float))
        local_rows.append(
            {
                "site_id": int(site_id),
                "layer": layer,
                "layer_label": layer_label(layer),
                "segment": segment,
                "segment_label": REGIME_LABELS[segment],
                "train_segments": len(train_ids),
                "test_segments": len(test_ids),
                "train_pairwise_edges": int(len(train_edges)),
                "train_registration_components": int(len(train_components)),
                "test_points": int(len(test_aligned)),
                **values,
            }
        )
        prediction_parts.append(test_aligned)
    local = pd.DataFrame(local_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    return local, predictions


def summarize_validation(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    layer_rows = []
    segment_rows = []
    for layer, data in predictions.groupby("layer", sort=True):
        layer_rows.append(
            {
                "layer": layer,
                "layer_label": layer_label(layer),
                "sites": int(data["site_id"].nunique()),
                "test_segments": int(data["event_id"].nunique()),
                "test_points": int(len(data)),
                **metrics(data["moisture_mm"].to_numpy(float), data["pred_mm"].to_numpy(float)),
            }
        )
    for (layer, segment), data in predictions.groupby(["layer", "segment"], sort=True):
        segment_rows.append(
            {
                "layer": layer,
                "layer_label": layer_label(layer),
                "segment": segment,
                "segment_label": REGIME_LABELS[segment],
                "sites": int(data["site_id"].nunique()),
                "test_segments": int(data["event_id"].nunique()),
                "test_points": int(len(data)),
                **metrics(data["moisture_mm"].to_numpy(float), data["pred_mm"].to_numpy(float)),
            }
        )
    return pd.DataFrame(layer_rows), pd.DataFrame(segment_rows)


def segment_composition(segments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        segments.groupby(["layer", "adaptive_segment_order", "segment_regime"])
        .size()
        .reset_index(name="segments")
    )
    totals = counts.groupby(["layer", "adaptive_segment_order"])["segments"].transform("sum")
    counts["share"] = counts["segments"] / totals
    counts["layer_label"] = counts["layer"].map(LAYER_LABELS)
    return counts.sort_values(["layer", "adaptive_segment_order", "segment_regime"])


def construction_pool(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in LAYER_ORDER:
        for regime in REGIME_ORDER:
            d = points[(points["layer"] == layer) & (points["segment"] == regime)]
            rows.append(
                {
                    "layer": layer,
                    "layer_label": layer_label(layer),
                    "segment": regime,
                    "segment_label": REGIME_LABELS[regime],
                    "dominant_process": REGIME_PROCESS[regime],
                    "segments": int(d["event_id"].nunique()),
                    "parent_events": int(d["parent_event_id"].nunique()) if "parent_event_id" in d.columns else 0,
                    "points": int(len(d)),
                }
            )
    return pd.DataFrame(rows)


def choose_representative(local: pd.DataFrame, curves: pd.DataFrame) -> tuple[int, str]:
    complete = local[(local["layer"] == REP_LAYER) & (local["site_id"] != EXCLUDE_REP_SITE)].copy()
    counts = complete.groupby("site_id")["segment"].nunique()
    candidates = counts[counts >= 2].index.tolist()
    if not candidates:
        if counts.empty:
            raise ValueError("No representative candidates found.")
        return int(counts.sort_values(ascending=False).index[0]), REP_LAYER
    site_summary = (
        complete[complete["site_id"].isin(candidates)]
        .groupby("site_id")
        .agg(regimes=("segment", "nunique"), total_segments=("events", "sum"), median_rmse_mm=("rmse_mm", "median"))
        .reset_index()
    )
    site_summary["score"] = -site_summary["regimes"] * 10 - 0.01 * site_summary["total_segments"] + site_summary["median_rmse_mm"]
    site_summary = site_summary.sort_values(["score", "site_id"])
    site_summary.to_csv(SOURCE / "adaptive_representative_site_candidates.csv", index=False)
    return int(site_summary.iloc[0]["site_id"]), REP_LAYER


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)


def plot_representative(curves: pd.DataFrame, aligned: pd.DataFrame, local: pd.DataFrame, rep_site: int, rep_layer: str) -> None:
    c = curves[(curves["site_id"] == rep_site) & (curves["layer"] == rep_layer)].copy()
    a = aligned[(aligned["site_id"] == rep_site) & (aligned["layer"] == rep_layer)].copy()
    m = local[(local["site_id"] == rep_site) & (local["layer"] == rep_layer)].copy()
    if c.empty:
        return
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], wspace=0.42, hspace=0.48)
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_process = fig.add_subplot(gs[0, 1])
    ax_support = fig.add_subplot(gs[1, 1])
    for regime in REGIME_ORDER:
        curve = c[c["segment"] == regime].sort_values("composite_x_h")
        points = a[a["segment"] == regime]
        if curve.empty:
            continue
        color = REGIME_COLORS[regime]
        if not points.empty:
            sample = points.sample(n=min(1600, len(points)), random_state=RANDOM_SEED) if len(points) > 1600 else points
            ax_curve.scatter(sample["composite_x_h"], sample["moisture_mm"], s=5, color=color, alpha=0.13, linewidths=0)
        ax_curve.plot(curve["composite_x_h"], curve["csr_mm_calibrated"], color=color, lw=2.1, label=REGIME_LABELS[regime])
    ax_curve.set_title(f"Adaptive regime-composite CSR: site {rep_site}, {layer_label(rep_layer)}", loc="left", fontsize=9, pad=6)
    ax_curve.set_xlabel("Calibrated composite CSR coordinate")
    ax_curve.set_ylabel("Soil water amount (mm)")
    ax_curve.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax_curve.legend(loc="upper right", fontsize=6.6)
    add_panel_label(ax_curve, "a", x=-0.12)

    ax_process.axis("off")
    ax_process.set_title("Adaptive regime interpretation", loc="left", fontsize=8, pad=6)
    y = 0.88
    for regime in REGIME_ORDER:
        ax_process.scatter([0.02], [y], s=34, color=REGIME_COLORS[regime], transform=ax_process.transAxes)
        ax_process.text(0.08, y, REGIME_LABELS[regime], transform=ax_process.transAxes, fontsize=7.2, fontweight="bold", va="center")
        ax_process.text(0.08, y - 0.11, REGIME_PROCESS[regime], transform=ax_process.transAxes, fontsize=6.5, color=COLORS["neutral_mid"], va="top", wrap=True)
        y -= 0.30
    add_panel_label(ax_process, "b")

    if not m.empty:
        m["segment"] = pd.Categorical(m["segment"], categories=REGIME_ORDER, ordered=True)
        m = m.sort_values("segment")
        x = np.arange(len(m))
        ax_support.bar(x, m["events"], color=[REGIME_COLORS[s] for s in m["segment"].astype(str)], width=0.65)
        for xi, val in zip(x, m["events"]):
            ax_support.text(xi, val + max(m["events"]) * 0.03, f"{int(val)}", ha="center", va="bottom", fontsize=6.4)
        ax_support.set_xticks(x, [REGIME_LABELS[s].replace("-like", "") for s in m["segment"].astype(str)])
        ax_support.set_ylabel("Adaptive segments")
        ax_support.set_title("Local support", loc="left", fontsize=8, pad=6)
        ax_support.grid(axis="y", color=COLORS["grid"], lw=0.7)
    add_panel_label(ax_support, "c")
    save_pub(fig, "fig_adaptive_regime_representative_csr")
    plt.close(fig)


def plot_network_summary(pool: pd.DataFrame, summary: pd.DataFrame, val_segment: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()
    x = np.arange(len(LAYER_ORDER))
    width = 0.24
    for i, regime in enumerate(REGIME_ORDER):
        d = pool[pool["segment"] == regime].set_index("layer").reindex(LAYER_ORDER)
        ax1.bar(x + (i - 1) * width, d["segments"].fillna(0), width=width, color=REGIME_COLORS[regime], label=REGIME_LABELS[regime])
    ax1.set_xticks(x, [LAYER_LABELS[l] for l in LAYER_ORDER])
    ax1.set_ylabel("Adaptive event-segments")
    ax1.set_title("Regime-specific construction pools", loc="left", fontsize=8, pad=6)
    ax1.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax1.legend(fontsize=6.4)
    add_panel_label(ax1, "a")

    if not summary.empty:
        pivot = summary.pivot_table(index="segment", columns="layer", values="local_models", aggfunc="first").reindex(REGIME_ORDER, columns=LAYER_ORDER)
        im = ax2.imshow(pivot.to_numpy(float), cmap="Blues", aspect="auto")
        ax2.set_xticks(np.arange(len(LAYER_ORDER)), [LAYER_LABELS[l] for l in LAYER_ORDER])
        ax2.set_yticks(np.arange(len(REGIME_ORDER)), [REGIME_LABELS[r] for r in REGIME_ORDER])
        for iy in range(pivot.shape[0]):
            for ix in range(pivot.shape[1]):
                val = pivot.iloc[iy, ix]
                if np.isfinite(val):
                    ax2.text(ix, iy, f"{int(val)}", ha="center", va="center", fontsize=6.5, color="white" if val >= 18 else "#111")
        ax2.set_title("Location-layer models fitted", loc="left", fontsize=8, pad=6)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02, label="models")
    add_panel_label(ax2, "b")

    if not summary.empty:
        pivot_rmse = summary.pivot_table(index="segment", columns="layer", values="median_rmse_mm", aggfunc="first").reindex(REGIME_ORDER, columns=LAYER_ORDER)
        im2 = ax3.imshow(pivot_rmse.to_numpy(float), cmap="YlGnBu", aspect="auto")
        ax3.set_xticks(np.arange(len(LAYER_ORDER)), [LAYER_LABELS[l] for l in LAYER_ORDER])
        ax3.set_yticks(np.arange(len(REGIME_ORDER)), [REGIME_LABELS[r] for r in REGIME_ORDER])
        for iy in range(pivot_rmse.shape[0]):
            for ix in range(pivot_rmse.shape[1]):
                val = pivot_rmse.iloc[iy, ix]
                if np.isfinite(val):
                    ax3.text(ix, iy, f"{val:.2f}", ha="center", va="center", fontsize=6.5)
        ax3.set_title("Median local fit RMSE (mm)", loc="left", fontsize=8, pad=6)
        fig.colorbar(im2, ax=ax3, fraction=0.046, pad=0.02, label="mm")
    add_panel_label(ax3, "c")

    if not val_segment.empty:
        val = val_segment.copy()
        val = val[(val["sites"] >= 3) & (val["test_segments"] >= 10)].copy()
        val["layer"] = pd.Categorical(val["layer"], categories=LAYER_ORDER, ordered=True)
        for regime in REGIME_ORDER:
            d = val[val["segment"] == regime].sort_values("layer")
            if not d.empty:
                ax4.plot(d["layer_label"], d["ccc"], color=REGIME_COLORS[regime], marker="o", lw=1.6, label=REGIME_LABELS[regime])
        ax4.set_ylim(0, 1.03)
        ax4.set_ylabel("Held-out CCC")
        ax4.set_title("80/20 validation by adaptive regime", loc="left", fontsize=8, pad=6)
        ax4.grid(axis="y", color=COLORS["grid"], lw=0.7)
        ax4.legend(fontsize=6.4)
    add_panel_label(ax4, "d")
    save_pub(fig, "fig_adaptive_regime_network_summary")
    plt.close(fig)


def write_report(
    composition: pd.DataFrame,
    pool: pd.DataFrame,
    summary: pd.DataFrame,
    val_layer: pd.DataFrame,
    rep_site: int,
    rep_layer: str,
) -> None:
    lines = [
        "# Experiment 3: Adaptive within-event regime CSR",
        "",
        "## Material Passport",
        "",
        "- Type: code experiment result",
        "- Status: completed",
        "- Data unit: clean rainfall-associated SMDE; adaptive within-event segment",
        "- Main change from prior run: regime boundaries are inferred from each event's soil-moisture trajectory and loss-storage diagnostics, not from fixed elapsed-time windows.",
        "",
        "## Experiment Design",
        "",
        "Each SMDE was segmented with an adaptive piecewise-linear fit to the soil-water decline curve. Each resulting segment was then diagnosed using its own loss-storage relation.",
        "Segments were classified as early transient when the first wet segment had a large drop share, elevated loss rate, or rapid loss-rate relaxation; as stage-II-like when loss increased with storage; and as stage-I-like when loss was approximately storage-invariant with low variability.",
        "Regime-specific CSR curves were fitted separately by location, layer, and adaptive regime using hydrologically constrained registration. Segment pairs were registered only when they had sufficient storage overlap and similar loss rates; disconnected low-support segments were not forced onto the same curve.",
        "",
        "## Adaptive Regime Composition",
        "",
        composition.to_markdown(index=False),
        "",
        "## CSR Construction Pools",
        "",
        pool.to_markdown(index=False),
        "",
        "## Local Model Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No local models met support thresholds.",
        "",
        "## Held-out 80/20 Validation by Layer",
        "",
        val_layer.to_markdown(index=False) if not val_layer.empty else "No validation models met split thresholds.",
        "",
        f"Representative example: FAWN site {rep_site}, {layer_label(rep_layer)}.",
        "",
        "Interpretation note: the adaptive labels diagnose dominant segment behavior from soil moisture alone. They are not direct flux partitions.",
    ]
    (OUT / "experiment3_adaptive_regime_csr_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out()
    _, raw_points = load_inputs()
    segments, segment_points = build_adaptive_segments(raw_points)
    csr_points = build_csr_points(segment_points)
    aligned, curves, binned, predictions, registration_edges, registration_components = fit_adaptive_csr(csr_points)
    local = local_metrics(predictions)
    summary = summarize_local(local)
    boundaries = transition_boundaries(csr_points)
    calibrated_curves = calibrate_composite_curves(curves, csr_points, boundaries)
    aligned_composite = add_aligned_composite_coordinate(aligned, calibrated_curves)
    val_local, val_predictions = split_validation(csr_points)
    val_layer, val_segment = summarize_validation(val_predictions)
    composition = segment_composition(segments)
    pool = construction_pool(csr_points)
    rep_site, rep_layer = choose_representative(local, calibrated_curves)
    plot_representative(calibrated_curves, aligned_composite, local, rep_site, rep_layer)
    plot_network_summary(pool, summary, val_segment)

    segments.to_csv(SOURCE / "adaptive_segment_diagnostics.csv", index=False)
    segment_points.to_parquet(SOURCE / "adaptive_segment_points_all.parquet", index=False)
    csr_points.to_parquet(SOURCE / "adaptive_regime_csr_points.parquet", index=False)
    composition.to_csv(SOURCE / "adaptive_regime_composition_by_order.csv", index=False)
    pool.to_csv(SOURCE / "adaptive_regime_construction_pool_summary.csv", index=False)
    aligned.to_parquet(SOURCE / "adaptive_regime_aligned_points.parquet", index=False)
    aligned_composite.to_parquet(SOURCE / "adaptive_regime_aligned_points_composite.parquet", index=False)
    curves.to_csv(SOURCE / "adaptive_regime_curves_raw.csv", index=False)
    calibrated_curves.to_csv(SOURCE / "adaptive_regime_curves_calibrated.csv", index=False)
    binned.to_csv(SOURCE / "adaptive_regime_binned_points.csv", index=False)
    predictions.to_parquet(SOURCE / "adaptive_regime_predictions.parquet", index=False)
    registration_edges.to_csv(SOURCE / "adaptive_regime_registration_edges.csv", index=False)
    registration_components.to_csv(SOURCE / "adaptive_regime_registration_components.csv", index=False)
    local.to_csv(SOURCE / "adaptive_regime_local_metrics.csv", index=False)
    summary.to_csv(SOURCE / "adaptive_regime_local_summary_by_layer_segment.csv", index=False)
    boundaries.to_csv(SOURCE / "adaptive_regime_transition_boundaries.csv", index=False)
    val_local.to_csv(SOURCE / "adaptive_regime_split_validation_local_metrics.csv", index=False)
    val_layer.to_csv(SOURCE / "adaptive_regime_split_validation_pooled_by_layer.csv", index=False)
    val_segment.to_csv(SOURCE / "adaptive_regime_split_validation_pooled_by_layer_segment.csv", index=False)
    if not val_predictions.empty:
        val_predictions.to_parquet(SOURCE / "adaptive_regime_split_validation_predictions.parquet", index=False)
    write_report(composition, pool, summary, val_layer, rep_site, rep_layer)
    print(OUT)


if __name__ == "__main__":
    main()
