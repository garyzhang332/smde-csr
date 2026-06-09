from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

from fawn_segmented_csr import concordance_correlation_coefficient, fit_segment_curve
from hydro_csr_registration import RegistrationConfig, hydrologically_constrained_registration


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
SEGMENT_POINTS = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data" / "adaptive_segment_points_all.parquet"
EVENT_AUDIT = ANALYSIS / "fawn_full_smde_audit" / "full_smde_event_audit.csv"
OUT = ANALYSIS / "experiment4_regime_conditioned_forecast"
SOURCE = OUT / "source_data"

RANDOM_SEED = 20260605
HORIZONS_H = [1, 3, 6, 12, 24]
HISTORY_H = 1.0
ORIGIN_STEP_H = 0.5
TRAIN_FRACTION = 0.80
MIN_TRAIN_EVENTS = 8
MIN_TEST_EVENTS = 2
MIN_ANALOG_ROWS = 25
K_NEIGHBORS = 75
CSR_STORAGE_TOLERANCE_MM = 0.50
CSR_MAX_EXTRAPOLATION_H = 6.0
POOLED_MAX_SEGMENTS = 240
CSR_REGISTRATION_CFG = RegistrationConfig()

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
LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "recent_slope": "Recent-slope",
    "registered_csr_operator": "Registered CSR operator",
    "nonregime_analog": "Non-regime analog",
    "online_regime_analog": "Hard-regime analog",
    "regime_mixture_analog": "Regime-mixture analog",
    "oracle_regime_analog": "Diagnosed-regime analog",
}
MODEL_COLORS = {
    "persistence": "#8A8A8A",
    "recent_slope": "#6B5B95",
    "registered_csr_operator": "#C67B2E",
    "nonregime_analog": "#2F6F9F",
    "online_regime_analog": "#B65F2A",
    "regime_mixture_analog": "#D69C3A",
    "oracle_regime_analog": "#3D8B67",
}
MODEL_ORDER = [
    "persistence",
    "recent_slope",
    "registered_csr_operator",
    "nonregime_analog",
    "online_regime_analog",
    "regime_mixture_analog",
    "oracle_regime_analog",
]
FEATURE_COLS = ["s0_mm", "recent_loss_1h", "drop_sofar_mm", "origin_t_h", "start_mm"]

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
    for obsolete in [
        SOURCE / "forecast_site_layer_thresholds.csv",
        SOURCE / "forecast_threshold_metrics_by_layer_horizon.csv",
    ]:
        if obsolete.exists():
            obsolete.unlink()


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
    points = pd.read_parquet(SEGMENT_POINTS)
    points = points[points["clean_48h"].fillna(False)].copy()
    points = points[points["segment_regime"].isin(REGIME_ORDER)].copy()
    events_cols = [
        "site_id",
        "layer",
        "event_id",
        "start",
        "end",
        "rain_before_48",
        "rain_during",
        "start_month",
        "start_doy",
    ]
    points = points.merge(events[events_cols], on=["site_id", "layer", "event_id"], how="left", validate="many_to_one")
    points["layer"] = pd.Categorical(points["layer"], categories=LAYER_ORDER, ordered=True)
    points = points.sort_values(["site_id", "layer", "event_id", "t_h"]).reset_index(drop=True)
    return events, points


def chronological_event_split(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_index = (
        points.groupby(["site_id", "layer", "event_id"], observed=True)
        .agg(start=("start", "first"), duration_h=("duration_h", "first"), points=("moisture_mm", "size"))
        .reset_index()
        .sort_values(["site_id", "layer", "start", "event_id"])
    )
    split_rows = []
    for (site_id, layer), group in event_index.groupby(["site_id", "layer"], observed=True):
        group = group.sort_values(["start", "event_id"]).reset_index(drop=True)
        n_events = len(group)
        if n_events < MIN_TRAIN_EVENTS + MIN_TEST_EVENTS:
            continue
        n_train = int(np.floor(n_events * TRAIN_FRACTION))
        n_train = max(MIN_TRAIN_EVENTS, n_train)
        if n_events - n_train < MIN_TEST_EVENTS:
            n_train = n_events - MIN_TEST_EVENTS
        if n_train < MIN_TRAIN_EVENTS or n_events - n_train < MIN_TEST_EVENTS:
            continue
        group = group.copy()
        group["split"] = np.where(np.arange(n_events) < n_train, "train", "test")
        split_rows.append(group)
    if not split_rows:
        raise RuntimeError("No site-layer groups satisfy the event split requirements.")
    split = pd.concat(split_rows, ignore_index=True)
    split_summary = (
        split.groupby(["site_id", "layer", "split"], observed=True)
        .agg(events=("event_id", "nunique"), points=("points", "sum"), first_start=("start", "min"), last_start=("start", "max"))
        .reset_index()
    )
    return split[["site_id", "layer", "event_id", "split"]], split_summary


def origin_rows_for_event(event: pd.DataFrame, split: str) -> list[dict[str, object]]:
    event = event.sort_values("t_h")
    t = event["t_h"].to_numpy(float)
    y = event["moisture_mm"].to_numpy(float)
    if len(t) < 6 or np.nanmax(t) < HISTORY_H + min(HORIZONS_H):
        return []
    start_mm = float(event["start_mm"].iloc[0])
    rows: list[dict[str, object]] = []
    valid_origin_mask = np.isclose((t / ORIGIN_STEP_H) % 1, 0, atol=1e-6) | np.isclose((t / ORIGIN_STEP_H) % 1, 1, atol=1e-6)
    for idx, t0 in enumerate(t):
        if t0 < HISTORY_H or not valid_origin_mask[idx]:
            continue
        s0 = float(y[idx])
        prev = float(np.interp(t0 - HISTORY_H, t, y))
        recent_loss = max(0.0, (prev - s0) / HISTORY_H)
        regime = str(event["segment_regime"].iloc[idx])
        if regime not in REGIME_ORDER:
            continue
        for horizon in HORIZONS_H:
            target_t = t0 + horizon
            if target_t > float(t[-1]) + 1e-9:
                continue
            target = float(np.interp(target_t, t, y))
            rows.append(
                {
                    "site_id": int(event["site_id"].iloc[0]),
                    "layer": str(event["layer"].iloc[0]),
                    "event_id": str(event["event_id"].iloc[0]),
                    "origin_id": f"{event['event_id'].iloc[0]}|{t0:.2f}",
                    "split": split,
                    "start": event["start"].iloc[0],
                    "origin_t_h": float(t0),
                    "horizon_h": int(horizon),
                    "target_t_h": float(target_t),
                    "s0_mm": s0,
                    "target_mm": target,
                    "delta_mm": target - s0,
                    "start_mm": start_mm,
                    "drop_sofar_mm": max(0.0, start_mm - s0),
                    "recent_loss_1h": recent_loss,
                    "true_regime": regime,
                    "adaptive_segment_order": int(event["adaptive_segment_order"].iloc[idx]),
                    "rain_before_48": float(event["rain_before_48"].iloc[0]) if pd.notna(event["rain_before_48"].iloc[0]) else np.nan,
                    "rain_during": float(event["rain_during"].iloc[0]) if pd.notna(event["rain_during"].iloc[0]) else np.nan,
                    "start_month": int(event["start_month"].iloc[0]) if pd.notna(event["start_month"].iloc[0]) else -1,
                }
            )
    return rows


def build_forecast_origin_table(points: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    points = points.merge(split, on=["site_id", "layer", "event_id"], how="inner", validate="many_to_one")
    rows: list[dict[str, object]] = []
    group_cols = ["site_id", "layer", "event_id", "split"]
    for _, event in points.groupby(group_cols, observed=True, sort=False):
        rows.extend(origin_rows_for_event(event, str(event["split"].iloc[0])))
    origins = pd.DataFrame(rows)
    if origins.empty:
        raise RuntimeError("No forecast origins were created.")
    origins["layer"] = pd.Categorical(origins["layer"], categories=LAYER_ORDER, ordered=True)
    origins["true_regime"] = pd.Categorical(origins["true_regime"], categories=REGIME_ORDER, ordered=True)
    origins = origins.sort_values(["split", "site_id", "layer", "event_id", "origin_t_h", "horizon_h"]).reset_index(drop=True)
    origins["forecast_id"] = np.arange(len(origins), dtype=int)
    return origins


def classify_online_regime(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_origins = train_rows.drop_duplicates("origin_id").copy()
    test_origins = test_rows.drop_duplicates("origin_id").copy()
    feature_frame = pd.concat(
        [
            train_origins[["origin_id", "true_regime", "layer", *FEATURE_COLS]].assign(part="train"),
            test_origins[["origin_id", "true_regime", "layer", *FEATURE_COLS]].assign(part="test"),
        ],
        ignore_index=True,
    )
    x_all = pd.get_dummies(feature_frame[["layer", *FEATURE_COLS]], columns=["layer"], dtype=float)
    train_mask = feature_frame["part"].eq("train").to_numpy()
    x_train = x_all.loc[train_mask]
    x_test = x_all.loc[~train_mask]
    y_train = feature_frame.loc[train_mask, "true_regime"].astype(str)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    prob_cols = [f"prob_{cls}" for cls in clf.classes_]
    pred_table = feature_frame.loc[~train_mask, ["origin_id", "true_regime"]].copy()
    pred_table["predicted_regime"] = pred
    pred_table = pd.concat([pred_table.reset_index(drop=True), pd.DataFrame(prob, columns=prob_cols)], axis=1)
    for regime in REGIME_ORDER:
        col = f"prob_{regime}"
        if col not in pred_table.columns:
            pred_table[col] = 0.0

    y_true = pred_table["true_regime"].astype(str).to_numpy()
    y_pred = pred_table["predicted_regime"].astype(str).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=REGIME_ORDER)
    cm_rows = []
    for i, true_regime in enumerate(REGIME_ORDER):
        row_total = int(cm[i, :].sum())
        for j, pred_regime in enumerate(REGIME_ORDER):
            cm_rows.append(
                {
                    "true_regime": true_regime,
                    "predicted_regime": pred_regime,
                    "origins": int(cm[i, j]),
                    "row_share": float(cm[i, j] / row_total) if row_total else np.nan,
                }
            )
    summary = pd.DataFrame(
        [
            {
                "test_origins": int(len(pred_table)),
                "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else np.nan,
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
            }
        ]
    )
    classifier_metrics = pd.concat([summary.assign(record="summary"), pd.DataFrame(cm_rows).assign(record="confusion")], ignore_index=True)
    return pred_table, classifier_metrics


def query_analog_pool(pool: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pool = pool.dropna(subset=FEATURE_COLS + ["delta_mm"])
    query = query.dropna(subset=FEATURE_COLS)
    n_pool = len(pool)
    if n_pool < MIN_ANALOG_ROWS or query.empty:
        return np.array([]), np.array([]), np.array([]), n_pool
    x_pool = pool[FEATURE_COLS].to_numpy(float)
    x_query = query[FEATURE_COLS].to_numpy(float)
    center = np.nanmean(x_pool, axis=0)
    scale = np.nanstd(x_pool, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    x_pool = (x_pool - center) / scale
    x_query = (x_query - center) / scale
    k = min(K_NEIGHBORS, n_pool)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_pool)
    _, indices = nn.kneighbors(x_query)
    deltas = pool["delta_mm"].to_numpy(float)
    neighbor_delta = deltas[indices]
    return (
        np.nanmedian(neighbor_delta, axis=1),
        np.nanpercentile(neighbor_delta, 10, axis=1),
        np.nanpercentile(neighbor_delta, 90, axis=1),
        n_pool,
    )


def group_key_iter(frame: pd.DataFrame, cols: list[str]):
    if not cols:
        yield (), frame.index.to_numpy()
        return
    for key, idx in frame.groupby(cols, observed=True, sort=False).groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        yield key, np.asarray(list(idx), dtype=int)


def matching_pool(frame: pd.DataFrame, cols: list[str], key: tuple[object, ...]) -> pd.DataFrame:
    pool = frame
    for col, value in zip(cols, key):
        pool = pool[pool[col].eq(value)]
    return pool


def analog_predict(train: pd.DataFrame, test: pd.DataFrame, regime_col: str | None, pred_name: str) -> pd.DataFrame:
    out = test[["forecast_id", "site_id", "layer", "horizon_h", *FEATURE_COLS]].copy()
    out["pred_delta_mm"] = np.nan
    out["q10_delta_mm"] = np.nan
    out["q90_delta_mm"] = np.nan
    out["analog_level"] = ""
    out["analog_candidates"] = 0

    train_work = train.copy()
    test_work = test.copy()
    if regime_col is not None:
        train_work["regime_key"] = train_work["true_regime"].astype(str)
        test_work["regime_key"] = test_work[regime_col].astype(str)
        levels = [
            ("site-layer-regime", ["site_id", "layer", "regime_key"]),
            ("layer-regime", ["layer", "regime_key"]),
            ("site-layer", ["site_id", "layer"]),
            ("layer", ["layer"]),
            ("network", []),
        ]
    else:
        levels = [
            ("site-layer", ["site_id", "layer"]),
            ("layer", ["layer"]),
            ("network", []),
        ]

    for horizon in HORIZONS_H:
        train_h = train_work[train_work["horizon_h"].eq(horizon)]
        if train_h.empty:
            continue
        horizon_idx = test_work.index[test_work["horizon_h"].eq(horizon)].to_numpy()
        pending = np.intersect1d(horizon_idx, out.index[out["pred_delta_mm"].isna()].to_numpy())
        for level_name, cols in levels:
            if len(pending) == 0:
                break
            pending_frame = test_work.loc[pending]
            for key, idx in group_key_iter(pending_frame, cols):
                pool = matching_pool(train_h, cols, key)
                if len(pool) < MIN_ANALOG_ROWS:
                    continue
                query = test_work.loc[idx]
                pred_delta, q10, q90, n_pool = query_analog_pool(pool, query)
                if len(pred_delta) == 0:
                    continue
                out.loc[idx, "pred_delta_mm"] = pred_delta
                out.loc[idx, "q10_delta_mm"] = q10
                out.loc[idx, "q90_delta_mm"] = q90
                out.loc[idx, "analog_level"] = level_name
                out.loc[idx, "analog_candidates"] = int(n_pool)
            pending = np.intersect1d(horizon_idx, out.index[out["pred_delta_mm"].isna()].to_numpy())

    out["model"] = pred_name
    out["pred_mm"] = test["s0_mm"].to_numpy(float) + out["pred_delta_mm"].to_numpy(float)
    out["q10_mm"] = test["s0_mm"].to_numpy(float) + out["q10_delta_mm"].to_numpy(float)
    out["q90_mm"] = test["s0_mm"].to_numpy(float) + out["q90_delta_mm"].to_numpy(float)
    return out[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]]


def regime_mixture_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    base = test[["forecast_id", "s0_mm"]].copy()
    numerator = np.zeros(len(base), dtype=float)
    denominator = np.zeros(len(base), dtype=float)
    candidate_sum = np.zeros(len(base), dtype=float)
    for regime in REGIME_ORDER:
        forced = test.copy()
        forced["forced_regime"] = regime
        regime_pred = analog_predict(train, forced, "forced_regime", f"forced_{regime}")
        regime_pred = regime_pred.sort_values("forecast_id")
        prob = test[f"prob_{regime}"].fillna(0.0).to_numpy(float)
        pred_delta = regime_pred["pred_mm"].to_numpy(float) - base["s0_mm"].to_numpy(float)
        ok = np.isfinite(pred_delta) & np.isfinite(prob) & (prob > 0)
        numerator[ok] += prob[ok] * pred_delta[ok]
        denominator[ok] += prob[ok]
        candidate_sum[ok] += prob[ok] * regime_pred["analog_candidates"].to_numpy(float)[ok]
    pred = base[["forecast_id"]].copy()
    pred["model"] = "regime_mixture_analog"
    pred["pred_mm"] = np.nan
    ok = denominator > 0
    pred.loc[ok, "pred_mm"] = base.loc[ok, "s0_mm"].to_numpy(float) + numerator[ok] / denominator[ok]
    pred["q10_mm"] = np.nan
    pred["q90_mm"] = np.nan
    pred["analog_level"] = "regime-probability mixture"
    pred["analog_candidates"] = np.where(ok, candidate_sum / np.maximum(denominator, 1e-12), 0).round().astype(int)
    return pred[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]]


def cap_registration_segments(group: pd.DataFrame, max_segments: int = POOLED_MAX_SEGMENTS) -> pd.DataFrame:
    segment_ids = group["event_id"].drop_duplicates().astype(str).to_numpy()
    if len(segment_ids) <= max_segments:
        return group
    summary = (
        group.sort_values(["event_id", "segment_t_h"])
        .groupby("event_id", observed=True)
        .agg(start_mm=("moisture_mm", "first"), end_mm=("moisture_mm", "last"), points=("moisture_mm", "size"))
        .reset_index()
        .sort_values(["start_mm", "end_mm"], ascending=[False, False])
    )
    keep_idx = np.unique(np.linspace(0, len(summary) - 1, max_segments).round().astype(int))
    keep = set(summary.iloc[keep_idx]["event_id"].astype(str))
    return group[group["event_id"].astype(str).isin(keep)].copy()


def fit_registered_curve_group(
    group: pd.DataFrame,
    *,
    scope: str,
    site_id: int,
    layer: str,
    segment: str,
    min_segments: int,
    cap_segments: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group = group.sort_values(["event_id", "segment_t_h"]).copy()
    if cap_segments:
        group = cap_registration_segments(group)
    if group["event_id"].nunique() < min_segments or len(group) < 40:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    aligned, edges, components = hydrologically_constrained_registration(group, CSR_REGISTRATION_CFG)
    if aligned.empty or aligned["event_id"].nunique() < min_segments:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    curve, _ = fit_segment_curve(aligned)
    if curve.empty or len(curve) < 6:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for frame in (curve, edges, components):
        if frame.empty:
            continue
        frame["curve_scope"] = scope
        frame["site_id"] = int(site_id)
        frame["layer"] = str(layer)
        frame["segment"] = str(segment)
        frame["segment_label"] = REGIME_LABELS[str(segment)]
        frame["registered_segments"] = int(aligned["event_id"].nunique())
        frame["registration_components"] = int(components["registration_component"].nunique()) if not components.empty else 0
        frame["pairwise_edges"] = int(len(edges))
        frame["registration_method"] = "hydrologically_constrained_pairwise"
    return curve, edges, components


def build_registered_csr_library(points: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_events = split[split["split"].astype(str).eq("train")][["site_id", "layer", "event_id"]].drop_duplicates()
    train_points = points.merge(train_events, on=["site_id", "layer", "event_id"], how="inner")
    train_points = train_points[train_points["segment_regime"].isin(REGIME_ORDER)].copy()
    if train_points.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_points["parent_event_id"] = train_points["event_id"].astype(str)
    train_points["event_id_original"] = train_points["event_id"].astype(str)
    train_points["event_id"] = train_points["adaptive_segment_id"].astype(str)
    train_points["segment"] = train_points["segment_regime"].astype(str)

    curves_out: list[pd.DataFrame] = []
    edges_out: list[pd.DataFrame] = []
    components_out: list[pd.DataFrame] = []
    for (site_id, layer, regime), group in train_points.groupby(["site_id", "layer", "segment"], observed=True, sort=True):
        curve, edges, components = fit_registered_curve_group(
            group,
            scope="site-layer-regime",
            site_id=int(site_id),
            layer=str(layer),
            segment=str(regime),
            min_segments=MIN_TRAIN_EVENTS,
            cap_segments=False,
        )
        if curve.empty:
            continue
        curves_out.append(curve)
        if not edges.empty:
            edges_out.append(edges)
        if not components.empty:
            components_out.append(components)

    for (layer, regime), group in train_points.groupby(["layer", "segment"], observed=True, sort=True):
        curve, edges, components = fit_registered_curve_group(
            group,
            scope="layer-regime",
            site_id=-1,
            layer=str(layer),
            segment=str(regime),
            min_segments=MIN_TRAIN_EVENTS,
            cap_segments=True,
        )
        if curve.empty:
            continue
        curves_out.append(curve)
        if not edges.empty:
            edges_out.append(edges)
        if not components.empty:
            components_out.append(components)

    for regime, group in train_points.groupby("segment", observed=True, sort=True):
        curve, edges, components = fit_registered_curve_group(
            group,
            scope="network-regime",
            site_id=-1,
            layer="network",
            segment=str(regime),
            min_segments=MIN_TRAIN_EVENTS,
            cap_segments=True,
        )
        if curve.empty:
            continue
        curves_out.append(curve)
        if not edges.empty:
            edges_out.append(edges)
        if not components.empty:
            components_out.append(components)

    curves = pd.concat(curves_out, ignore_index=True) if curves_out else pd.DataFrame()
    edges = pd.concat(edges_out, ignore_index=True) if edges_out else pd.DataFrame()
    components = pd.concat(components_out, ignore_index=True) if components_out else pd.DataFrame()
    return curves, edges, components


def _curve_predict_future(curve: pd.DataFrame, s0: float, horizon_h: float) -> float:
    curve = curve.sort_values("csr_x_h").dropna(subset=["csr_x_h", "csr_mm"]).copy()
    if len(curve) < 4 or not np.isfinite(s0):
        return np.nan
    x = curve["csr_x_h"].to_numpy(float)
    s = curve["csr_mm"].to_numpy(float)
    order = np.argsort(x)
    x = x[order]
    s = s[order]
    _, unique_idx = np.unique(x, return_index=True)
    x = x[unique_idx]
    s = s[unique_idx]
    if len(x) < 4 or np.nanmax(x) <= np.nanmin(x):
        return np.nan
    s_min = float(np.nanmin(s))
    s_max = float(np.nanmax(s))
    if s0 > s_max + CSR_STORAGE_TOLERANCE_MM or s0 < s_min - CSR_STORAGE_TOLERANCE_MM:
        return np.nan
    storage_order = np.argsort(s)
    storage = s[storage_order]
    coord = x[storage_order]
    storage_unique, unique_storage_idx = np.unique(storage, return_index=True)
    coord_unique = coord[unique_storage_idx]
    if len(storage_unique) < 4:
        return np.nan
    s0_clamped = float(np.clip(s0, storage_unique.min(), storage_unique.max()))
    x0 = float(np.interp(s0_clamped, storage_unique, coord_unique))
    target_x = x0 + float(horizon_h)
    if target_x <= float(x.max()):
        return float(np.interp(target_x, x, s))
    if target_x > float(x.max()) + CSR_MAX_EXTRAPOLATION_H:
        return np.nan
    tail_n = min(8, len(x))
    tail_x = x[-tail_n:]
    tail_s = s[-tail_n:]
    if np.nanmax(tail_x) > np.nanmin(tail_x):
        slope = float(np.polyfit(tail_x, tail_s, 1)[0])
        terminal_loss = max(0.0, -slope)
    else:
        terminal_loss = 0.0
    return float(max(0.0, s[-1] - terminal_loss * (target_x - x[-1])))


def _prepare_curve_arrays(curve: pd.DataFrame) -> dict[str, object] | None:
    curve = curve.sort_values("csr_x_h").dropna(subset=["csr_x_h", "csr_mm"]).copy()
    if len(curve) < 4:
        return None
    x = curve["csr_x_h"].to_numpy(float)
    s = curve["csr_mm"].to_numpy(float)
    order = np.argsort(x)
    x = x[order]
    s = s[order]
    _, unique_idx = np.unique(x, return_index=True)
    x = x[unique_idx]
    s = s[unique_idx]
    if len(x) < 4 or np.nanmax(x) <= np.nanmin(x):
        return None
    storage_order = np.argsort(s)
    storage = s[storage_order]
    coord = x[storage_order]
    storage_unique, unique_storage_idx = np.unique(storage, return_index=True)
    coord_unique = coord[unique_storage_idx]
    if len(storage_unique) < 4:
        return None
    tail_n = min(8, len(x))
    tail_x = x[-tail_n:]
    tail_s = s[-tail_n:]
    if np.nanmax(tail_x) > np.nanmin(tail_x):
        slope = float(np.polyfit(tail_x, tail_s, 1)[0])
        terminal_loss = max(0.0, -slope)
    else:
        terminal_loss = 0.0
    return {
        "x": x,
        "s": s,
        "storage": storage_unique,
        "coord": coord_unique,
        "storage_min": float(storage_unique.min()),
        "storage_max": float(storage_unique.max()),
        "x_max": float(x.max()),
        "s_end": float(s[-1]),
        "terminal_loss": terminal_loss,
    }


def _curve_predict_future_many(curve: pd.DataFrame, s0: np.ndarray, horizon_h: np.ndarray) -> np.ndarray:
    prepared = _prepare_curve_arrays(curve)
    out = np.full(len(s0), np.nan, dtype=float)
    if prepared is None or len(s0) == 0:
        return out
    s0 = np.asarray(s0, dtype=float)
    horizon_h = np.asarray(horizon_h, dtype=float)
    ok = np.isfinite(s0) & np.isfinite(horizon_h)
    ok &= s0 <= prepared["storage_max"] + CSR_STORAGE_TOLERANCE_MM
    ok &= s0 >= prepared["storage_min"] - CSR_STORAGE_TOLERANCE_MM
    if not ok.any():
        return out
    s0_clamped = np.clip(s0[ok], prepared["storage_min"], prepared["storage_max"])
    x0 = np.interp(s0_clamped, prepared["storage"], prepared["coord"])
    target_x = x0 + horizon_h[ok]
    pred = np.full(len(target_x), np.nan, dtype=float)
    inside = target_x <= prepared["x_max"]
    if inside.any():
        pred[inside] = np.interp(target_x[inside], prepared["x"], prepared["s"])
    tail = (~inside) & (target_x <= prepared["x_max"] + CSR_MAX_EXTRAPOLATION_H)
    if tail.any():
        pred[tail] = np.maximum(0.0, prepared["s_end"] - prepared["terminal_loss"] * (target_x[tail] - prepared["x_max"]))
    out[np.where(ok)[0]] = pred
    return out


def _csr_curve_lookup(library: pd.DataFrame) -> dict[tuple[str, int, str, str], pd.DataFrame]:
    lookup: dict[tuple[str, int, str, str], pd.DataFrame] = {}
    if library.empty:
        return lookup
    work = library.copy()
    if "curve_scope" not in work.columns:
        work["curve_scope"] = "site-layer-regime"
    for (scope, site_id, layer, regime), curve in work.groupby(["curve_scope", "site_id", "layer", "segment"], observed=True, sort=False):
        lookup[(str(scope), int(site_id), str(layer), str(regime))] = curve.sort_values("csr_x_h").copy()
    return lookup


def registered_csr_predict(library: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    base = test[["forecast_id", "site_id", "layer", "horizon_h", "s0_mm"]].copy()
    lookup = _csr_curve_lookup(library)
    test_work = test.reset_index(drop=True).copy()
    predictions = np.full(len(test_work), np.nan, dtype=float)
    numerator = np.zeros(len(test_work), dtype=float)
    denominator = np.zeros(len(test_work), dtype=float)
    candidates = np.zeros(len(test_work), dtype=float)
    level_labels = np.array([""] * len(test_work), dtype=object)

    def apply_curve(indices: np.ndarray, curve: pd.DataFrame, regime: str, level_name: str) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=int)
        pred_values = _curve_predict_future_many(
            curve,
            test_work.loc[indices, "s0_mm"].to_numpy(float),
            test_work.loc[indices, "horizon_h"].to_numpy(float),
        )
        valid_local = np.isfinite(pred_values)
        valid_idx = indices[valid_local]
        if len(valid_idx) == 0:
            return valid_idx
        prob = test_work.loc[valid_idx, f"prob_{regime}"].to_numpy(float)
        numerator[valid_idx] += prob * pred_values[valid_local]
        denominator[valid_idx] += prob
        candidates[valid_idx] += prob * float(curve["registered_segments"].iloc[0])
        suffix = f"{regime}:{level_name};"
        for idx in valid_idx:
            level_labels[idx] += suffix
        return valid_idx

    for regime in REGIME_ORDER:
        prob = test_work[f"prob_{regime}"].fillna(0.0).to_numpy(float)
        remaining = np.where(prob > 0)[0]
        if len(remaining) == 0:
            continue

        rem_frame = test_work.loc[remaining, ["site_id", "layer"]]
        for (site_id, layer), idx_values in rem_frame.groupby(["site_id", "layer"], observed=True).groups.items():
            curve = lookup.get(("site-layer-regime", int(site_id), str(layer), regime))
            if curve is None:
                continue
            valid_idx = apply_curve(np.asarray(list(idx_values), dtype=int), curve, regime, "site-layer-regime")
            if len(valid_idx):
                remaining = np.setdiff1d(remaining, valid_idx, assume_unique=False)
        if len(remaining) == 0:
            continue

        rem_frame = test_work.loc[remaining, ["layer"]]
        for layer, idx_values in rem_frame.groupby("layer", observed=True).groups.items():
            curve = lookup.get(("layer-regime", -1, str(layer), regime))
            if curve is None:
                continue
            valid_idx = apply_curve(np.asarray(list(idx_values), dtype=int), curve, regime, "layer-regime")
            if len(valid_idx):
                remaining = np.setdiff1d(remaining, valid_idx, assume_unique=False)
        if len(remaining) == 0:
            continue

        curve = lookup.get(("network-regime", -1, "network", regime))
        if curve is not None:
            apply_curve(remaining, curve, regime, "network-regime")

    ok = denominator > 0
    predictions[ok] = numerator[ok] / denominator[ok]
    candidates[ok] = candidates[ok] / denominator[ok]
    level_labels[ok] = np.char.add("registered CSR:", level_labels[ok].astype(str))
    out = base[["forecast_id"]].copy()
    out["model"] = "registered_csr_operator"
    out["pred_mm"] = predictions
    out["q10_mm"] = np.nan
    out["q90_mm"] = np.nan
    out["analog_level"] = level_labels
    out["analog_candidates"] = np.round(candidates).astype(int)
    return out[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]]


def build_predictions(train_rows: pd.DataFrame, test_rows: pd.DataFrame, regime_predictions: pd.DataFrame, csr_library: pd.DataFrame | None = None) -> pd.DataFrame:
    test = test_rows.merge(regime_predictions.drop(columns=["true_regime"]), on="origin_id", how="left", validate="many_to_one")
    test = test.reset_index(drop=True)
    test["forecast_id"] = np.arange(len(test), dtype=int)
    train = train_rows.reset_index(drop=True)
    train["true_regime"] = train["true_regime"].astype(str)
    test["true_regime"] = test["true_regime"].astype(str)

    base_cols = [
        "forecast_id",
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
        "delta_mm",
        "start_mm",
        "drop_sofar_mm",
        "recent_loss_1h",
        "true_regime",
        "predicted_regime",
        "adaptive_segment_order",
    ]
    base = test[base_cols].copy()
    model_frames = []

    persistence = base[["forecast_id", "s0_mm"]].copy()
    persistence["model"] = "persistence"
    persistence["pred_mm"] = persistence["s0_mm"]
    persistence["q10_mm"] = np.nan
    persistence["q90_mm"] = np.nan
    persistence["analog_level"] = "none"
    persistence["analog_candidates"] = 0
    model_frames.append(persistence[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]])

    recent = base[["forecast_id", "s0_mm", "recent_loss_1h", "horizon_h"]].copy()
    recent["model"] = "recent_slope"
    recent["pred_mm"] = np.maximum(0.0, recent["s0_mm"] - recent["recent_loss_1h"] * recent["horizon_h"])
    recent["q10_mm"] = np.nan
    recent["q90_mm"] = np.nan
    recent["analog_level"] = "none"
    recent["analog_candidates"] = 0
    model_frames.append(recent[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]])

    if csr_library is not None and not csr_library.empty:
        model_frames.append(registered_csr_predict(csr_library, test))
    model_frames.append(analog_predict(train, test, None, "nonregime_analog"))
    model_frames.append(analog_predict(train, test, "predicted_regime", "online_regime_analog"))
    model_frames.append(regime_mixture_predict(train, test))
    model_frames.append(analog_predict(train, test, "true_regime", "oracle_regime_analog"))

    pred = pd.concat(model_frames, ignore_index=True)
    pred = pred.merge(base, on="forecast_id", how="left", validate="many_to_one")
    pred["layer"] = pd.Categorical(pred["layer"], categories=LAYER_ORDER, ordered=True)
    pred["model"] = pd.Categorical(pred["model"], categories=MODEL_ORDER, ordered=True)
    pred["residual_mm"] = pred["pred_mm"] - pred["target_mm"]
    return pred.sort_values(["model", "site_id", "layer", "event_id", "origin_t_h", "horizon_h"]).reset_index(drop=True)


def r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 3:
        return np.nan
    sse = float(np.sum((pred - obs) ** 2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


def metrics_for_group(frame: pd.DataFrame) -> dict[str, object]:
    obs = frame["target_mm"].to_numpy(float)
    pred = frame["pred_mm"].to_numpy(float)
    residual = pred - obs
    return {
        "forecasts": int(np.isfinite(residual).sum()),
        "sites": int(frame["site_id"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "origins": int(frame["origin_id"].nunique()),
        "ccc": concordance_correlation_coefficient(obs, pred),
        "r2": r2_score(obs, pred),
        "rmse_mm": float(np.sqrt(np.nanmean(residual**2))),
        "mae_mm": float(np.nanmean(np.abs(residual))),
        "bias_mm": float(np.nanmean(residual)),
    }


def summarize_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = pred.dropna(subset=["pred_mm", "target_mm"]).copy()
    layer_rows = []
    for (model, layer, horizon), group in valid.groupby(["model", "layer", "horizon_h"], observed=True):
        row = {"model": str(model), "layer": str(layer), "layer_label": layer_label(str(layer)), "horizon_h": int(horizon)}
        row.update(metrics_for_group(group))
        layer_rows.append(row)
    overall_rows = []
    for (model, horizon), group in valid.groupby(["model", "horizon_h"], observed=True):
        row = {"model": str(model), "horizon_h": int(horizon)}
        row.update(metrics_for_group(group))
        overall_rows.append(row)
    return pd.DataFrame(layer_rows), pd.DataFrame(overall_rows)


def summarize_site_layer_forecast(pred: pd.DataFrame) -> pd.DataFrame:
    valid = pred[
        pred["model"].astype(str).eq("registered_csr_operator")
        & pred["pred_mm"].notna()
        & pred["target_mm"].notna()
    ].copy()
    rows = []
    for (site_id, layer), group in valid.groupby(["site_id", "layer"], observed=True):
        row = {"site_id": int(site_id), "layer": str(layer), "layer_label": layer_label(str(layer))}
        row.update(metrics_for_group(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["site_id", "layer"])


def heatmap_matrix(site_layer_metrics: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[int]]:
    sites = sorted(site_layer_metrics["site_id"].dropna().astype(int).unique().tolist())
    matrix = np.full((len(sites), len(LAYER_ORDER)), np.nan)
    for i, site_id in enumerate(sites):
        for j, layer in enumerate(LAYER_ORDER):
            cell = site_layer_metrics[(site_layer_metrics["site_id"].eq(site_id)) & (site_layer_metrics["layer"].eq(layer))]
            if not cell.empty:
                matrix[i, j] = float(cell[value_col].iloc[0])
    return matrix, sites


def plot_forecast_figure(pred: pd.DataFrame, site_layer_metrics: pd.DataFrame) -> None:
    model_pred = pred[
        pred["model"].astype(str).eq("registered_csr_operator")
        & pred["pred_mm"].notna()
        & pred["target_mm"].notna()
    ].copy()
    layer_metrics = []
    for layer in LAYER_ORDER:
        layer_data = model_pred[model_pred["layer"].astype(str).eq(layer)]
        if layer_data.empty:
            continue
        row = {"layer": layer, "layer_label": layer_label(layer)}
        row.update(metrics_for_group(layer_data))
        layer_metrics.append(row)
    layer_metrics_df = pd.DataFrame(layer_metrics)

    fig = plt.figure(figsize=(7.35, 8.1))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.95, 1.55], hspace=0.46, wspace=0.45)
    scatter_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax_rmse = fig.add_subplot(gs[1, :3])
    ax_ccc = fig.add_subplot(gs[1, 3:])

    point_color = "#3B78A6"
    line_color = "#9B2D3A"
    for i, (ax, layer) in enumerate(zip(scatter_axes, LAYER_ORDER)):
        data = model_pred[model_pred["layer"].astype(str).eq(layer)].copy()
        if len(data) > 1600:
            data = data.sample(1600, random_state=RANDOM_SEED + i)
        if data.empty:
            ax.text(0.5, 0.5, "No held-out\nforecasts", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(("a  " if i == 0 else "") + layer_label(layer), loc="left", fontweight="bold")
            ax.set_axis_off()
            continue
        ax.scatter(data["target_mm"], data["pred_mm"], s=6, alpha=0.24, color=point_color, edgecolors="none", rasterized=True)
        lo = float(np.nanmin([data["target_mm"].min(), data["pred_mm"].min()]))
        hi = float(np.nanmax([data["target_mm"].max(), data["pred_mm"].max()]))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color=line_color, lw=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        metric = layer_metrics_df[layer_metrics_df["layer"].eq(layer)]
        if not metric.empty:
            ax.text(
                0.05,
                0.95,
                f"RMSE {metric['rmse_mm'].iloc[0]:.2f}\nCCC {metric['ccc'].iloc[0]:.3f}\nn={int(metric['forecasts'].iloc[0])}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
            )
        ax.set_title(("a  " if i == 0 else "") + layer_label(layer), loc="left", fontweight="bold")
        ax.grid(True, color="#E8EDF1", lw=0.5)
        if i == 0:
            ax.set_ylabel("Predicted (mm)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Observed (mm)")

    rmse_matrix, sites = heatmap_matrix(site_layer_metrics, "rmse_mm")
    ccc_matrix, _ = heatmap_matrix(site_layer_metrics, "ccc")
    layer_ticks = [layer_label(layer) for layer in LAYER_ORDER]
    cmap_rmse = mpl.colormaps["YlOrRd"].copy()
    cmap_rmse.set_bad("#F0F0F0")
    cmap_ccc = mpl.colormaps["YlGnBu"].copy()
    cmap_ccc.set_bad("#F0F0F0")
    rmse_vmax = float(np.nanpercentile(rmse_matrix, 92)) if np.isfinite(rmse_matrix).any() else 1.0
    rmse_vmax = max(rmse_vmax, 0.5)

    im_rmse = ax_rmse.imshow(np.ma.masked_invalid(rmse_matrix), aspect="auto", cmap=cmap_rmse, vmin=0, vmax=rmse_vmax)
    ax_rmse.set_title("b  Station-layer registered-CSR forecast error", loc="left", fontweight="bold")
    ax_rmse.set_xticks(np.arange(len(LAYER_ORDER)))
    ax_rmse.set_xticklabels(layer_ticks)
    ax_rmse.set_yticks(np.arange(len(sites)))
    ax_rmse.set_yticklabels([str(site) for site in sites], fontsize=5.2)
    ax_rmse.set_ylabel("FAWN station")
    ax_rmse.set_xlabel("Sensor depth")
    ax_rmse.set_xticks(np.arange(-0.5, len(LAYER_ORDER), 1), minor=True)
    ax_rmse.set_yticks(np.arange(-0.5, len(sites), 1), minor=True)
    ax_rmse.grid(which="minor", color="white", linewidth=0.55)
    ax_rmse.tick_params(which="minor", bottom=False, left=False)
    cbar_rmse = fig.colorbar(im_rmse, ax=ax_rmse, fraction=0.035, pad=0.025)
    cbar_rmse.ax.set_title("RMSE\n(mm)", fontsize=6, pad=4)

    im_ccc = ax_ccc.imshow(np.ma.masked_invalid(ccc_matrix), aspect="auto", cmap=cmap_ccc, vmin=0.70, vmax=1.0)
    ax_ccc.set_title("Station-layer concordance", loc="left", fontweight="bold")
    ax_ccc.set_xticks(np.arange(len(LAYER_ORDER)))
    ax_ccc.set_xticklabels(layer_ticks)
    ax_ccc.set_yticks(np.arange(len(sites)))
    ax_ccc.set_yticklabels([])
    ax_ccc.set_xlabel("Sensor depth")
    ax_ccc.set_xticks(np.arange(-0.5, len(LAYER_ORDER), 1), minor=True)
    ax_ccc.set_yticks(np.arange(-0.5, len(sites), 1), minor=True)
    ax_ccc.grid(which="minor", color="white", linewidth=0.55)
    ax_ccc.tick_params(which="minor", bottom=False, left=False)
    cbar_ccc = fig.colorbar(im_ccc, ax=ax_ccc, fraction=0.052, pad=0.03)
    cbar_ccc.ax.set_title("CCC", fontsize=6, pad=4)

    fig.suptitle("Hydrologically registered CSR functions forecast held-out soil moisture drydowns across depths and stations", x=0.02, y=0.985, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.02, 0.955, "Panel a pools held-out predictions from the registered CSR operator within each sensor depth; panel b summarizes station-layer error across 1-24 h horizons; gray cells indicate unavailable held-out validation.", fontsize=7, color="#555555")
    save_pub(fig, "fig_regime_conditioned_forecast_validation")
    plt.close(fig)


def write_report(
    origins: pd.DataFrame,
    split_summary: pd.DataFrame,
    classifier_metrics: pd.DataFrame,
    metrics_layer: pd.DataFrame,
    metrics_overall: pd.DataFrame,
    site_layer_metrics: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df.empty:
            return "_No rows._"
        return df.head(max_rows).to_markdown(index=False)

    overall_short = metrics_overall[metrics_overall["model"].isin(MODEL_ORDER)].copy()
    layer_short = metrics_layer[
        metrics_layer["model"].isin(["registered_csr_operator", "nonregime_analog", "online_regime_analog", "regime_mixture_analog", "oracle_regime_analog"])
    ].copy()
    station_layer_short = site_layer_metrics[
        ["site_id", "layer_label", "forecasts", "events", "origins", "ccc", "r2", "rmse_mm", "mae_mm", "bias_mm"]
    ].copy()
    clf_summary = classifier_metrics[classifier_metrics["record"].eq("summary")][
        ["test_origins", "accuracy", "balanced_accuracy"]
    ].copy()
    clf_confusion = classifier_metrics[classifier_metrics["record"].eq("confusion")][
        ["true_regime", "predicted_regime", "origins", "row_share"]
    ].copy()
    report = f"""# Experiment 4: Hydrologically registered CSR drydown forecasting

## Material Passport

- Type: code experiment result
- Status: completed
- Data unit: clean rainfall-associated SMDE, adaptive within-event segment, forecast origin
- Split: chronological event-level 80/20 within each FAWN location-layer group
- Forecast horizons: {", ".join(str(h) + " h" for h in HORIZONS_H)}
- Operational target: forecast future soil water amount from current soil water amount and recent drydown behavior.

## Experiment Design

The regime-specific CSR curves were not stitched into one static drydown curve. Instead, each location-layer-regime library was built with hydrologically constrained registration. Segment pairs were connected only when they shared sufficient soil-water storage overlap and similar loss rates. For a test forecast origin, the operational model first estimates the probability of each current regime from information available at the origin: current soil water amount, 1 h recent loss rate, cumulative drop since drydown start, elapsed drydown time, start storage, and layer. It then locates the current soil-water amount on each registered regime-specific CSR curve, advances along that empirical response coordinate by the requested horizon, and combines the regime-specific forecasts by online regime probabilities.

The forecast operator can be written as:

```text
S_hat(t + h) = sum_r p(r | X_t) C_{{r,l,s}}( C_{{r,l,s}}^-1(S_t) + h ),
```

where C is the hydrologically registered CSR function for regime r, layer l, and station s. Analog models are retained as sensitivity checks, but the registered CSR operator is the main prospective forecast model.

## Forecast-Origin Sample

| split | forecast rows | forecast origins | events | sites |
|:--|--:|--:|--:|--:|
| train | {len(origins[origins["split"].eq("train")]):,} | {origins[origins["split"].eq("train")]["origin_id"].nunique():,} | {origins[origins["split"].eq("train")]["event_id"].nunique():,} | {origins[origins["split"].eq("train")]["site_id"].nunique():,} |
| test | {len(origins[origins["split"].eq("test")]):,} | {origins[origins["split"].eq("test")]["origin_id"].nunique():,} | {origins[origins["split"].eq("test")]["event_id"].nunique():,} | {origins[origins["split"].eq("test")]["site_id"].nunique():,} |

## Online Regime Classifier

{md_table(clf_summary)}

Classifier confusion matrix, row-normalized:

{md_table(clf_confusion, max_rows=12)}

## Overall Forecast Skill by Horizon

{md_table(overall_short.sort_values(["horizon_h", "model"]))}

## Regime-Conditioned Forecast Skill Across Sensor Depths

{md_table(layer_short.sort_values(["layer", "horizon_h", "model"]), max_rows=80)}

## Station-Layer Validation Summary

{md_table(station_layer_short.sort_values(["layer_label", "site_id"]), max_rows=80)}

## Interpretation

This experiment changes the role of regime-specific CSR. The curves are no longer interpreted as pieces of a single universal drydown curve; they become conditional forecast operators. The hydrologically constrained registration resolves the jump problem in the stitched curve, because no artificial continuity is forced without storage-overlap and loss-rate evidence. The output is a future soil-moisture trajectory from the current state; users can later couple this forecast with crop- or site-specific irrigation rules, but those decision thresholds are not part of this experiment.

The diagnosed-regime analog represents an upper bound because it assumes the current regime is known perfectly. The analog models are useful as sensitivity checks, but the registered CSR operator is the preferred operational form because it keeps the empirical drydown response in CSR space while allowing regime uncertainty to enter smoothly.

## Output Files

- Figure: `{OUT / "fig_regime_conditioned_forecast_validation.pdf"}`
- Predictions: `{SOURCE / "regime_conditioned_forecast_predictions.parquet"}`
- Registered CSR library: `{SOURCE / "registered_csr_forecast_library.csv"}`
- Registered CSR registration edges: `{SOURCE / "registered_csr_forecast_registration_edges.csv"}`
- Metrics by layer and horizon: `{SOURCE / "forecast_metrics_by_layer_horizon.csv"}`
- Overall metrics by horizon: `{SOURCE / "forecast_metrics_overall_horizon.csv"}`
- Station-layer metrics: `{SOURCE / "forecast_metrics_by_station_layer.csv"}`
- Regime classifier metrics: `{SOURCE / "online_regime_classifier_metrics.csv"}`
"""
    (OUT / "experiment4_regime_conditioned_forecast_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_out()
    events, points = load_inputs()
    split, split_summary = chronological_event_split(points)
    csr_library, csr_edges, csr_components = build_registered_csr_library(points, split)
    origins = build_forecast_origin_table(points, split)
    train_rows = origins[origins["split"].eq("train")].copy()
    test_rows = origins[origins["split"].eq("test")].copy()
    regime_predictions, classifier_metrics = classify_online_regime(train_rows, test_rows)
    predictions = build_predictions(train_rows, test_rows, regime_predictions, csr_library)
    metrics_layer, metrics_overall = summarize_metrics(predictions)
    site_layer_metrics = summarize_site_layer_forecast(predictions)

    split_summary.to_csv(SOURCE / "forecast_event_split_summary.csv", index=False)
    csr_library.to_csv(SOURCE / "registered_csr_forecast_library.csv", index=False)
    csr_edges.to_csv(SOURCE / "registered_csr_forecast_registration_edges.csv", index=False)
    csr_components.to_csv(SOURCE / "registered_csr_forecast_registration_components.csv", index=False)
    origins.to_parquet(SOURCE / "forecast_origin_rows.parquet", index=False)
    regime_predictions.to_csv(SOURCE / "online_regime_predictions_by_origin.csv", index=False)
    classifier_metrics.to_csv(SOURCE / "online_regime_classifier_metrics.csv", index=False)
    predictions.to_parquet(SOURCE / "regime_conditioned_forecast_predictions.parquet", index=False)
    metrics_layer.to_csv(SOURCE / "forecast_metrics_by_layer_horizon.csv", index=False)
    metrics_overall.to_csv(SOURCE / "forecast_metrics_overall_horizon.csv", index=False)
    site_layer_metrics.to_csv(SOURCE / "forecast_metrics_by_station_layer.csv", index=False)
    plot_forecast_figure(predictions, site_layer_metrics)
    write_report(origins, split_summary, classifier_metrics, metrics_layer, metrics_overall, site_layer_metrics)

    print(f"Wrote Experiment 4 outputs to {OUT}")
    print(metrics_overall[metrics_overall["model"].eq("registered_csr_operator")].sort_values("horizon_h").to_string(index=False))


if __name__ == "__main__":
    main()
