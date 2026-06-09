from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from build_experiment4_regime_conditioned_forecast import (
    FEATURE_COLS,
    HORIZONS_H,
    LAYER_ORDER,
    RANDOM_SEED,
    REGIME_COLORS,
    REGIME_LABELS,
    REGIME_ORDER,
    analog_predict,
    metrics_for_group,
    r2_score,
    regime_mixture_predict,
    registered_csr_predict,
)
from fawn_full_smde_audit import (
    MOISTURE_LAYERS,
    MOISTURE_RENAME,
    MOISTURE_SOURCE_COLS,
    PCT_TO_MM_4IN,
    WX_CONTEXT_COLS,
    WX_VALUE_COLS,
    make_config,
    merge_station,
    normalize_moisture_pct,
    read_station_parquet,
)
from smde_regime_audit import build_event_tables


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
DATA = ANALYSIS / "fawn_db_export" / "data"
SOURCE_EXP4 = ANALYSIS / "experiment4_regime_conditioned_forecast" / "source_data"
CSR_LIBRARY = SOURCE_EXP4 / "registered_csr_forecast_library.csv"
OUT = ANALYSIS / "experiment4_2026_external_forecast"
SOURCE = OUT / "source_data"

YEAR = 2026
MIN_FIG_EVENT_H = 8.0
MAIN_MODEL = "registered_csr_operator"

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


def layer_label(layer: str) -> str:
    return LAYER_LABELS.get(str(layer), str(layer))


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def require_2026_files() -> tuple[Path, Path]:
    soil_path = DATA / "soil_moisture_2026.parquet"
    wx_path = DATA / "wx_selected_2026.parquet"
    missing = [str(p) for p in [soil_path, wx_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing 2026 FAWN exports. Pull them first with fawn_db_pull.py. Missing: "
            + "; ".join(missing)
        )
    return soil_path, wx_path


def station_ids_from_2026(soil_path: Path) -> list[int]:
    ids = pd.read_parquet(soil_path, columns=["ID"])["ID"].dropna().astype(int).unique().tolist()
    return sorted(ids)


def prepare_soil_2026(site_id: int, soil_path: Path) -> pd.DataFrame:
    cols = ["ID", "UTC", *MOISTURE_SOURCE_COLS]
    soil = read_station_parquet([soil_path], site_id, cols)
    if soil.empty:
        return soil
    soil["UTC"] = pd.to_datetime(soil["UTC"], errors="coerce")
    soil = soil.dropna(subset=["UTC"]).sort_values("UTC")
    soil = soil.drop_duplicates(subset=["UTC"], keep="last")
    for col in MOISTURE_SOURCE_COLS:
        soil[col] = normalize_moisture_pct(soil[col]) * PCT_TO_MM_4IN
    return soil.rename(columns=MOISTURE_RENAME)


def prepare_wx_2026(site_id: int, wx_path: Path) -> pd.DataFrame:
    cols = ["ID", "UTC", *WX_VALUE_COLS]
    wx = read_station_parquet([wx_path], site_id, cols)
    if wx.empty:
        return wx
    wx["UTC"] = pd.to_datetime(wx["UTC"], errors="coerce")
    wx = wx.dropna(subset=["UTC"]).sort_values("UTC")
    for col in WX_VALUE_COLS:
        wx[col] = pd.to_numeric(wx[col], errors="coerce")
    primary_rain = wx["rain_2m_inches"]
    backup_rain = wx["rain_backup_2m_inches"]
    wx["Rain"] = primary_rain.where(primary_rain.notna(), backup_rain).fillna(0.0).clip(lower=0.0)
    wx["rain_backup_used"] = primary_rain.isna() & backup_rain.notna()
    agg = {"Rain": "sum", "rain_backup_used": "max"}
    for col in WX_CONTEXT_COLS:
        agg[col] = "mean"
    return wx.groupby("UTC", as_index=False).agg(agg).sort_values("UTC")


def event_raw_points(site_id: int, merged: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for row in events.itertuples(index=False):
        layer = str(row.layer)
        event_id = str(row.event_id)
        segment = merged.loc[pd.Timestamp(row.start) : pd.Timestamp(row.end), [layer]].dropna(subset=[layer])
        if len(segment) < 4:
            continue
        t_h = (segment.index - segment.index[0]).total_seconds().to_numpy() / 3600.0
        part = pd.DataFrame(
            {
                "site_id": site_id,
                "event_id": event_id,
                "layer": layer,
                "t_h": t_h,
                "moisture_mm": segment[layer].to_numpy(float),
                "start_mm": float(row.start_mm),
                "end_mm": float(row.end_mm),
                "total_drop_mm": float(row.total_drop_mm),
                "duration_h": float(row.duration_h),
                "associated_48h": bool(row.associated_48h),
                "interrupted_by_rain": bool(row.interrupted_by_rain),
                "clean_48h": bool(row.associated_48h) and not bool(row.interrupted_by_rain),
                "mean_loss_mm_h": float(row.mean_loss_mm_h),
                "rain_before_48": float(row.rain_before_48),
                "rain_during": float(row.rain_during),
                "start_month": int(pd.Timestamp(row.start).month),
                "start": pd.Timestamp(row.start),
                "end": pd.Timestamp(row.end),
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_2026_external_points() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    soil_path, wx_path = require_2026_files()
    cfg = make_config()
    all_events = []
    all_raw = []
    status_rows = []
    for site_id in station_ids_from_2026(soil_path):
        soil = prepare_soil_2026(site_id, soil_path)
        wx = prepare_wx_2026(site_id, wx_path)
        status = {"site_id": site_id, "soil_rows": len(soil), "wx_rows": len(wx), "events": 0, "clean_events": 0, "raw_points": 0}
        if soil.empty or wx.empty:
            status_rows.append(status)
            continue
        merged = merge_station(soil, wx)
        events, _ = build_event_tables(merged[[*MOISTURE_LAYERS, "Rain"]], cfg)
        if not events.empty:
            events = events.copy()
            events["event_id"] = events["event_id"].map(lambda x: f"S{site_id}_2026_{x}")
            events.insert(0, "site_id", site_id)
            events["clean_48h"] = events["associated_48h"].astype(bool) & ~events["interrupted_by_rain"].astype(bool)
            clean_events = events[events["clean_48h"]].copy()
            raw = event_raw_points(site_id, merged, clean_events)
            status["events"] = len(events)
            status["clean_events"] = len(clean_events)
            status["raw_points"] = len(raw)
            all_events.append(events)
            if not raw.empty:
                all_raw.append(raw)
        status_rows.append(status)
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    raw_df = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    status_df = pd.DataFrame(status_rows)
    return events_df, raw_df, status_df


def origin_rows_for_external_event(event: pd.DataFrame) -> list[dict[str, object]]:
    event = event.sort_values("t_h")
    t = event["t_h"].to_numpy(float)
    y = event["moisture_mm"].to_numpy(float)
    if len(t) < 6 or np.nanmax(t) < 1.0 + min(HORIZONS_H):
        return []
    start_mm = float(event["start_mm"].iloc[0])
    rows = []
    valid_origin_mask = np.isclose((t / 0.5) % 1, 0, atol=1e-6) | np.isclose((t / 0.5) % 1, 1, atol=1e-6)
    for idx, t0 in enumerate(t):
        if t0 < 1.0 or not valid_origin_mask[idx]:
            continue
        s0 = float(y[idx])
        prev = float(np.interp(t0 - 1.0, t, y))
        recent_loss = max(0.0, (prev - s0) / 1.0)
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
                    "split": "external_2026",
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
                    "rain_before_48": float(event["rain_before_48"].iloc[0]) if pd.notna(event["rain_before_48"].iloc[0]) else np.nan,
                    "rain_during": float(event["rain_during"].iloc[0]) if pd.notna(event["rain_during"].iloc[0]) else np.nan,
                    "start_month": int(event["start_month"].iloc[0]) if pd.notna(event["start_month"].iloc[0]) else -1,
                }
            )
    return rows


def build_external_forecast_origins(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, event in points.groupby(["site_id", "layer", "event_id"], observed=True, sort=False):
        rows.extend(origin_rows_for_external_event(event))
    origins = pd.DataFrame(rows)
    if origins.empty:
        return origins
    origins["layer"] = pd.Categorical(origins["layer"], categories=LAYER_ORDER, ordered=True)
    origins = origins.sort_values(["site_id", "layer", "event_id", "origin_t_h", "horizon_h"]).reset_index(drop=True)
    return origins


def predict_external_regime_probabilities(train_rows: pd.DataFrame, external_rows: pd.DataFrame) -> pd.DataFrame:
    train_origins = train_rows.drop_duplicates("origin_id").copy()
    external_origins = external_rows.drop_duplicates("origin_id").copy()
    feature_frame = pd.concat(
        [
            train_origins[["origin_id", "true_regime", "layer", *FEATURE_COLS]].assign(part="train"),
            external_origins[["origin_id", "layer", *FEATURE_COLS]].assign(true_regime=np.nan, part="external"),
        ],
        ignore_index=True,
    )
    x_all = pd.get_dummies(feature_frame[["layer", *FEATURE_COLS]], columns=["layer"], dtype=float)
    train_mask = feature_frame["part"].eq("train").to_numpy()
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(x_all.loc[train_mask], feature_frame.loc[train_mask, "true_regime"].astype(str))
    pred = clf.predict(x_all.loc[~train_mask])
    prob = clf.predict_proba(x_all.loc[~train_mask])
    table = feature_frame.loc[~train_mask, ["origin_id"]].copy().reset_index(drop=True)
    table["predicted_regime"] = pred
    prob_df = pd.DataFrame(prob, columns=[f"prob_{cls}" for cls in clf.classes_])
    table = pd.concat([table, prob_df], axis=1)
    for regime in REGIME_ORDER:
        col = f"prob_{regime}"
        if col not in table.columns:
            table[col] = 0.0
    return table


def build_external_predictions(train_rows: pd.DataFrame, external_rows: pd.DataFrame, regime_predictions: pd.DataFrame, csr_library: pd.DataFrame) -> pd.DataFrame:
    test = external_rows.merge(regime_predictions, on="origin_id", how="left", validate="many_to_one").reset_index(drop=True)
    test["forecast_id"] = np.arange(len(test), dtype=int)
    train = train_rows.reset_index(drop=True).copy()
    train["true_regime"] = train["true_regime"].astype(str)
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
        "predicted_regime",
    ]
    base = test[base_cols].copy()
    frames = []
    persistence = base[["forecast_id", "s0_mm"]].copy()
    persistence["model"] = "persistence"
    persistence["pred_mm"] = persistence["s0_mm"]
    persistence["q10_mm"] = np.nan
    persistence["q90_mm"] = np.nan
    persistence["analog_level"] = "none"
    persistence["analog_candidates"] = 0
    frames.append(persistence[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]])

    recent = base[["forecast_id", "s0_mm", "recent_loss_1h", "horizon_h"]].copy()
    recent["model"] = "recent_slope"
    recent["pred_mm"] = np.maximum(0.0, recent["s0_mm"] - recent["recent_loss_1h"] * recent["horizon_h"])
    recent["q10_mm"] = np.nan
    recent["q90_mm"] = np.nan
    recent["analog_level"] = "none"
    recent["analog_candidates"] = 0
    frames.append(recent[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]])

    if not csr_library.empty:
        frames.append(registered_csr_predict(csr_library, test))
    frames.append(analog_predict(train, test, None, "nonregime_analog"))
    frames.append(analog_predict(train, test, "predicted_regime", "online_regime_analog"))
    frames.append(regime_mixture_predict(train, test))
    pred = pd.concat(frames, ignore_index=True).merge(base, on="forecast_id", how="left", validate="many_to_one")
    pred["layer"] = pd.Categorical(pred["layer"], categories=LAYER_ORDER, ordered=True)
    pred["residual_mm"] = pred["pred_mm"] - pred["target_mm"]
    return pred.sort_values(["model", "site_id", "layer", "event_id", "origin_t_h", "horizon_h"]).reset_index(drop=True)


def summarize_site_layer(pred: pd.DataFrame) -> pd.DataFrame:
    valid = pred[pred["model"].astype(str).eq(MAIN_MODEL)].dropna(subset=["pred_mm", "target_mm"])
    rows = []
    for (site_id, layer), group in valid.groupby(["site_id", "layer"], observed=True):
        obs = group["target_mm"].to_numpy(float)
        p = group["pred_mm"].to_numpy(float)
        residual = p - obs
        rows.append(
            {
                "site_id": int(site_id),
                "layer": str(layer),
                "layer_label": layer_label(str(layer)),
                "forecasts": int(len(group)),
                "events": int(group["event_id"].nunique()),
                "origins": int(group["origin_id"].nunique()),
                "ccc": metrics_for_group(group)["ccc"],
                "r2": r2_score(obs, p),
                "rmse_mm": float(np.sqrt(np.nanmean(residual**2))),
                "mae_mm": float(np.nanmean(np.abs(residual))),
                "bias_mm": float(np.nanmean(residual)),
            }
        )
    return pd.DataFrame(rows).sort_values(["site_id", "layer"])


def choose_showcase_event(points: pd.DataFrame, pred: pd.DataFrame) -> tuple[int, str, str]:
    valid = pred[pred["model"].astype(str).eq(MAIN_MODEL)].dropna(subset=["pred_mm"])
    event_info = (
        points.groupby(["site_id", "layer", "event_id"], observed=True)
        .agg(
            duration_h=("duration_h", "first"),
            points=("moisture_mm", "size"),
        )
        .reset_index()
    )
    metric = (
        valid.groupby(["site_id", "layer", "event_id"], observed=True)
        .agg(
            forecasts=("pred_mm", "size"),
            horizons=("horizon_h", "nunique"),
            predicted_regimes=("predicted_regime", "nunique"),
            rmse=("residual_mm", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
        )
        .reset_index()
    )
    candidates = event_info.merge(metric, on=["site_id", "layer", "event_id"], how="inner")
    candidates = candidates[(candidates["duration_h"] >= MIN_FIG_EVENT_H) & (candidates["forecasts"] >= 8)]
    if candidates.empty:
        row = valid.iloc[0]
        return int(row.site_id), str(row.layer), str(row.event_id)
    candidates["layer_rank"] = candidates["layer"].map({layer: i for i, layer in enumerate(LAYER_ORDER)}).fillna(9)
    median_rmse = candidates["rmse"].median()
    candidates["score"] = (
        0.7 * candidates["layer_rank"]
        - 1.0 * candidates["predicted_regimes"]
        - 0.3 * candidates["horizons"]
        + np.abs(candidates["rmse"] - median_rmse)
    )
    row = candidates.sort_values("score").iloc[0]
    return int(row.site_id), str(row.layer), str(row.event_id)


def choose_layer_showcase_events(points: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    valid = pred[pred["model"].astype(str).eq(MAIN_MODEL)].dropna(subset=["pred_mm"])
    event_info = (
        points.groupby(["site_id", "layer", "event_id"], observed=True)
        .agg(duration_h=("duration_h", "first"), points=("moisture_mm", "size"))
        .reset_index()
    )
    metric = (
        valid.groupby(["site_id", "layer", "event_id"], observed=True)
        .agg(
            forecasts=("pred_mm", "size"),
            horizons=("horizon_h", "nunique"),
            predicted_regimes=("predicted_regime", "nunique"),
            rmse=("residual_mm", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
        )
        .reset_index()
    )
    candidates = event_info.merge(metric, on=["site_id", "layer", "event_id"], how="inner")
    candidates = candidates[(candidates["duration_h"] >= MIN_FIG_EVENT_H) & (candidates["forecasts"] >= 8)]
    rows = []
    for layer in LAYER_ORDER:
        layer_candidates = candidates[candidates["layer"].eq(layer)].copy()
        if layer_candidates.empty:
            fallback = valid[valid["layer"].eq(layer)]
            if fallback.empty:
                continue
            row = fallback.iloc[0]
            rows.append({"site_id": int(row.site_id), "layer": layer, "event_id": str(row.event_id)})
            continue
        target_rmse = float(layer_candidates["rmse"].quantile(0.60))
        layer_candidates["score"] = (
            np.abs(layer_candidates["rmse"] - target_rmse)
            - 0.30 * layer_candidates["predicted_regimes"]
            - 0.08 * layer_candidates["horizons"]
        )
        row = layer_candidates.sort_values("score").iloc[0]
        rows.append({"site_id": int(row.site_id), "layer": layer, "event_id": str(row.event_id)})
    return pd.DataFrame(rows)


def prediction_envelope(pred: pd.DataFrame, event_id: str) -> pd.DataFrame:
    event_pred = pred[(pred["event_id"].eq(event_id)) & pred["model"].astype(str).eq(MAIN_MODEL)].dropna(subset=["pred_mm"])
    if event_pred.empty:
        return event_pred
    return (
        event_pred.groupby("target_t_h", observed=True)
        .agg(pred_med=("pred_mm", "median"), pred_q10=("pred_mm", lambda x: float(np.quantile(x, 0.10))), pred_q90=("pred_mm", lambda x: float(np.quantile(x, 0.90))))
        .reset_index()
        .sort_values("target_t_h")
    )


def plot_training_csr_library(ax: plt.Axes, train_points_path: Path, layer: str | None = None) -> None:
    train_points = pd.read_parquet(train_points_path)
    train_points = train_points[train_points["segment_regime"].isin(REGIME_ORDER)].copy()
    if layer is not None:
        train_points = train_points[train_points["layer"].eq(layer)].copy()
    if train_points.empty:
        ax.text(0.5, 0.5, "No training segments", transform=ax.transAxes, ha="center", va="center")
        return
    train_points["seg_norm_t"] = train_points.groupby("adaptive_segment_id", observed=True)["segment_t_h"].transform(
        lambda s: s / s.max() if s.max() > 0 else s
    )
    for regime in REGIME_ORDER:
        use = train_points[train_points["segment_regime"].eq(regime)]
        if use.empty:
            continue
        for _, seg in use.groupby("adaptive_segment_id", observed=True):
            stable = sum(bytearray(str(seg["adaptive_segment_id"].iloc[0]).encode("utf-8")))
            if np.random.default_rng(stable).random() > 0.035:
                continue
            y0 = float(seg["moisture_mm"].iloc[0])
            y1 = float(seg["moisture_mm"].iloc[-1])
            denom = max(y0 - y1, 1e-6)
            ax.plot(seg["seg_norm_t"], (seg["moisture_mm"] - y1) / denom, color=REGIME_COLORS[regime], alpha=0.10, lw=0.55)
        binned = (
            use.assign(
                y_norm=use.groupby("adaptive_segment_id", observed=True)["moisture_mm"].transform(
                    lambda s: (s - s.iloc[-1]) / max(s.iloc[0] - s.iloc[-1], 1e-6)
                ),
                bin=lambda d: pd.cut(d["seg_norm_t"], np.linspace(0, 1, 26), include_lowest=True),
            )
            .groupby("bin", observed=True)
            .agg(x=("seg_norm_t", "mean"), y=("y_norm", "median"))
            .dropna()
        )
        ax.plot(binned["x"], binned["y"], color=REGIME_COLORS[regime], lw=1.8, label=REGIME_LABELS[regime])
    title_suffix = "all depths" if layer is None else layer_label(layer)
    ax.set_title("b  Regime-specific CSR libraries", loc="left", fontweight="bold")
    ax.text(0.02, 0.93, f"Training set: 2023-2025, {title_suffix}", transform=ax.transAxes, fontsize=6.2, color="#555555")
    ax.set_xlabel("Normalized segment time")
    ax.set_ylabel("Normalized segment storage")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, color="#E8EDF1", lw=0.55)
    ax.legend(loc="lower left", fontsize=6)


def heatmap_matrix(site_layer: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[int]]:
    sites = sorted(site_layer["site_id"].dropna().astype(int).unique().tolist())
    matrix = np.full((len(sites), len(LAYER_ORDER)), np.nan)
    for i, site_id in enumerate(sites):
        for j, layer in enumerate(LAYER_ORDER):
            cell = site_layer[(site_layer["site_id"].eq(site_id)) & (site_layer["layer"].eq(layer))]
            if not cell.empty:
                matrix[i, j] = float(cell[value_col].iloc[0])
    return matrix, sites


def plot_external_figure(points: pd.DataFrame, pred: pd.DataFrame, site_layer: pd.DataFrame) -> None:
    showcases = choose_layer_showcase_events(points, pred)
    fig = plt.figure(figsize=(7.25, 7.65))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.26, 1.0],
        height_ratios=[0.78, 1.0],
        left=0.075,
        right=0.97,
        bottom=0.075,
        top=0.865,
        hspace=0.42,
        wspace=0.34,
    )
    left = gs[:, 0].subgridspec(len(LAYER_ORDER), 1, hspace=0.12)
    axes_a = [fig.add_subplot(left[i, 0]) for i in range(len(LAYER_ORDER))]
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    for ax, layer_name in zip(axes_a, LAYER_ORDER):
        row = showcases[showcases["layer"].eq(layer_name)]
        if row.empty:
            ax.text(0.5, 0.5, f"No {layer_label(layer_name)} forecast", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        site_id = int(row["site_id"].iloc[0])
        event_id = str(row["event_id"].iloc[0])
        event = points[(points["site_id"].eq(site_id)) & points["layer"].eq(layer_name) & points["event_id"].eq(event_id)].sort_values("t_h")
        envelope = prediction_envelope(pred, event_id)
        ax.plot(event["t_h"], event["moisture_mm"], color="#222222", lw=1.25, label="Observed 2026 SMDE")
        if not envelope.empty:
            ax.fill_between(envelope["target_t_h"], envelope["pred_q10"], envelope["pred_q90"], color="#D9A441", alpha=0.20, lw=0, label="Forecast 10-90%")
            ax.plot(envelope["target_t_h"], envelope["pred_med"], color="#C67B2E", lw=1.35, label="Forecast median")
        origin_table = (
            pred[(pred["event_id"].eq(event_id)) & pred["model"].astype(str).eq(MAIN_MODEL)]
            .drop_duplicates("origin_id")
            .sort_values("origin_t_h")
        )
        step = max(1, len(origin_table) // 4)
        for _, origin in origin_table.iloc[::step].iterrows():
            regime = str(origin.get("predicted_regime", ""))
            ax.scatter(origin["origin_t_h"], origin["s0_mm"], s=13, color=REGIME_COLORS.get(regime, "#777777"), zorder=3)
        ax.text(
            0.012,
            0.82,
            f"{layer_label(layer_name)}  S{site_id}",
            transform=ax.transAxes,
            fontsize=6.5,
            fontweight="bold",
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        ax.grid(True, color="#E8EDF1", lw=0.50)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("S (mm)", fontsize=6.5)
        if ax is not axes_a[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Elapsed time since drydown start (h)")

    axes_a[0].set_title("a  External 2026 forecasts by sensor depth", loc="left", fontweight="bold")
    obs_handle = plt.Line2D([0], [0], color="#222222", lw=1.25, label="Observed")
    pred_handle = plt.Line2D([0], [0], color="#C67B2E", lw=1.35, label="Forecast median")
    shade_handle = mpl.patches.Patch(facecolor="#D9A441", alpha=0.20, label="Forecast 10-90%")
    axes_a[-1].legend(handles=[obs_handle, pred_handle, shade_handle], loc="lower left", ncol=1, fontsize=5.8)

    train_points_path = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data" / "adaptive_segment_points_all.parquet"
    plot_training_csr_library(ax_b, train_points_path, layer=None)

    rmse_matrix, sites = heatmap_matrix(site_layer, "rmse_mm")
    ccc_matrix, _ = heatmap_matrix(site_layer, "ccc")
    show_metric = rmse_matrix
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad("#EFEFEF")
    vmax = max(0.5, float(np.nanpercentile(show_metric, 92))) if np.isfinite(show_metric).any() else 1.0
    im = ax_c.imshow(np.ma.masked_invalid(show_metric), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax_c.set_title("c  External 2026 validation by station-layer\ncolor = RMSE, text = CCC", loc="left", fontweight="bold")
    ax_c.set_xticks(np.arange(len(LAYER_ORDER)))
    ax_c.set_xticklabels([layer_label(layer) for layer in LAYER_ORDER])
    ax_c.set_yticks(np.arange(len(sites)))
    ax_c.set_yticklabels([str(s) for s in sites], fontsize=5.2)
    ax_c.set_xlabel("Sensor depth")
    ax_c.set_ylabel("FAWN station")
    ax_c.set_xticks(np.arange(-0.5, len(LAYER_ORDER), 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(sites), 1), minor=True)
    ax_c.grid(which="minor", color="white", lw=0.55)
    ax_c.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.025)
    cbar.ax.set_title("RMSE\n(mm)", fontsize=6, pad=4)
    for i, site in enumerate(sites):
        for j, layer_name in enumerate(LAYER_ORDER):
            cell = site_layer[(site_layer["site_id"].eq(site)) & (site_layer["layer"].eq(layer_name))]
            if cell.empty or not np.isfinite(float(cell["ccc"].iloc[0])):
                continue
            ax_c.text(j, i, f"{float(cell['ccc'].iloc[0]):.2f}", ha="center", va="center", fontsize=4.5, color="#1F1F1F")

    fig.suptitle("Out-of-year 2026 observations test online regime-conditioned CSR forecasting", x=0.02, y=0.99, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.02, 0.955, "The forecast library and regime classifier are built from 2023-2025; 2026 soil moisture is used only for SMDE detection, forecast origins, and scoring. Colored origin points indicate the online-predicted regime.", fontsize=7, color="#555555")
    save_pub(fig, "fig_2026_external_regime_segmented_forecast")
    plt.close(fig)


def write_report(events: pd.DataFrame, points: pd.DataFrame, origins: pd.DataFrame, pred: pd.DataFrame, site_layer: pd.DataFrame) -> None:
    overall = []
    valid = pred[pred["model"].astype(str).eq(MAIN_MODEL)].dropna(subset=["pred_mm", "target_mm"])
    for horizon, group in valid.groupby("horizon_h"):
        row = {"horizon_h": int(horizon)}
        row.update(metrics_for_group(group))
        overall.append(row)
    overall_df = pd.DataFrame(overall).sort_values("horizon_h")
    overall_df.to_csv(SOURCE / "external_2026_metrics_by_horizon.csv", index=False)
    model_rows = []
    for (model, horizon), group in pred.dropna(subset=["pred_mm", "target_mm"]).groupby(["model", "horizon_h"], observed=True):
        row = {"model": str(model), "horizon_h": int(horizon)}
        row.update(metrics_for_group(group))
        model_rows.append(row)
    model_df = pd.DataFrame(model_rows).sort_values(["model", "horizon_h"]) if model_rows else pd.DataFrame()
    model_df.to_csv(SOURCE / "external_2026_metrics_by_model_horizon.csv", index=False)
    regime_distribution = (
        pred[pred["model"].astype(str).eq(MAIN_MODEL)]
        .drop_duplicates("origin_id")
        .groupby(["layer", "predicted_regime"], observed=True)
        .agg(origins=("origin_id", "nunique"))
        .reset_index()
        .sort_values(["layer", "predicted_regime"])
    )
    regime_distribution.to_csv(SOURCE / "external_2026_online_regime_distribution.csv", index=False)
    report = f"""# Experiment 4b: 2026 external regime-segmented forecast

## Design

This experiment uses 2026 FAWN soil-moisture observations as an out-of-year external demonstration and validation set. The 2023-2025 hydrologically registered CSR library remains the source for regime-conditioned forecasts. The 2026 observations are used only to detect SMDEs, define forecast origins, and score observed future soil water amounts. No 2026 adaptive regime segmentation is used for prediction. At each forecast origin, the regime is predicted online from current and past-state features by a classifier trained on 2023-2025 diagnosed segments.

## Sample

- Detected 2026 SMDEs: {len(events):,}
- 2026 SMDE observation points: {len(points):,}
- 2026 forecast rows: {len(origins):,}
- 2026 external station-layers with forecasts: {len(site_layer):,}

## Overall External Forecast Skill

{overall_df.to_markdown(index=False) if not overall_df.empty else "_No external forecasts._"}

## Station-Layer External Validation

{site_layer.to_markdown(index=False) if not site_layer.empty else "_No station-layer validation._"}

## Online Regime Distribution

{regime_distribution.to_markdown(index=False) if not regime_distribution.empty else "_No online regime predictions._"}
"""
    (OUT / "experiment4b_2026_external_forecast_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_out()
    train_origins = pd.read_parquet(SOURCE_EXP4 / "forecast_origin_rows.parquet")
    train_rows = train_origins[train_origins["split"].eq("train")].copy()
    csr_library = pd.read_csv(CSR_LIBRARY) if CSR_LIBRARY.exists() else pd.DataFrame()
    events, points, status = build_2026_external_points()
    status.to_csv(SOURCE / "external_2026_processing_status.csv", index=False)
    events.to_csv(SOURCE / "external_2026_detected_smde_events.csv", index=False)
    points.to_parquet(SOURCE / "external_2026_detected_smde_points.parquet", index=False)
    origins = build_external_forecast_origins(points)
    origins.to_parquet(SOURCE / "external_2026_forecast_origin_rows.parquet", index=False)
    if origins.empty:
        raise RuntimeError("No 2026 external forecast origins were created.")
    regime_pred = predict_external_regime_probabilities(train_rows, origins)
    regime_pred.to_csv(SOURCE / "external_2026_online_regime_predictions_by_origin.csv", index=False)
    pred = build_external_predictions(train_rows, origins, regime_pred, csr_library)
    pred.to_parquet(SOURCE / "external_2026_regime_conditioned_forecast_predictions.parquet", index=False)
    site_layer = summarize_site_layer(pred)
    site_layer.to_csv(SOURCE / "external_2026_metrics_by_station_layer.csv", index=False)
    plot_external_figure(points, pred, site_layer)
    write_report(events, points, origins, pred, site_layer)
    print(f"Wrote 2026 external forecast outputs to {OUT}")


if __name__ == "__main__":
    main()
