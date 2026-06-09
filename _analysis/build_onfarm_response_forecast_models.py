from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
OUT = ANALYSIS / "onfarm_smde_response_forecast"
SOURCE = OUT / "source_data"
ORIGINS_FILE = SOURCE / "onfarm_forecast_origins.parquet"

MIN_GROUP_ROWS = 8
ALPHA_MIN = 0.0
ALPHA_MAX = 1.5
STATIC_SHRINK_K = 40.0
RESPONSE_SHRINK_K = 30.0

MODEL_ORDER = [
    "persistence",
    "recent_slope",
    "static_recession",
    "response_library_analog",
    "response_regime_rate",
    "response_input_regime_rate",
]

ANALOG_FEATURE_COLS = [
    "origin_t_h",
    "s0_value",
    "drop_sofar_value",
    "recent_loss_1h_value",
    "start_value",
    "rain_before_48",
    "irrigation_before_48_count",
    "start_month",
]
MIN_ANALOG_ROWS = 10
K_NEIGHBORS = 15


def ensure_out() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)


def fit_alpha(frame: pd.DataFrame) -> float:
    x = frame["recent_linear_loss_value"].to_numpy(float)
    y = frame["target_loss_value"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 1e-9)
    if int(ok.sum()) < MIN_GROUP_ROWS:
        return np.nan
    alpha = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2))
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def fit_coefficients(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_rows(level: str, cols: list[str]) -> None:
        for keys, group in train.groupby(cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(cols, keys))
            rec.update({"level": level, "rows": int(len(group)), "alpha": fit_alpha(group)})
            rows.append(rec)

    add_rows("horizon", ["horizon_h"])
    add_rows("layer_horizon", ["layer", "horizon_h"])
    add_rows("farm_probe_layer_horizon", ["farm_name", "probe_id", "layer", "horizon_h"])
    add_rows("regime_horizon", ["regime_proxy", "horizon_h"])
    add_rows("layer_regime_horizon", ["layer", "regime_proxy", "horizon_h"])
    add_rows(
        "farm_probe_layer_regime_horizon",
        ["farm_name", "probe_id", "layer", "regime_proxy", "horizon_h"],
    )
    add_rows("input_regime_horizon", ["input_source_48h", "regime_proxy", "horizon_h"])
    add_rows(
        "layer_input_regime_horizon",
        ["layer", "input_source_48h", "regime_proxy", "horizon_h"],
    )
    add_rows(
        "farm_probe_layer_input_regime_horizon",
        ["farm_name", "probe_id", "layer", "input_source_48h", "regime_proxy", "horizon_h"],
    )
    coeffs = pd.DataFrame(rows)
    return coeffs.sort_values(["level", "horizon_h", "farm_name", "probe_id", "layer", "input_source_48h", "regime_proxy"], na_position="last")


def table_for(coeffs: pd.DataFrame, level: str, cols: list[str]) -> dict[tuple[Any, ...], tuple[float, int]]:
    out: dict[tuple[Any, ...], tuple[float, int]] = {}
    use = coeffs[coeffs["level"].eq(level)].copy()
    for row in use.itertuples(index=False):
        alpha = float(getattr(row, "alpha"))
        if not np.isfinite(alpha):
            continue
        key = tuple(getattr(row, col) for col in cols)
        out[key] = (alpha, int(getattr(row, "rows")))
    return out


def coefficient_maps(coeffs: pd.DataFrame) -> dict[str, dict[tuple[Any, ...], tuple[float, int]]]:
    return {
        "horizon": table_for(coeffs, "horizon", ["horizon_h"]),
        "layer_horizon": table_for(coeffs, "layer_horizon", ["layer", "horizon_h"]),
        "farm_probe_layer_horizon": table_for(
            coeffs,
            "farm_probe_layer_horizon",
            ["farm_name", "probe_id", "layer", "horizon_h"],
        ),
        "regime_horizon": table_for(coeffs, "regime_horizon", ["regime_proxy", "horizon_h"]),
        "layer_regime_horizon": table_for(coeffs, "layer_regime_horizon", ["layer", "regime_proxy", "horizon_h"]),
        "farm_probe_layer_regime_horizon": table_for(
            coeffs,
            "farm_probe_layer_regime_horizon",
            ["farm_name", "probe_id", "layer", "regime_proxy", "horizon_h"],
        ),
        "input_regime_horizon": table_for(
            coeffs,
            "input_regime_horizon",
            ["input_source_48h", "regime_proxy", "horizon_h"],
        ),
        "layer_input_regime_horizon": table_for(
            coeffs,
            "layer_input_regime_horizon",
            ["layer", "input_source_48h", "regime_proxy", "horizon_h"],
        ),
        "farm_probe_layer_input_regime_horizon": table_for(
            coeffs,
            "farm_probe_layer_input_regime_horizon",
            ["farm_name", "probe_id", "layer", "input_source_48h", "regime_proxy", "horizon_h"],
        ),
    }


def _lookup(mapping: dict[tuple[Any, ...], tuple[float, int]], key: tuple[Any, ...]) -> tuple[float, int]:
    return mapping.get(key, (np.nan, 0))


def static_alpha(row: pd.Series, maps: dict[str, dict[tuple[Any, ...], tuple[float, int]]]) -> tuple[float, str]:
    horizon = int(row["horizon_h"])
    layer = str(row["layer"])
    farm = row["farm_name"]
    probe = row["probe_id"]
    base, base_n = _lookup(maps["horizon"], (horizon,))
    if not np.isfinite(base):
        return np.nan, "missing"
    layer_alpha, layer_n = _lookup(maps["layer_horizon"], (layer, horizon))
    parent = layer_alpha if np.isfinite(layer_alpha) else base
    parent_n = layer_n if np.isfinite(layer_alpha) else base_n
    local_alpha, local_n = _lookup(maps["farm_probe_layer_horizon"], (farm, probe, layer, horizon))
    if np.isfinite(local_alpha):
        w = local_n / (local_n + STATIC_SHRINK_K)
        return float(w * local_alpha + (1.0 - w) * parent), f"farm-probe-layer shrink n={local_n}"
    if parent_n > base_n:
        return float(parent), "layer fallback"
    return float(base), "horizon fallback"


def response_regime_alpha(row: pd.Series, maps: dict[str, dict[tuple[Any, ...], tuple[float, int]]]) -> tuple[float, str]:
    horizon = int(row["horizon_h"])
    layer = str(row["layer"])
    farm = row["farm_name"]
    probe = row["probe_id"]
    regime = str(row["regime_proxy"])
    base, base_n = _lookup(maps["horizon"], (horizon,))
    if not np.isfinite(base):
        return np.nan, "missing"
    regime_alpha, regime_n = _lookup(maps["regime_horizon"], (regime, horizon))
    parent = regime_alpha if np.isfinite(regime_alpha) else base
    parent_n = regime_n if np.isfinite(regime_alpha) else base_n
    layer_alpha, layer_n = _lookup(maps["layer_regime_horizon"], (layer, regime, horizon))
    if np.isfinite(layer_alpha):
        w = layer_n / (layer_n + RESPONSE_SHRINK_K)
        parent = w * layer_alpha + (1.0 - w) * parent
        parent_n += layer_n
    local_alpha, local_n = _lookup(maps["farm_probe_layer_regime_horizon"], (farm, probe, layer, regime, horizon))
    if np.isfinite(local_alpha):
        w = local_n / (local_n + RESPONSE_SHRINK_K)
        return float(w * local_alpha + (1.0 - w) * parent), f"farm-probe-layer-regime shrink n={local_n}"
    if parent_n > base_n:
        return float(parent), "layer/regime fallback"
    return float(base), "horizon fallback"


def response_input_regime_alpha(row: pd.Series, maps: dict[str, dict[tuple[Any, ...], tuple[float, int]]]) -> tuple[float, str]:
    horizon = int(row["horizon_h"])
    layer = str(row["layer"])
    farm = row["farm_name"]
    probe = row["probe_id"]
    input_class = str(row["input_source_48h"])
    regime = str(row["regime_proxy"])
    base_alpha, base_level = response_regime_alpha(row, maps)
    if not np.isfinite(base_alpha):
        return np.nan, "missing"
    input_alpha, input_n = _lookup(maps["input_regime_horizon"], (input_class, regime, horizon))
    parent = input_alpha if np.isfinite(input_alpha) else base_alpha
    parent_n = input_n if np.isfinite(input_alpha) else 0
    layer_alpha, layer_n = _lookup(maps["layer_input_regime_horizon"], (layer, input_class, regime, horizon))
    if np.isfinite(layer_alpha):
        w = layer_n / (layer_n + RESPONSE_SHRINK_K)
        parent = w * layer_alpha + (1.0 - w) * parent
        parent_n += layer_n
    local_alpha, local_n = _lookup(
        maps["farm_probe_layer_input_regime_horizon"],
        (farm, probe, layer, input_class, regime, horizon),
    )
    if np.isfinite(local_alpha):
        w = local_n / (local_n + RESPONSE_SHRINK_K)
        return float(w * local_alpha + (1.0 - w) * parent), f"farm-probe-layer-input-regime shrink n={local_n}"
    if parent_n > 0:
        return float(parent), "input/regime fallback"
    return float(base_alpha), base_level


def build_predictions(train: pd.DataFrame, test: pd.DataFrame, coeffs: pd.DataFrame) -> pd.DataFrame:
    maps = coefficient_maps(coeffs)
    frames: list[pd.DataFrame] = []
    base_cols = [
        "forecast_id",
        "farm_name",
        "probe_id",
        "weather_station_id",
        "layer",
        "layer_depth",
        "event_id",
        "origin_id",
        "split",
        "start",
        "origin_t_h",
        "horizon_h",
        "target_t_h",
        "s0_value",
        "target_value",
        "delta_value",
        "start_value",
        "drop_sofar_value",
        "recent_loss_1h_value",
        "target_loss_value",
        "recent_linear_loss_value",
        "input_source_48h",
        "regime_proxy",
        "rain_before_48",
        "irrigation_before_48_count",
        "start_month",
    ]
    base = test[base_cols].copy()

    persistence = base[["forecast_id", "s0_value"]].copy()
    persistence["model"] = "persistence"
    persistence["pred_value"] = persistence["s0_value"]
    persistence["model_level"] = "none"
    frames.append(persistence[["forecast_id", "model", "pred_value", "model_level"]])

    recent = base[["forecast_id", "s0_value", "recent_linear_loss_value"]].copy()
    recent["model"] = "recent_slope"
    recent["pred_value"] = recent["s0_value"] - recent["recent_linear_loss_value"]
    recent["model_level"] = "none"
    frames.append(recent[["forecast_id", "model", "pred_value", "model_level"]])

    frames.append(response_library_analog_predictions(train, test))

    for model, alpha_func in [
        ("static_recession", static_alpha),
        ("response_regime_rate", response_regime_alpha),
        ("response_input_regime_rate", response_input_regime_alpha),
    ]:
        alpha_level = base.apply(lambda row: alpha_func(row, maps), axis=1)
        alpha = alpha_level.map(lambda x: x[0]).astype(float)
        level = alpha_level.map(lambda x: x[1]).astype(str)
        out = base[["forecast_id", "s0_value", "recent_linear_loss_value"]].copy()
        out["model"] = model
        out["pred_value"] = out["s0_value"] - alpha.to_numpy(float) * out["recent_linear_loss_value"].to_numpy(float)
        out.loc[~np.isfinite(alpha.to_numpy(float)), "pred_value"] = np.nan
        out["model_level"] = level
        frames.append(out[["forecast_id", "model", "pred_value", "model_level"]])

    pred = pd.concat(frames, ignore_index=True).merge(base, on="forecast_id", how="left", validate="many_to_one")
    pred["residual_value"] = pred["pred_value"] - pred["target_value"]
    pred["model"] = pd.Categorical(pred["model"].astype(str), categories=MODEL_ORDER, ordered=True)
    return pred.sort_values(["model", "farm_name", "probe_id", "layer", "event_id", "origin_t_h", "horizon_h"]).reset_index(drop=True)


def group_key_iter(frame: pd.DataFrame, cols: list[str]):
    if not cols:
        yield (), frame.index.to_numpy()
        return
    for key, idx in frame.groupby(cols, observed=True, sort=False, dropna=False).groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        yield key, np.asarray(list(idx), dtype=int)


def matching_pool(frame: pd.DataFrame, cols: list[str], key: tuple[object, ...]) -> pd.DataFrame:
    pool = frame
    for col, value in zip(cols, key):
        pool = pool[pool[col].eq(value)]
    return pool


def query_analog_pool(pool: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pool = pool.dropna(subset=ANALOG_FEATURE_COLS + ["delta_value"]).copy()
    query = query.dropna(subset=ANALOG_FEATURE_COLS).copy()
    n_pool = len(pool)
    if n_pool < MIN_ANALOG_ROWS or query.empty:
        return np.array([]), np.array([]), np.array([]), n_pool
    x_pool = pool[ANALOG_FEATURE_COLS].to_numpy(float)
    x_query = query[ANALOG_FEATURE_COLS].to_numpy(float)
    center = np.nanmean(x_pool, axis=0)
    scale = np.nanstd(x_pool, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    x_pool = (x_pool - center) / scale
    x_query = (x_query - center) / scale
    k = min(K_NEIGHBORS, n_pool)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_pool)
    _, indices = nn.kneighbors(x_query)
    deltas = pool["delta_value"].to_numpy(float)
    neighbor_delta = deltas[indices]
    return (
        np.nanmedian(neighbor_delta, axis=1),
        np.nanpercentile(neighbor_delta, 10, axis=1),
        np.nanpercentile(neighbor_delta, 90, axis=1),
        n_pool,
    )


def response_library_analog_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = test[
        [
            "forecast_id",
            "farm_name",
            "probe_id",
            "layer",
            "input_source_48h",
            "regime_proxy",
            "horizon_h",
            *ANALOG_FEATURE_COLS,
        ]
    ].copy()
    out["pred_delta_value"] = np.nan
    out["q10_delta_value"] = np.nan
    out["q90_delta_value"] = np.nan
    out["model_level"] = ""
    out["analog_candidates"] = 0

    levels = [
        ("farm-probe-layer-input-regime", ["farm_name", "probe_id", "layer", "input_source_48h", "regime_proxy"]),
        ("farm-probe-layer-regime", ["farm_name", "probe_id", "layer", "regime_proxy"]),
        ("layer-input-regime", ["layer", "input_source_48h", "regime_proxy"]),
        ("layer-regime", ["layer", "regime_proxy"]),
        ("layer", ["layer"]),
        ("network", []),
    ]
    for horizon in sorted(test["horizon_h"].dropna().unique()):
        train_h = train[train["horizon_h"].eq(horizon)]
        if train_h.empty:
            continue
        horizon_idx = test.index[test["horizon_h"].eq(horizon)].to_numpy()
        pending = np.intersect1d(horizon_idx, out.index[out["pred_delta_value"].isna()].to_numpy())
        for level_name, cols in levels:
            if len(pending) == 0:
                break
            pending_frame = test.loc[pending]
            for key, idx in group_key_iter(pending_frame, cols):
                pool = matching_pool(train_h, cols, key)
                if len(pool) < MIN_ANALOG_ROWS:
                    continue
                query = test.loc[idx]
                pred_delta, q10, q90, n_pool = query_analog_pool(pool, query)
                if len(pred_delta) == 0:
                    continue
                out.loc[idx, "pred_delta_value"] = pred_delta
                out.loc[idx, "q10_delta_value"] = q10
                out.loc[idx, "q90_delta_value"] = q90
                out.loc[idx, "model_level"] = level_name
                out.loc[idx, "analog_candidates"] = int(n_pool)
            pending = np.intersect1d(horizon_idx, out.index[out["pred_delta_value"].isna()].to_numpy())

    pred = test[["forecast_id", "s0_value"]].copy()
    pred["model"] = "response_library_analog"
    pred["pred_value"] = pred["s0_value"].to_numpy(float) + out["pred_delta_value"].to_numpy(float)
    pred["model_level"] = out["model_level"].astype(str).to_numpy()
    return pred[["forecast_id", "model", "pred_value", "model_level"]]


def concordance_correlation_coefficient(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 2:
        return np.nan
    mean_obs = float(np.mean(obs))
    mean_pred = float(np.mean(pred))
    var_obs = float(np.var(obs))
    var_pred = float(np.var(pred))
    cov = float(np.mean((obs - mean_obs) * (pred - mean_pred)))
    denom = var_obs + var_pred + (mean_obs - mean_pred) ** 2
    return float((2.0 * cov) / denom) if denom > 0 else np.nan


def r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 2:
        return np.nan
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def metric_row(frame: pd.DataFrame, model: str, horizon: int, comparison_set: str) -> dict[str, Any]:
    valid = frame.dropna(subset=["pred_value", "target_value"]).copy()
    obs = valid["target_value"].to_numpy(float)
    pred = valid["pred_value"].to_numpy(float)
    residual = pred - obs
    return {
        "comparison_set": comparison_set,
        "model": model,
        "horizon_h": int(horizon),
        "forecasts": int(len(valid)),
        "farms": int(valid["farm_name"].nunique()) if len(valid) else 0,
        "probes": int(valid["probe_id"].nunique()) if len(valid) else 0,
        "events": int(valid["event_id"].nunique()) if len(valid) else 0,
        "origins": int(valid["origin_id"].nunique()) if len(valid) else 0,
        "rmse_value": float(np.sqrt(np.nanmean(residual**2))) if len(valid) else np.nan,
        "mae_value": float(np.nanmean(np.abs(residual))) if len(valid) else np.nan,
        "bias_value": float(np.nanmean(residual)) if len(valid) else np.nan,
        "ccc": concordance_correlation_coefficient(obs, pred) if len(valid) else np.nan,
        "r2": r2_score(obs, pred) if len(valid) else np.nan,
    }


def common_origin_subset(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    valid = pred[pred["model"].astype(str).isin(models)].dropna(subset=["pred_value", "target_value"]).copy()
    counts = valid.groupby(["forecast_id", "horizon_h"], observed=True)["model"].nunique().reset_index(name="valid_models")
    ids = counts[counts["valid_models"].eq(len(models))][["forecast_id", "horizon_h"]]
    return pred[pred["model"].astype(str).isin(models)].merge(ids, on=["forecast_id", "horizon_h"], how="inner")


def summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_sets = {
        "model_valid": MODEL_ORDER,
        "all_common_origin": MODEL_ORDER,
        "static_vs_response_common": ["static_recession", "response_input_regime_rate"],
    }
    rows: list[dict[str, Any]] = []
    for comparison_set, models in comparison_sets.items():
        use = pred[pred["model"].astype(str).isin(models)].copy()
        if comparison_set.endswith("common") or comparison_set.endswith("common_origin") or comparison_set == "static_vs_response_common":
            use = common_origin_subset(pred, models)
        for (model, horizon), group in use.groupby(["model", "horizon_h"], observed=True):
            rows.append(metric_row(group, str(model), int(horizon), comparison_set))
    metrics = pd.DataFrame(rows).sort_values(["comparison_set", "horizon_h", "model"]).reset_index(drop=True)

    input_rows: list[dict[str, Any]] = []
    use = common_origin_subset(pred, ["static_recession", "response_input_regime_rate"])
    for (input_class, model, horizon), group in use.groupby(["input_source_48h", "model", "horizon_h"], observed=True, dropna=False):
        rec = metric_row(group, str(model), int(horizon), "static_vs_response_common_by_input")
        rec["input_source_48h"] = input_class
        input_rows.append(rec)
    by_input = pd.DataFrame(input_rows).sort_values(["input_source_48h", "horizon_h", "model"]).reset_index(drop=True)

    total = pred[["forecast_id", "horizon_h"]].drop_duplicates()
    total_by_h = total.groupby("horizon_h", observed=True)["forecast_id"].nunique().to_dict()
    cov_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        valid = pred[pred["model"].astype(str).eq(model)].dropna(subset=["pred_value", "target_value"])
        by_h = valid.groupby("horizon_h", observed=True)["forecast_id"].nunique().to_dict()
        for horizon, total_n in total_by_h.items():
            valid_n = int(by_h.get(horizon, 0))
            cov_rows.append(
                {
                    "model": model,
                    "horizon_h": int(horizon),
                    "total_forecast_origins": int(total_n),
                    "valid_forecasts": valid_n,
                    "coverage": valid_n / total_n if total_n else np.nan,
                }
            )
    coverage = pd.DataFrame(cov_rows).sort_values(["horizon_h", "model"]).reset_index(drop=True)
    return metrics, by_input, coverage


def write_report(metrics: pd.DataFrame, by_input: pd.DataFrame, coverage: pd.DataFrame, coeffs: pd.DataFrame) -> None:
    main = metrics[metrics["comparison_set"].eq("static_vs_response_common")].copy()
    lines = [
        "# On-farm response forecast model comparison",
        "",
        "Material Passport:",
        "",
        "- Type: first-pass on-farm forecast model result",
        "- Status: exploratory, ready for 2 x 2 design review",
        "- Target: future `soil_moisture_value` in source-data units",
        "- Primary metrics: R2 and CCC; RMSE is retained as `rmse_value` until unit conversion is confirmed.",
        "",
        "Models:",
        "",
        "- `static_recession`: farm/probe/layer/horizon recent-loss damping without regime or input class.",
        "- `response_regime_rate`: same forecast form with regime-aware shrinkage.",
        "- `response_input_regime_rate`: same forecast form with input-class and regime-aware shrinkage.",
        "",
        "Main static-vs-response comparison:",
        "",
        main[["horizon_h", "model", "forecasts", "events", "r2", "ccc", "rmse_value", "mae_value", "bias_value"]].to_markdown(index=False),
        "",
        "By-input-class comparison:",
        "",
        by_input[
            ["input_source_48h", "horizon_h", "model", "forecasts", "events", "r2", "ccc", "rmse_value", "mae_value", "bias_value"]
        ].to_markdown(index=False),
        "",
        "Coverage:",
        "",
        coverage.to_markdown(index=False),
        "",
        "Coefficient support:",
        "",
        coeffs.groupby("level", dropna=False).agg(rows=("alpha", "size"), finite_alpha=("alpha", lambda s: int(np.isfinite(s).sum()))).reset_index().to_markdown(index=False),
        "",
        "Caveats:",
        "",
        "- The chronological test split is rain-only dominated, with limited irrigation-only and very sparse mixed-input test support.",
        "- Soil moisture units are retained from source files. R2 and CCC are the safest first comparison metrics.",
        "- This is not yet a full CSR registration model; it is the first regime/input-aware recent-loss model needed before adding CSR libraries.",
    ]
    (OUT / "onfarm_response_forecast_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs() -> None:
    ensure_out()
    if not ORIGINS_FILE.exists():
        raise FileNotFoundError(f"Missing {ORIGINS_FILE}. Run build_onfarm_response_forecast_origins.py first.")
    origins = pd.read_parquet(ORIGINS_FILE)
    train = origins[origins["split"].astype(str).eq("train")].copy()
    test = origins[origins["split"].astype(str).eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("Both train and test origins are required.")
    coeffs = fit_coefficients(train)
    pred = build_predictions(train, test, coeffs)
    metrics, by_input, coverage = summarize_predictions(pred)

    coeffs.to_csv(SOURCE / "onfarm_response_forecast_coefficients.csv", index=False)
    pred.to_parquet(SOURCE / "onfarm_response_forecast_predictions.parquet", index=False)
    metrics.to_csv(SOURCE / "onfarm_response_forecast_metrics.csv", index=False)
    by_input.to_csv(SOURCE / "onfarm_response_forecast_metrics_by_input.csv", index=False)
    coverage.to_csv(SOURCE / "onfarm_response_forecast_coverage.csv", index=False)
    manifest = {
        "origin_rows": int(len(origins)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "prediction_rows": int(len(pred)),
        "models": MODEL_ORDER,
        "min_group_rows": MIN_GROUP_ROWS,
        "static_shrink_k": STATIC_SHRINK_K,
        "response_shrink_k": RESPONSE_SHRINK_K,
        "target_units": "source soil_moisture_value units",
        "notes": [
            "First-pass on-farm static versus response-centric forecast comparison.",
            "Response-centric models are regime/input-aware recent-loss models; CSR registration is not yet included.",
            "Use R2 and CCC first because on-farm source-data units still need final confirmation before manuscript RMSE reporting.",
        ],
    }
    (SOURCE / "onfarm_response_forecast_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(metrics, by_input, coverage, coeffs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    write_outputs()
