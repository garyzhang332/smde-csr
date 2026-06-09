from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_onfarm_response_forecast_models import (
    concordance_correlation_coefficient,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
OUT = ANALYSIS / "response_centric_2x2"
SOURCE = OUT / "source_data"

FAWN_SOURCE = ANALYSIS / "experiment4c_all_train_rate_forecast" / "source_data"
FAWN_TRAIN = FAWN_SOURCE / "all_2023_2025_training_forecast_origins.parquet"
FAWN_TEST = FAWN_SOURCE / "external_2026_forecast_origins.parquet"
FAWN_METRICS = FAWN_SOURCE / "external_2026_all_train_forecast_metrics.csv"

ONFARM_SOURCE = ANALYSIS / "onfarm_smde_response_forecast" / "source_data"
ONFARM_ORIGINS = ONFARM_SOURCE / "onfarm_forecast_origins.parquet"
ONFARM_METRICS = ONFARM_SOURCE / "onfarm_response_forecast_metrics.csv"

MIN_THRESHOLD_EVENTS = 12
MIN_RATE_ROWS = 80


def ensure_out() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)


def _safe_quantile(values: pd.Series, q: float) -> float:
    arr = values.to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.nanquantile(arr, q))


def _threshold_record(
    frame: pd.DataFrame,
    level: str,
    keys: dict[str, Any],
    storage_col: str,
    target_col: str,
    start_col: str,
    event_col: str,
) -> dict[str, Any]:
    unique_events = frame.drop_duplicates(event_col)
    s_upper = _safe_quantile(unique_events[start_col], 0.75)
    lower_source = pd.concat([frame[storage_col], frame[target_col]], ignore_index=True)
    s_lower = _safe_quantile(lower_source, 0.10)
    if np.isfinite(s_upper) and np.isfinite(s_lower) and s_upper <= s_lower:
        values = pd.concat([frame[storage_col], frame[target_col], unique_events[start_col]], ignore_index=True)
        s_upper = _safe_quantile(values, 0.90)
        s_lower = _safe_quantile(values, 0.10)
    rec = {
        "level": level,
        "rows": int(len(frame)),
        "events": int(unique_events[event_col].nunique()),
        "s_upper": s_upper,
        "s_lower": s_lower,
    }
    rec.update(keys)
    return rec


def fit_thresholds(
    train: pd.DataFrame,
    local_cols: list[str],
    layer_col: str,
    storage_col: str,
    target_col: str,
    start_col: str,
    event_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        _threshold_record(train, "setting", {}, storage_col, target_col, start_col, event_col)
    ]
    for layer, group in train.groupby(layer_col, observed=True, sort=False):
        rows.append(
            _threshold_record(
                group,
                "layer",
                {layer_col: layer},
                storage_col,
                target_col,
                start_col,
                event_col,
            )
        )
    for keys, group in train.groupby(local_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            _threshold_record(
                group,
                "local",
                dict(zip(local_cols, keys)),
                storage_col,
                target_col,
                start_col,
                event_col,
            )
        )
    return pd.DataFrame(rows)


def attach_thresholds(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    local_cols: list[str],
    layer_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    setting = thresholds[thresholds["level"].eq("setting")].iloc[0]
    out["s_upper_setting"] = float(setting["s_upper"])
    out["s_lower_setting"] = float(setting["s_lower"])

    layer = thresholds[thresholds["level"].eq("layer")][
        [layer_col, "events", "s_upper", "s_lower"]
    ].rename(
        columns={
            "events": "threshold_events_layer",
            "s_upper": "s_upper_layer",
            "s_lower": "s_lower_layer",
        }
    )
    out = out.merge(layer, on=layer_col, how="left")

    local = thresholds[thresholds["level"].eq("local")][
        local_cols + ["events", "s_upper", "s_lower"]
    ].rename(
        columns={
            "events": "threshold_events_local",
            "s_upper": "s_upper_local",
            "s_lower": "s_lower_local",
        }
    )
    out = out.merge(local, on=local_cols, how="left")

    local_ok = (
        out["threshold_events_local"].fillna(0).ge(MIN_THRESHOLD_EVENTS)
        & np.isfinite(out["s_upper_local"])
        & np.isfinite(out["s_lower_local"])
        & out["s_upper_local"].gt(out["s_lower_local"])
    )
    layer_ok = (
        out["threshold_events_layer"].fillna(0).ge(MIN_THRESHOLD_EVENTS)
        & np.isfinite(out["s_upper_layer"])
        & np.isfinite(out["s_lower_layer"])
        & out["s_upper_layer"].gt(out["s_lower_layer"])
    )

    out["s_upper_static"] = out["s_upper_setting"]
    out["s_lower_static"] = out["s_lower_setting"]
    out["threshold_level"] = "setting"
    out.loc[layer_ok, "s_upper_static"] = out.loc[layer_ok, "s_upper_layer"]
    out.loc[layer_ok, "s_lower_static"] = out.loc[layer_ok, "s_lower_layer"]
    out.loc[layer_ok, "threshold_level"] = "layer"
    out.loc[local_ok, "s_upper_static"] = out.loc[local_ok, "s_upper_local"]
    out.loc[local_ok, "s_lower_static"] = out.loc[local_ok, "s_lower_local"]
    out.loc[local_ok, "threshold_level"] = "local"
    return out


def fit_rate_model(frame: pd.DataFrame, storage_col: str, target_col: str, horizon_col: str) -> dict[str, float]:
    work = frame.copy()
    denom = work["s_upper_static"].astype(float) - work["s_lower_static"].astype(float)
    x = (work[storage_col].astype(float) - work["s_lower_static"].astype(float)) / denom
    y = (work[storage_col].astype(float) - work[target_col].astype(float)) / work[horizon_col].astype(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(denom) & (denom > 1e-9) & (work[horizon_col].astype(float) > 0)
    ok &= y >= 0
    if int(ok.sum()) < MIN_RATE_ROWS:
        return {"b0": np.nan, "b1": np.nan, "b2": np.nan, "rate_cap": np.nan, "rows": int(ok.sum())}
    x_ok = np.clip(x[ok].to_numpy(float), 0.0, 1.0)
    y_ok = y[ok].to_numpy(float)
    design = np.column_stack([np.ones_like(x_ok), x_ok, x_ok**2])
    coef, *_ = np.linalg.lstsq(design, y_ok, rcond=None)
    return {
        "b0": float(coef[0]),
        "b1": float(coef[1]),
        "b2": float(coef[2]),
        "rate_cap": float(np.nanquantile(y_ok, 0.99)),
        "rows": int(ok.sum()),
    }


def fit_rate_models(
    train_with_thresholds: pd.DataFrame,
    local_cols: list[str],
    layer_col: str,
    storage_col: str,
    target_col: str,
    horizon_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(level: str, keys: dict[str, Any], group: pd.DataFrame) -> None:
        rec = {"level": level}
        rec.update(keys)
        rec.update(fit_rate_model(group, storage_col, target_col, horizon_col))
        rows.append(rec)

    add("setting", {}, train_with_thresholds)
    for layer, group in train_with_thresholds.groupby(layer_col, observed=True, sort=False):
        add("layer", {layer_col: layer}, group)
    for keys, group in train_with_thresholds.groupby(local_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        add("local", dict(zip(local_cols, keys)), group)

    return pd.DataFrame(rows)


def attach_rate_models(
    frame: pd.DataFrame,
    coeffs: pd.DataFrame,
    local_cols: list[str],
    layer_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    coef_cols = ["b0", "b1", "b2", "rate_cap", "rows"]
    setting = coeffs[coeffs["level"].eq("setting")].iloc[0]
    for col in coef_cols:
        out[f"{col}_setting"] = float(setting[col])

    layer = coeffs[coeffs["level"].eq("layer")][[layer_col] + coef_cols].rename(
        columns={col: f"{col}_layer" for col in coef_cols}
    )
    out = out.merge(layer, on=layer_col, how="left")

    local = coeffs[coeffs["level"].eq("local")][local_cols + coef_cols].rename(
        columns={col: f"{col}_local" for col in coef_cols}
    )
    out = out.merge(local, on=local_cols, how="left")

    local_ok = out["rows_local"].fillna(0).ge(MIN_RATE_ROWS) & np.isfinite(out["b0_local"])
    layer_ok = out["rows_layer"].fillna(0).ge(MIN_RATE_ROWS) & np.isfinite(out["b0_layer"])

    out["rate_model_level"] = "setting"
    for col in ["b0", "b1", "b2", "rate_cap"]:
        out[col] = out[f"{col}_setting"]
        out.loc[layer_ok, col] = out.loc[layer_ok, f"{col}_layer"]
        out.loc[local_ok, col] = out.loc[local_ok, f"{col}_local"]
    out.loc[layer_ok, "rate_model_level"] = "layer"
    out.loc[local_ok, "rate_model_level"] = "local"
    return out


def predict_static_threshold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    local_cols: list[str],
    layer_col: str,
    event_col: str,
    origin_col: str,
    horizon_col: str,
    storage_col: str,
    target_col: str,
    start_col: str,
    pred_col: str,
    residual_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = fit_thresholds(train, local_cols, layer_col, storage_col, target_col, start_col, event_col)
    train_thr = attach_thresholds(train, thresholds, local_cols, layer_col)
    coeffs = fit_rate_models(train_thr, local_cols, layer_col, storage_col, target_col, horizon_col)
    pred = attach_thresholds(test, thresholds, local_cols, layer_col)
    pred = attach_rate_models(pred, coeffs, local_cols, layer_col)

    denom = pred["s_upper_static"].astype(float) - pred["s_lower_static"].astype(float)
    s_norm = (pred[storage_col].astype(float) - pred["s_lower_static"].astype(float)) / denom
    s_norm = np.clip(s_norm.to_numpy(float), 0.0, 1.0)
    rate = pred["b0"].to_numpy(float) + pred["b1"].to_numpy(float) * s_norm + pred["b2"].to_numpy(float) * s_norm**2
    rate = np.clip(rate, 0.0, pred["rate_cap"].to_numpy(float))
    forecast = pred[storage_col].to_numpy(float) - rate * pred[horizon_col].to_numpy(float)
    forecast = np.maximum(forecast, pred["s_lower_static"].to_numpy(float))

    pred["model"] = "static_threshold_recession"
    pred[pred_col] = forecast
    pred[residual_col] = pred[pred_col].astype(float) - pred[target_col].astype(float)
    pred["origin_count_key"] = pred[origin_col]
    return pred, thresholds, coeffs


def summarize_predictions(
    pred: pd.DataFrame,
    *,
    setting: str,
    horizon_col: str,
    target_col: str,
    pred_col: str,
    residual_col: str,
    event_col: str,
    origin_col: str,
    site_col: str,
    rmse_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in pred.groupby(horizon_col, observed=True):
        valid = group.dropna(subset=[pred_col, target_col]).copy()
        obs = valid[target_col].to_numpy(float)
        fit = valid[pred_col].to_numpy(float)
        residual = valid[residual_col].to_numpy(float)
        rows.append(
            {
                "comparison_set": "model_valid",
                "model": "static_threshold_recession",
                "data_setting": setting,
                "horizon_h": int(horizon),
                "forecasts": int(len(valid)),
                "events": int(valid[event_col].nunique()) if len(valid) else 0,
                "origins": int(valid[origin_col].nunique()) if len(valid) else 0,
                "sites_or_probes": int(valid[site_col].nunique()) if len(valid) else 0,
                rmse_name: float(np.sqrt(np.nanmean(residual**2))) if len(valid) else np.nan,
                "mae": float(np.nanmean(np.abs(residual))) if len(valid) else np.nan,
                "bias": float(np.nanmean(residual)) if len(valid) else np.nan,
                "ccc": concordance_correlation_coefficient(obs, fit) if len(valid) else np.nan,
                "r2": r2_score(obs, fit) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_h").reset_index(drop=True)


def load_fawn_response_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(FAWN_METRICS)
    return metrics[
        metrics["comparison_set"].eq("model_valid")
        & metrics["model"].eq("local_regime_shrink_rate")
    ].copy()


def load_onfarm_response_metrics(model: str) -> pd.DataFrame:
    metrics = pd.read_csv(ONFARM_METRICS)
    return metrics[
        metrics["comparison_set"].eq("model_valid")
        & metrics["model"].eq(model)
    ].copy()


def build_summary(
    fawn_static: pd.DataFrame,
    fawn_response: pd.DataFrame,
    onfarm_static: pd.DataFrame,
    onfarm_response: pd.DataFrame,
    onfarm_response_model: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in fawn_static.itertuples(index=False):
        rows.append(
            {
                "data_setting": "FAWN rainfall-only",
                "model_family": "property_centric_static_threshold",
                "model_name": "static_threshold_recession",
                "horizon_h": int(row.horizon_h),
                "forecasts": int(row.forecasts),
                "events": int(row.events),
                "sites_or_probes": int(row.sites_or_probes),
                "coverage": 1.0,
                "r2": float(row.r2),
                "ccc": float(row.ccc),
                "rmse": float(row.rmse_mm),
                "mae": float(row.mae),
                "bias": float(row.bias),
                "unit_label": "mm",
            }
        )
    for row in fawn_response.itertuples(index=False):
        rows.append(
            {
                "data_setting": "FAWN rainfall-only",
                "model_family": "response_centric",
                "model_name": "local_regime_shrink_rate",
                "horizon_h": int(row.horizon_h),
                "forecasts": int(row.forecasts),
                "events": int(row.events),
                "sites_or_probes": int(row.sites),
                "coverage": 1.0,
                "r2": float(row.r2),
                "ccc": float(row.ccc),
                "rmse": float(row.rmse_mm),
                "mae": float(row.mae_mm),
                "bias": float(row.bias_mm),
                "unit_label": "mm",
            }
        )
    for row in onfarm_static.itertuples(index=False):
        rows.append(
            {
                "data_setting": "On-farm managed",
                "model_family": "property_centric_static_threshold",
                "model_name": "static_threshold_recession",
                "horizon_h": int(row.horizon_h),
                "forecasts": int(row.forecasts),
                "events": int(row.events),
                "sites_or_probes": int(row.sites_or_probes),
                "coverage": 1.0,
                "r2": float(row.r2),
                "ccc": float(row.ccc),
                "rmse": float(row.rmse_value),
                "mae": float(row.mae),
                "bias": float(row.bias),
                "unit_label": "source soil_moisture_value",
            }
        )
    for row in onfarm_response.itertuples(index=False):
        rows.append(
            {
                "data_setting": "On-farm managed",
                "model_family": "response_centric",
                "model_name": onfarm_response_model,
                "horizon_h": int(row.horizon_h),
                "forecasts": int(row.forecasts),
                "events": int(row.events),
                "sites_or_probes": int(row.probes),
                "coverage": 1.0,
                "r2": float(row.r2),
                "ccc": float(row.ccc),
                "rmse": float(row.rmse_value),
                "mae": float(row.mae_value),
                "bias": float(row.bias_value),
                "unit_label": "source soil_moisture_value",
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_h", "data_setting", "model_family"]).reset_index(drop=True)


def build_delta(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (setting, horizon), group in summary.groupby(["data_setting", "horizon_h"], sort=False):
        wide = group.pivot_table(index="horizon_h", columns="model_family", values=["r2", "ccc", "rmse"], aggfunc="first")
        if wide.empty:
            continue
        if "response_centric" not in wide.columns.get_level_values(1):
            continue
        if "property_centric_static_threshold" not in wide.columns.get_level_values(1):
            continue
        rows.append(
            {
                "data_setting": setting,
                "horizon_h": int(horizon),
                "delta_r2_response_minus_static_threshold": float(
                    wide[("r2", "response_centric")].iloc[0]
                    - wide[("r2", "property_centric_static_threshold")].iloc[0]
                ),
                "delta_ccc_response_minus_static_threshold": float(
                    wide[("ccc", "response_centric")].iloc[0]
                    - wide[("ccc", "property_centric_static_threshold")].iloc[0]
                ),
                "rmse_ratio_response_over_static_threshold": float(
                    wide[("rmse", "response_centric")].iloc[0]
                    / wide[("rmse", "property_centric_static_threshold")].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_h", "data_setting"]).reset_index(drop=True)


def write_report(
    main_summary: pd.DataFrame,
    main_delta: pd.DataFrame,
    library_summary: pd.DataFrame,
    library_delta: pd.DataFrame,
) -> None:
    lines = [
        "# Static-threshold 2 x 2 comparison",
        "",
        "Material Passport:",
        "",
        "- Type: preliminary property-centric versus response-centric comparison.",
        "- Static baseline: empirical FC/WP-like normalized-storage recession model.",
        "- Static baseline inputs: current storage, fixed local/layer/setting thresholds, and a fixed loss-storage relation.",
        "- Static baseline exclusions: no recent loss, no regime, no input-source class, no response library.",
        "- Main on-farm response model: `response_input_regime_rate`.",
        "- Sensitivity on-farm response model: `response_library_analog`.",
        "- Status: exploratory evidence for manuscript framing, not final manuscript table.",
        "",
        "Main static-threshold 2 x 2 summary:",
        "",
        main_summary[["data_setting", "model_family", "model_name", "horizon_h", "forecasts", "events", "r2", "ccc", "rmse", "unit_label"]].to_markdown(index=False),
        "",
        "Main response-minus-static-threshold deltas:",
        "",
        main_delta.to_markdown(index=False),
        "",
        "Response-library sensitivity summary:",
        "",
        library_summary[["data_setting", "model_family", "model_name", "horizon_h", "forecasts", "events", "r2", "ccc", "rmse", "unit_label"]].to_markdown(index=False),
        "",
        "Response-library response-minus-static-threshold deltas:",
        "",
        library_delta.to_markdown(index=False),
        "",
        "Interpretation:",
        "",
        "- This baseline is closer to the property-centric contrast in the revised theory.",
        "- It should be called an empirical static-threshold baseline, not a full Richards-equation or measured FC/WP model.",
        "- The static-threshold baseline shows that a fixed normalized-storage coordinate alone is not a sufficient event-scale forecast representation.",
        "- It should not be used by itself as the main cross-setting interaction test, because the simpler threshold model is also weak in FAWN.",
        "- Use the existing static-recession comparison as the stronger conservative test of whether response-centric gains are larger under on-farm local variability.",
        "",
        "Caveats:",
        "",
        "- On-farm soil moisture remains in source units, so cross-setting RMSE is not directly comparable.",
        "- The response-library analog is response-memory evidence, not full on-farm CSR registration.",
        "- This static baseline intentionally excludes recent loss; use the existing static-recession comparison as a stronger dynamic-baseline sensitivity.",
    ]
    (OUT / "static_threshold_2x2_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs() -> None:
    ensure_out()

    fawn_train = pd.read_parquet(FAWN_TRAIN)
    fawn_test = pd.read_parquet(FAWN_TEST)
    fawn_pred, fawn_thresholds, fawn_coeffs = predict_static_threshold(
        fawn_train,
        fawn_test,
        local_cols=["site_id", "layer"],
        layer_col="layer",
        event_col="event_id",
        origin_col="origin_id",
        horizon_col="horizon_h",
        storage_col="s0_mm",
        target_col="target_mm",
        start_col="start_mm",
        pred_col="pred_mm",
        residual_col="residual_mm",
    )
    fawn_static_metrics = summarize_predictions(
        fawn_pred,
        setting="FAWN rainfall-only",
        horizon_col="horizon_h",
        target_col="target_mm",
        pred_col="pred_mm",
        residual_col="residual_mm",
        event_col="event_id",
        origin_col="origin_id",
        site_col="site_id",
        rmse_name="rmse_mm",
    )

    onfarm = pd.read_parquet(ONFARM_ORIGINS)
    onfarm_train = onfarm[onfarm["split"].eq("train")].copy()
    onfarm_test = onfarm[onfarm["split"].eq("test")].copy()
    onfarm_pred, onfarm_thresholds, onfarm_coeffs = predict_static_threshold(
        onfarm_train,
        onfarm_test,
        local_cols=["farm_name", "probe_id", "layer"],
        layer_col="layer",
        event_col="event_id",
        origin_col="origin_id",
        horizon_col="horizon_h",
        storage_col="s0_value",
        target_col="target_value",
        start_col="start_value",
        pred_col="pred_value",
        residual_col="residual_value",
    )
    onfarm_static_metrics = summarize_predictions(
        onfarm_pred,
        setting="On-farm managed",
        horizon_col="horizon_h",
        target_col="target_value",
        pred_col="pred_value",
        residual_col="residual_value",
        event_col="event_id",
        origin_col="origin_id",
        site_col="probe_id",
        rmse_name="rmse_value",
    )

    fawn_response = load_fawn_response_metrics()
    onfarm_response = load_onfarm_response_metrics("response_input_regime_rate")
    onfarm_library = load_onfarm_response_metrics("response_library_analog")

    main_summary = build_summary(
        fawn_static_metrics,
        fawn_response,
        onfarm_static_metrics,
        onfarm_response,
        "response_input_regime_rate",
    )
    main_delta = build_delta(main_summary)
    library_summary = build_summary(
        fawn_static_metrics,
        fawn_response,
        onfarm_static_metrics,
        onfarm_library,
        "response_library_analog",
    )
    library_delta = build_delta(library_summary)

    fawn_pred.to_parquet(SOURCE / "fawn_static_threshold_recession_predictions.parquet", index=False)
    fawn_thresholds.to_csv(SOURCE / "fawn_static_threshold_thresholds.csv", index=False)
    fawn_coeffs.to_csv(SOURCE / "fawn_static_threshold_coefficients.csv", index=False)
    fawn_static_metrics.to_csv(SOURCE / "fawn_static_threshold_metrics.csv", index=False)
    onfarm_pred.to_parquet(SOURCE / "onfarm_static_threshold_recession_predictions.parquet", index=False)
    onfarm_thresholds.to_csv(SOURCE / "onfarm_static_threshold_thresholds.csv", index=False)
    onfarm_coeffs.to_csv(SOURCE / "onfarm_static_threshold_coefficients.csv", index=False)
    onfarm_static_metrics.to_csv(SOURCE / "onfarm_static_threshold_metrics.csv", index=False)
    main_summary.to_csv(OUT / "SMDE_CSR_static_threshold_2x2_summary.csv", index=False)
    main_delta.to_csv(OUT / "SMDE_CSR_static_threshold_2x2_deltas.csv", index=False)
    library_summary.to_csv(OUT / "SMDE_CSR_static_threshold_response_library_sensitivity_summary.csv", index=False)
    library_delta.to_csv(OUT / "SMDE_CSR_static_threshold_response_library_sensitivity_deltas.csv", index=False)

    manifest = {
        "fawn_train_rows": int(len(fawn_train)),
        "fawn_test_rows": int(len(fawn_test)),
        "onfarm_train_rows": int(len(onfarm_train)),
        "onfarm_test_rows": int(len(onfarm_test)),
        "main_summary_rows": int(len(main_summary)),
        "main_delta_rows": int(len(main_delta)),
        "library_summary_rows": int(len(library_summary)),
        "library_delta_rows": int(len(library_delta)),
        "static_threshold_baseline": "empirical FC/WP-like normalized-storage recession",
        "status": "preliminary property-centric baseline comparison",
        "notes": [
            "Static-threshold baseline uses fixed training thresholds and normalized current storage.",
            "Static-threshold baseline excludes recent loss, regime labels, input-source class, and response libraries.",
            "Use with existing static_recession comparison to distinguish property-centric and stronger recent-loss baselines.",
        ],
    }
    (OUT / "static_threshold_2x2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(main_summary, main_delta, library_summary, library_delta)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    write_outputs()
