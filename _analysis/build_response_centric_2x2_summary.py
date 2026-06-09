from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_onfarm_response_forecast_models import (
    ALPHA_MAX,
    ALPHA_MIN,
    MIN_GROUP_ROWS,
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
ONFARM_METRICS = ONFARM_SOURCE / "onfarm_response_forecast_metrics.csv"

STATIC_SHRINK_K = 150.0


def ensure_out() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)


def fit_alpha(frame: pd.DataFrame) -> float:
    x = frame["recent_linear_loss"].to_numpy(float)
    y = frame["target_loss"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 1e-9)
    if int(ok.sum()) < MIN_GROUP_ROWS:
        return np.nan
    alpha = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2))
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def fit_fawn_static_coefficients(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(level: str, cols: list[str]) -> None:
        for keys, group in train.groupby(cols, observed=True, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(cols, keys))
            rec.update({"level": level, "rows": int(len(group)), "alpha": fit_alpha(group)})
            rows.append(rec)

    add("horizon", ["horizon_h"])
    add("layer_horizon", ["layer", "horizon_h"])
    add("site_layer_horizon", ["site_id", "layer", "horizon_h"])
    return pd.DataFrame(rows).sort_values(["level", "horizon_h", "site_id", "layer"], na_position="last")


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


def fawn_static_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work_train = train.copy()
    if "target_loss" not in work_train.columns:
        work_train["target_loss"] = work_train["s0_mm"].astype(float) - work_train["target_mm"].astype(float)
    if "recent_linear_loss" not in work_train.columns:
        work_train["recent_linear_loss"] = work_train["recent_loss_1h"].astype(float) * work_train["horizon_h"].astype(float)
    coeffs = fit_fawn_static_coefficients(work_train)
    maps = {
        "horizon": table_for(coeffs, "horizon", ["horizon_h"]),
        "layer_horizon": table_for(coeffs, "layer_horizon", ["layer", "horizon_h"]),
        "site_layer_horizon": table_for(coeffs, "site_layer_horizon", ["site_id", "layer", "horizon_h"]),
    }

    def alpha_for(row: pd.Series) -> tuple[float, str]:
        site = int(row["site_id"])
        layer = str(row["layer"])
        horizon = int(row["horizon_h"])
        base, base_n = maps["horizon"].get((horizon,), (np.nan, 0))
        if not np.isfinite(base):
            return np.nan, "missing"
        layer_alpha, layer_n = maps["layer_horizon"].get((layer, horizon), (np.nan, 0))
        parent = layer_alpha if np.isfinite(layer_alpha) else base
        parent_n = layer_n if np.isfinite(layer_alpha) else base_n
        local_alpha, local_n = maps["site_layer_horizon"].get((site, layer, horizon), (np.nan, 0))
        if np.isfinite(local_alpha):
            w = local_n / (local_n + STATIC_SHRINK_K)
            return float(w * local_alpha + (1.0 - w) * parent), f"site-layer shrink n={local_n}"
        if parent_n > base_n:
            return float(parent), "layer fallback"
        return float(base), "horizon fallback"

    pred = test.copy()
    pred["recent_linear_loss"] = pred["recent_loss_1h"].astype(float) * pred["horizon_h"].astype(float)
    alpha_level = pred.apply(alpha_for, axis=1)
    alpha = alpha_level.map(lambda x: x[0]).astype(float)
    pred["model"] = "static_recession"
    pred["model_level"] = alpha_level.map(lambda x: x[1]).astype(str)
    pred["pred_mm"] = pred["s0_mm"].astype(float) - alpha.to_numpy(float) * pred["recent_linear_loss"].to_numpy(float)
    pred.loc[~np.isfinite(alpha.to_numpy(float)), "pred_mm"] = np.nan
    pred["residual_mm"] = pred["pred_mm"] - pred["target_mm"]
    metrics = summarize_fawn_static(pred)
    return pred, coeffs, metrics


def summarize_fawn_static(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in pred.groupby("horizon_h", observed=True):
        valid = group.dropna(subset=["pred_mm", "target_mm"]).copy()
        obs = valid["target_mm"].to_numpy(float)
        fit = valid["pred_mm"].to_numpy(float)
        residual = fit - obs
        rows.append(
            {
                "comparison_set": "model_valid",
                "model": "static_recession",
                "horizon_h": int(horizon),
                "forecasts": int(len(valid)),
                "sites": int(valid["site_id"].nunique()) if len(valid) else 0,
                "events": int(valid["event_id"].nunique()) if len(valid) else 0,
                "origins": int(valid["origin_id"].nunique()) if len(valid) else 0,
                "rmse_mm": float(np.sqrt(np.nanmean(residual**2))) if len(valid) else np.nan,
                "mae_mm": float(np.nanmean(np.abs(residual))) if len(valid) else np.nan,
                "bias_mm": float(np.nanmean(residual)) if len(valid) else np.nan,
                "ccc": concordance_correlation_coefficient(obs, fit) if len(valid) else np.nan,
                "r2": r2_score(obs, fit) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_h").reset_index(drop=True)


def load_fawn_response_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(FAWN_METRICS)
    use = metrics[
        metrics["comparison_set"].eq("model_valid")
        & metrics["model"].eq("local_regime_shrink_rate")
    ].copy()
    return use[
        ["comparison_set", "model", "horizon_h", "forecasts", "sites", "events", "origins", "rmse_mm", "mae_mm", "bias_mm", "ccc", "r2"]
    ]


def load_onfarm_metrics(response_model: str = "response_input_regime_rate") -> pd.DataFrame:
    metrics = pd.read_csv(ONFARM_METRICS)
    use = metrics[
        metrics["comparison_set"].eq("static_vs_response_common")
        & metrics["model"].isin(["static_recession", response_model])
    ].copy()
    if response_model != "response_input_regime_rate":
        extra = metrics[
            metrics["comparison_set"].eq("model_valid")
            & metrics["model"].isin(["static_recession", response_model])
        ].copy()
        use = extra
    return use


def build_2x2_summary(
    fawn_static: pd.DataFrame,
    fawn_response: pd.DataFrame,
    onfarm: pd.DataFrame,
    onfarm_response_model: str = "response_input_regime_rate",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame, setting, model_family, model_name, unit_label in [
        (fawn_static, "FAWN rainfall-only", "static_or_less_local", "static_recession", "mm"),
        (fawn_response, "FAWN rainfall-only", "response_centric", "local_regime_shrink_rate", "mm"),
    ]:
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "data_setting": setting,
                    "model_family": model_family,
                    "model_name": model_name,
                    "horizon_h": int(row.horizon_h),
                    "forecasts": int(row.forecasts),
                    "events": int(row.events),
                    "sites_or_probes": int(getattr(row, "sites", 0)),
                    "coverage": 1.0,
                    "r2": float(row.r2),
                    "ccc": float(row.ccc),
                    "rmse": float(row.rmse_mm),
                    "mae": float(row.mae_mm),
                    "bias": float(row.bias_mm),
                    "unit_label": unit_label,
                    "status": "current FAWN external-validation result",
                }
            )

    for row in onfarm.itertuples(index=False):
        model = str(row.model)
        family = "response_centric" if model == onfarm_response_model else "static_or_less_local"
        rows.append(
            {
                "data_setting": "On-farm managed",
                "model_family": family,
                "model_name": model,
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
                "status": "first-pass on-farm exploratory result",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["horizon_h", "data_setting", "model_family"]).reset_index(drop=True)


def build_delta(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (setting, horizon), group in summary.groupby(["data_setting", "horizon_h"], sort=False):
        wide = group.pivot_table(index="horizon_h", columns="model_family", values=["r2", "ccc", "rmse"], aggfunc="first")
        if wide.empty or "response_centric" not in wide.columns.get_level_values(1) or "static_or_less_local" not in wide.columns.get_level_values(1):
            continue
        rows.append(
            {
                "data_setting": setting,
                "horizon_h": int(horizon),
                "delta_r2_response_minus_static": float(wide[("r2", "response_centric")].iloc[0] - wide[("r2", "static_or_less_local")].iloc[0]),
                "delta_ccc_response_minus_static": float(wide[("ccc", "response_centric")].iloc[0] - wide[("ccc", "static_or_less_local")].iloc[0]),
                "rmse_ratio_response_over_static": float(wide[("rmse", "response_centric")].iloc[0] / wide[("rmse", "static_or_less_local")].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_h", "data_setting"]).reset_index(drop=True)


def write_report(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    library_summary: pd.DataFrame,
    library_delta: pd.DataFrame,
) -> None:
    lines = [
        "# Response-centric 2 x 2 preliminary summary",
        "",
        "Material Passport:",
        "",
        "- Type: preliminary cross-setting model comparison",
        "- Status: first-pass evidence table, not final manuscript table",
        "- FAWN response model: `local_regime_shrink_rate` from current 2026 external validation.",
        "- FAWN static baseline: newly fitted site-layer-horizon static recession baseline.",
        "- On-farm response model: first-pass input/regime-aware recent-loss model.",
        "- On-farm response-library sensitivity: nearest-neighbor response-library analog model.",
        "- On-farm static baseline: first-pass farm/probe/layer/horizon static recession baseline.",
        "",
        "Primary 2 x 2 summary:",
        "",
        summary[["data_setting", "model_family", "model_name", "horizon_h", "forecasts", "events", "r2", "ccc", "rmse", "unit_label"]].to_markdown(index=False),
        "",
        "Response-minus-static deltas:",
        "",
        delta.to_markdown(index=False),
        "",
        "Response-library sensitivity:",
        "",
        "This sensitivity replaces the on-farm input/regime-aware recent-loss model with `response_library_analog`, a nearest-neighbor response-memory model. It is closer to the manuscript's response-library argument, but it is still not full CSR registration.",
        "",
        library_summary[["data_setting", "model_family", "model_name", "horizon_h", "forecasts", "events", "r2", "ccc", "rmse", "unit_label"]].to_markdown(index=False),
        "",
        "Response-library response-minus-static deltas:",
        "",
        library_delta.to_markdown(index=False),
        "",
        "Reading guide:",
        "",
        "- Positive `delta_r2_response_minus_static` supports the response-centric claim.",
        "- Positive `delta_ccc_response_minus_static` supports the response-centric claim.",
        "- `rmse_ratio_response_over_static < 1` supports the response-centric claim.",
        "",
        "Caveats:",
        "",
        "- FAWN and on-farm RMSE units are not yet directly comparable; use R2 and CCC for cross-setting comparison.",
        "- On-farm results are first-pass and use source soil-moisture units.",
        "- On-farm chronological test support is rain-only dominated; irrigation-only support exists but mixed-input test support is sparse.",
        "- The main on-farm response model is regime/input-aware recent-loss calibration, not full CSR registration yet.",
        "- The response-library analog sensitivity is response-memory evidence, not proof that the full on-farm CSR registration has been implemented.",
    ]
    (OUT / "response_centric_2x2_preliminary_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs() -> None:
    ensure_out()
    train = pd.read_parquet(FAWN_TRAIN)
    test = pd.read_parquet(FAWN_TEST)
    fawn_static_pred, fawn_static_coeffs, fawn_static_metrics = fawn_static_predictions(train, test)
    fawn_response = load_fawn_response_metrics()
    onfarm = load_onfarm_metrics("response_input_regime_rate")
    summary = build_2x2_summary(fawn_static_metrics, fawn_response, onfarm, "response_input_regime_rate")
    delta = build_delta(summary)
    onfarm_library = load_onfarm_metrics("response_library_analog")
    library_summary = build_2x2_summary(
        fawn_static_metrics,
        fawn_response,
        onfarm_library,
        "response_library_analog",
    )
    library_delta = build_delta(library_summary)

    fawn_static_pred.to_parquet(SOURCE / "fawn_static_recession_predictions.parquet", index=False)
    fawn_static_coeffs.to_csv(SOURCE / "fawn_static_recession_coefficients.csv", index=False)
    fawn_static_metrics.to_csv(SOURCE / "fawn_static_recession_metrics.csv", index=False)
    summary.to_csv(OUT / "SMDE_CSR_response_centric_2x2_preliminary_summary.csv", index=False)
    delta.to_csv(OUT / "SMDE_CSR_response_centric_2x2_preliminary_deltas.csv", index=False)
    library_summary.to_csv(OUT / "SMDE_CSR_response_library_2x2_sensitivity_summary.csv", index=False)
    library_delta.to_csv(OUT / "SMDE_CSR_response_library_2x2_sensitivity_deltas.csv", index=False)
    manifest = {
        "fawn_train_rows": int(len(train)),
        "fawn_test_rows": int(len(test)),
        "fawn_static_prediction_rows": int(len(fawn_static_pred)),
        "summary_rows": int(len(summary)),
        "delta_rows": int(len(delta)),
        "response_library_sensitivity_rows": int(len(library_summary)),
        "response_library_sensitivity_delta_rows": int(len(library_delta)),
        "status": "preliminary first-pass 2 x 2 comparison",
        "notes": [
            "FAWN static baseline was fitted with the same recent-loss damping form as the on-farm static baseline.",
            "FAWN response model uses existing local_regime_shrink_rate external-validation metrics.",
            "Main on-farm response model is input/regime-aware recent-loss calibration; full CSR registration is not yet included.",
            "Response-library sensitivity uses response_library_analog as response-memory evidence closer to the CSR-library argument.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(summary, delta, library_summary, library_delta)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    write_outputs()
