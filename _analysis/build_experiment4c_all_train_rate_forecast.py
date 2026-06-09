from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from build_experiment4_regime_conditioned_forecast import (
    FEATURE_COLS,
    HORIZONS_H,
    LAYER_ORDER,
    RANDOM_SEED,
    REGIME_ORDER,
    analog_predict,
    build_forecast_origin_table,
    build_registered_csr_library,
    concordance_correlation_coefficient,
    r2_score,
    regime_mixture_predict,
    registered_csr_predict,
)
from build_experiment4_2026_external_forecast_figure import (
    build_2026_external_points,
    build_external_forecast_origins,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
EXP3_SOURCE = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data"
AUDIT_SOURCE = ANALYSIS / "fawn_full_smde_audit"
OUT = ANALYSIS / "experiment4c_all_train_rate_forecast"
SOURCE = OUT / "source_data"

TRAIN_POINTS_FILE = EXP3_SOURCE / "adaptive_segment_points_all.parquet"
EVENT_AUDIT_FILE = AUDIT_SOURCE / "full_smde_event_audit.csv"

BASELINE_MODELS = ["persistence", "recent_slope"]
CSR_MODELS = ["registered_csr_operator", "nonregime_analog", "online_regime_analog", "regime_mixture_analog"]
RATE_MODELS = [
    "horizon_calibrated_rate",
    "regime_calibrated_rate",
    "layer_regime_calibrated_rate",
    "local_regime_shrink_rate",
]
MODEL_ORDER = [*BASELINE_MODELS, *CSR_MODELS, *RATE_MODELS]

MIN_GROUP_ROWS = 8
SHRINK_K_SITE = 150.0
SHRINK_K_LAYER = 250.0
ALPHA_MIN = 0.0
ALPHA_MAX = 1.5


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)


def md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame if max_rows is None else frame.head(max_rows)
    return show.to_markdown(index=False)


def fit_alpha(frame: pd.DataFrame) -> float:
    x = frame["recent_linear_loss"].to_numpy(float)
    y = frame["target_loss"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 1e-9)
    if ok.sum() < MIN_GROUP_ROWS:
        return np.nan
    alpha = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2))
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def prepare_train_points() -> pd.DataFrame:
    points = pd.read_parquet(TRAIN_POINTS_FILE)
    points = points[points["segment_regime"].astype(str).isin(REGIME_ORDER)].copy()
    event_meta = pd.read_csv(
        EVENT_AUDIT_FILE,
        usecols=["site_id", "event_id", "layer", "start", "rain_before_48", "rain_during", "start_month"],
        parse_dates=["start"],
    )
    event_meta["event_id"] = event_meta["event_id"].astype(str)
    event_meta["layer"] = event_meta["layer"].astype(str)
    points["event_id"] = points["event_id"].astype(str)
    points["layer"] = points["layer"].astype(str)
    points = points.merge(event_meta, on=["site_id", "layer", "event_id"], how="left", validate="many_to_one")
    if points["start"].isna().any():
        missing = int(points["start"].isna().sum())
        raise RuntimeError(f"Missing parent-event metadata for {missing:,} training segment points.")
    points["layer"] = pd.Categorical(points["layer"].astype(str), categories=LAYER_ORDER, ordered=True)
    return points.sort_values(["site_id", "layer", "event_id", "t_h"]).reset_index(drop=True)


def all_train_split(points: pd.DataFrame) -> pd.DataFrame:
    split = points[["site_id", "layer", "event_id"]].drop_duplicates().copy()
    split["split"] = "train"
    return split


def prepare_training_origins(points: pd.DataFrame) -> pd.DataFrame:
    split = all_train_split(points)
    origins = build_forecast_origin_table(points, split)
    origins = origins[origins["split"].astype(str).eq("train")].copy()
    origins["true_regime"] = origins["true_regime"].astype(str)
    origins["layer"] = origins["layer"].astype(str)
    origins["target_loss"] = origins["s0_mm"].astype(float) - origins["target_mm"].astype(float)
    origins["recent_linear_loss"] = origins["recent_loss_1h"].astype(float) * origins["horizon_h"].astype(float)
    return origins.reset_index(drop=True)


def fit_rate_coefficients(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_rows(group_cols: list[str], level: str) -> None:
        for keys, group in train.groupby(group_cols, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(group_cols, keys))
            rec.update({"level": level, "rows": int(len(group)), "alpha": fit_alpha(group)})
            rows.append(rec)

    add_rows(["horizon_h"], "horizon")
    add_rows(["horizon_h", "true_regime"], "horizon_regime")
    add_rows(["layer", "horizon_h"], "layer_horizon")
    add_rows(["layer", "horizon_h", "true_regime"], "layer_horizon_regime")
    add_rows(["site_id", "layer", "horizon_h"], "site_layer_horizon")
    add_rows(["site_id", "layer", "horizon_h", "true_regime"], "site_layer_horizon_regime")

    coeffs = pd.DataFrame(rows)
    return coeffs.sort_values(["level", "horizon_h", "layer", "site_id", "true_regime"], na_position="last").reset_index(drop=True)


def _lookup_table(coeffs: pd.DataFrame, level: str, keys: list[str]) -> dict[tuple[object, ...], tuple[float, int]]:
    table: dict[tuple[object, ...], tuple[float, int]] = {}
    use = coeffs[coeffs["level"].eq(level)].copy()
    for row in use.itertuples(index=False):
        alpha = float(getattr(row, "alpha"))
        if not np.isfinite(alpha):
            continue
        key = tuple(getattr(row, key) for key in keys)
        table[key] = (alpha, int(getattr(row, "rows")))
    return table


def coefficient_maps(coeffs: pd.DataFrame) -> dict[str, dict[tuple[object, ...], tuple[float, int]]]:
    return {
        "horizon": _lookup_table(coeffs, "horizon", ["horizon_h"]),
        "horizon_regime": _lookup_table(coeffs, "horizon_regime", ["horizon_h", "true_regime"]),
        "layer_horizon": _lookup_table(coeffs, "layer_horizon", ["layer", "horizon_h"]),
        "layer_horizon_regime": _lookup_table(coeffs, "layer_horizon_regime", ["layer", "horizon_h", "true_regime"]),
        "site_layer_horizon": _lookup_table(coeffs, "site_layer_horizon", ["site_id", "layer", "horizon_h"]),
        "site_layer_horizon_regime": _lookup_table(
            coeffs, "site_layer_horizon_regime", ["site_id", "layer", "horizon_h", "true_regime"]
        ),
    }


def get_alpha(table: dict[tuple[object, ...], tuple[float, int]], key: tuple[object, ...]) -> tuple[float, int]:
    return table.get(key, (np.nan, 0))


def hierarchical_alpha(row: pd.Series, maps: dict[str, dict[tuple[object, ...], tuple[float, int]]]) -> tuple[float, str]:
    site = int(row["site_id"])
    layer = str(row["layer"])
    horizon = int(row["horizon_h"])
    regime = str(row["predicted_regime"])

    base, base_n = get_alpha(maps["horizon"], (horizon,))
    if not np.isfinite(base):
        return np.nan, "missing"

    regime_alpha, regime_n = get_alpha(maps["horizon_regime"], (horizon, regime))
    parent = regime_alpha if np.isfinite(regime_alpha) else base
    parent_n = regime_n if np.isfinite(regime_alpha) else base_n

    layer_regime, layer_regime_n = get_alpha(maps["layer_horizon_regime"], (layer, horizon, regime))
    if np.isfinite(layer_regime):
        w = layer_regime_n / (layer_regime_n + SHRINK_K_LAYER)
        parent = w * layer_regime + (1.0 - w) * parent
        parent_n += layer_regime_n

    site_regime, site_regime_n = get_alpha(maps["site_layer_horizon_regime"], (site, layer, horizon, regime))
    if np.isfinite(site_regime):
        w = site_regime_n / (site_regime_n + SHRINK_K_SITE)
        alpha = w * site_regime + (1.0 - w) * parent
        return float(alpha), f"site-layer-regime shrink n={site_regime_n}"

    site_alpha, site_n = get_alpha(maps["site_layer_horizon"], (site, layer, horizon))
    if np.isfinite(site_alpha):
        w = site_n / (site_n + SHRINK_K_SITE)
        alpha = w * site_alpha + (1.0 - w) * parent
        return float(alpha), f"site-layer shrink n={site_n}"

    if parent_n > base_n:
        return float(parent), "layer/regime shrink"
    return float(base), "horizon fallback"


def predict_external_regimes_all_train(train: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    train_origins = train.drop_duplicates("origin_id").copy()
    external_origins = external.drop_duplicates("origin_id").copy()
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
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=15,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(x_all.loc[train_mask], feature_frame.loc[train_mask, "true_regime"].astype(str))
    pred = clf.predict(x_all.loc[~train_mask])
    prob = clf.predict_proba(x_all.loc[~train_mask])
    out = feature_frame.loc[~train_mask, ["origin_id"]].reset_index(drop=True)
    out["predicted_regime"] = pred
    out = pd.concat([out, pd.DataFrame(prob, columns=[f"prob_{cls}" for cls in clf.classes_])], axis=1)
    for regime in REGIME_ORDER:
        col = f"prob_{regime}"
        if col not in out.columns:
            out[col] = 0.0
    return out


def calibrated_rate_predictions(
    base: pd.DataFrame,
    coeffs: pd.DataFrame,
) -> pd.DataFrame:
    maps = coefficient_maps(coeffs)
    frames: list[pd.DataFrame] = []
    work = base.copy()
    work["recent_linear_loss"] = work["recent_loss_1h"].astype(float) * work["horizon_h"].astype(float)
    horizon = {int(k[0]): v[0] for k, v in maps["horizon"].items()}
    horizon_regime = {(int(k[0]), str(k[1])): v[0] for k, v in maps["horizon_regime"].items()}
    layer_regime = {(str(k[0]), int(k[1]), str(k[2])): v[0] for k, v in maps["layer_horizon_regime"].items()}

    for model in RATE_MODELS:
        out = work[["forecast_id", "s0_mm", "recent_linear_loss"]].copy()
        out["model"] = model
        out["analog_candidates"] = 0
        if model == "horizon_calibrated_rate":
            alpha = work["horizon_h"].astype(int).map(horizon).astype(float)
            level = "horizon damping"
        elif model == "regime_calibrated_rate":
            alpha = pd.Series(
                [horizon_regime.get((int(h), str(r)), horizon.get(int(h), np.nan)) for h, r in zip(work["horizon_h"], work["predicted_regime"])],
                index=work.index,
                dtype=float,
            )
            level = "horizon + online regime damping"
        elif model == "layer_regime_calibrated_rate":
            alpha = pd.Series(
                [
                    layer_regime.get((str(layer), int(h), str(r)), horizon_regime.get((int(h), str(r)), horizon.get(int(h), np.nan)))
                    for layer, h, r in zip(work["layer"], work["horizon_h"], work["predicted_regime"])
                ],
                index=work.index,
                dtype=float,
            )
            level = "layer + online regime damping"
        else:
            alpha_level = work.apply(lambda row: hierarchical_alpha(row, maps), axis=1)
            alpha = alpha_level.map(lambda x: x[0]).astype(float)
            level = alpha_level.map(lambda x: x[1]).astype(str)
        out["pred_mm"] = np.maximum(0.0, out["s0_mm"].astype(float) - alpha.to_numpy(float) * out["recent_linear_loss"].to_numpy(float))
        out.loc[~np.isfinite(alpha.to_numpy(float)), "pred_mm"] = np.nan
        out["q10_mm"] = np.nan
        out["q90_mm"] = np.nan
        out["analog_level"] = level
        frames.append(out[["forecast_id", "model", "pred_mm", "q10_mm", "q90_mm", "analog_level", "analog_candidates"]])
    return pd.concat(frames, ignore_index=True)


def build_external_predictions(
    train: pd.DataFrame,
    external: pd.DataFrame,
    regime_pred: pd.DataFrame,
    csr_library: pd.DataFrame,
    coeffs: pd.DataFrame,
) -> pd.DataFrame:
    test = external.merge(regime_pred, on="origin_id", how="left", validate="many_to_one").reset_index(drop=True)
    test["forecast_id"] = np.arange(len(test), dtype=int)
    train = train.reset_index(drop=True).copy()
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
    frames: list[pd.DataFrame] = []

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

    if csr_library is not None and not csr_library.empty:
        frames.append(registered_csr_predict(csr_library, test))
    frames.append(analog_predict(train, test, None, "nonregime_analog"))
    frames.append(analog_predict(train, test, "predicted_regime", "online_regime_analog"))
    frames.append(regime_mixture_predict(train, test))
    frames.append(calibrated_rate_predictions(base, coeffs))

    pred = pd.concat(frames, ignore_index=True).merge(base, on="forecast_id", how="left", validate="many_to_one")
    pred["layer"] = pd.Categorical(pred["layer"], categories=LAYER_ORDER, ordered=True)
    pred["model"] = pd.Categorical(pred["model"].astype(str), categories=MODEL_ORDER, ordered=True)
    pred["residual_mm"] = pred["pred_mm"] - pred["target_mm"]
    return pred.sort_values(["model", "site_id", "layer", "event_id", "origin_t_h", "horizon_h"]).reset_index(drop=True)


def metric_row(model: str, horizon: int, frame: pd.DataFrame, comparison_set: str) -> dict[str, object]:
    valid = frame.dropna(subset=["pred_mm", "target_mm"]).copy()
    obs = valid["target_mm"].to_numpy(float)
    pred = valid["pred_mm"].to_numpy(float)
    residual = pred - obs
    return {
        "comparison_set": comparison_set,
        "model": model,
        "horizon_h": int(horizon),
        "forecasts": int(len(valid)),
        "sites": int(valid["site_id"].nunique()) if len(valid) else 0,
        "events": int(valid["event_id"].nunique()) if len(valid) else 0,
        "origins": int(valid["origin_id"].nunique()) if len(valid) else 0,
        "rmse_mm": float(np.sqrt(np.nanmean(residual**2))) if len(valid) else np.nan,
        "mae_mm": float(np.nanmean(np.abs(residual))) if len(valid) else np.nan,
        "bias_mm": float(np.nanmean(residual)) if len(valid) else np.nan,
        "ccc": concordance_correlation_coefficient(obs, pred) if len(valid) else np.nan,
        "r2": r2_score(obs, pred) if len(valid) else np.nan,
    }


def common_origin_subset(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    valid = pred[pred["model"].astype(str).isin(models)].dropna(subset=["pred_mm", "target_mm"]).copy()
    counts = valid.groupby(["forecast_id", "horizon_h"], observed=True)["model"].nunique().reset_index(name="valid_models")
    ids = counts[counts["valid_models"].eq(len(models))][["forecast_id", "horizon_h"]]
    return pred[pred["model"].astype(str).isin(models)].merge(ids, on=["forecast_id", "horizon_h"], how="inner")


def summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    comparison_defs = {
        "model_valid": MODEL_ORDER,
        "core_common_origin": ["persistence", "recent_slope", "registered_csr_operator", *RATE_MODELS],
        "all_common_origin": MODEL_ORDER,
    }
    for comparison_set, models in comparison_defs.items():
        use = pred[pred["model"].astype(str).isin(models)].copy()
        if comparison_set.endswith("common_origin"):
            use = common_origin_subset(pred, models)
        for (model, horizon), group in use.groupby(["model", "horizon_h"], observed=True):
            rows.append(metric_row(str(model), int(horizon), group, comparison_set))
    metrics = pd.DataFrame(rows).sort_values(["comparison_set", "horizon_h", "model"]).reset_index(drop=True)

    total = pred[["forecast_id", "horizon_h"]].drop_duplicates()
    total_by_h = total.groupby("horizon_h", observed=True)["forecast_id"].nunique().to_dict()
    cov_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        valid = pred[pred["model"].astype(str).eq(model)].dropna(subset=["pred_mm", "target_mm"])
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
    return metrics, coverage


def write_report(
    train_points: pd.DataFrame,
    train_origins: pd.DataFrame,
    events_2026: pd.DataFrame,
    points_2026: pd.DataFrame,
    origins_2026: pd.DataFrame,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    coeffs: pd.DataFrame,
) -> None:
    core = metrics[metrics["comparison_set"].eq("core_common_origin")].copy()
    all_common = metrics[metrics["comparison_set"].eq("all_common_origin")].copy()
    best = (
        core.dropna(subset=["rmse_mm"])
        .sort_values(["horizon_h", "rmse_mm"])
        .groupby("horizon_h", observed=True)
        .head(1)
        [["horizon_h", "model", "rmse_mm", "mae_mm", "bias_mm", "ccc", "forecasts"]]
    )
    report = f"""# Experiment 4c: all-2023-2025 training and 2026 external validation

## Material Passport

- Type: code experiment result
- Status: completed
- Training data: all interpretable 2023-2025 SMDE segments from the updated adaptive-regime library
- Validation data: 2026 SMDEs only
- No 2023-2025 train/test split is used for the final training set.

## Design

This run addresses the weakness found in the registered CSR forecast operator. The previous operator advanced along a hydrologically registered CSR coordinate as though it were physical forecast time. Here, the registered CSR and analog models are retained as comparison models, but new forecast operators use the observed recent loss rate at the forecast origin:

```text
S_hat(t+h) = S(t) - alpha * L_recent(t) * h
```

The damping coefficient `alpha` is learned from all 2023-2025 training origins. Variants estimate `alpha` by horizon, by horizon and online regime, by layer-regime, or through local station-layer-regime shrinkage.

## Sample

- 2023-2025 training segment points: {len(train_points):,}
- 2023-2025 training forecast rows: {len(train_origins):,}
- 2023-2025 training events: {train_origins["event_id"].nunique():,}
- 2026 detected SMDEs: {len(events_2026):,}
- 2026 detected SMDE points: {len(points_2026):,}
- 2026 validation forecast rows: {len(origins_2026):,}

## Best Core Model by Horizon

{md_table(best)}

## Core Common-Origin Metrics

{md_table(core)}

## All-Model Common-Origin Metrics

{md_table(all_common)}

## Coverage

{md_table(coverage, max_rows=80)}

## Calibration Coefficients

{md_table(coeffs.head(120))}
"""
    (OUT / "experiment4c_all_train_rate_forecast_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_out()
    train_points = prepare_train_points()
    split = all_train_split(train_points)
    csr_library, csr_edges, csr_components = build_registered_csr_library(train_points, split)
    train_origins = prepare_training_origins(train_points)
    coeffs = fit_rate_coefficients(train_origins)

    events_2026, points_2026, status_2026 = build_2026_external_points()
    origins_2026 = build_external_forecast_origins(points_2026)
    if origins_2026.empty:
        raise RuntimeError("No 2026 external forecast origins were created.")
    regime_pred = predict_external_regimes_all_train(train_origins, origins_2026)
    pred = build_external_predictions(train_origins, origins_2026, regime_pred, csr_library, coeffs)
    metrics, coverage = summarize_predictions(pred)

    train_points.to_parquet(SOURCE / "all_2023_2025_training_segment_points.parquet", index=False)
    train_origins.to_parquet(SOURCE / "all_2023_2025_training_forecast_origins.parquet", index=False)
    csr_library.to_csv(SOURCE / "all_train_registered_csr_forecast_library.csv", index=False)
    csr_edges.to_csv(SOURCE / "all_train_registered_csr_registration_edges.csv", index=False)
    csr_components.to_csv(SOURCE / "all_train_registered_csr_registration_components.csv", index=False)
    coeffs.to_csv(SOURCE / "all_train_rate_calibration_coefficients.csv", index=False)
    events_2026.to_csv(SOURCE / "external_2026_detected_smde_events.csv", index=False)
    points_2026.to_parquet(SOURCE / "external_2026_detected_smde_points.parquet", index=False)
    status_2026.to_csv(SOURCE / "external_2026_processing_status.csv", index=False)
    origins_2026.to_parquet(SOURCE / "external_2026_forecast_origins.parquet", index=False)
    regime_pred.to_csv(SOURCE / "external_2026_online_regime_predictions_all_train.csv", index=False)
    pred.to_parquet(SOURCE / "external_2026_all_train_forecast_predictions.parquet", index=False)
    metrics.to_csv(SOURCE / "external_2026_all_train_forecast_metrics.csv", index=False)
    coverage.to_csv(SOURCE / "external_2026_all_train_forecast_coverage.csv", index=False)
    write_report(train_points, train_origins, events_2026, points_2026, origins_2026, metrics, coverage, coeffs)

    core = metrics[metrics["comparison_set"].eq("core_common_origin")]
    print(core.pivot(index="horizon_h", columns="model", values="rmse_mm").round(4).to_string())
    print(OUT / "experiment4c_all_train_rate_forecast_report.md")


if __name__ == "__main__":
    main()
