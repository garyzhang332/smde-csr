from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
ALACHUA_WITH_RAIN = Path(os.environ.get("ALACHUA_WITH_RAIN", ROOT / "data" / "examples" / "Alachua_with_rain.csv"))
OUT_DIR = Path(os.environ.get("SMDE_REGIME_AUDIT_OUT_DIR", ROOT / "_analysis" / "smde_regime_audit"))

DEFAULT_LAYER_THICKNESS_INCHES = 6.0
PCT_POINT_TO_MM = DEFAULT_LAYER_THICKNESS_INCHES * 25.4 / 100.0
ASSOCIATION_RAIN_THRESHOLD_IN = 0.0
INTERRUPTION_RAIN_THRESHOLD_IN = 0.02


@dataclass(frozen=True)
class DetectionConfig:
    min_steps: int = 3
    smooth_window_length: int = 49
    smooth_poly_order: int = 3
    noise_tolerance_mm: float = 0.001 * PCT_POINT_TO_MM
    min_total_decrease_mm: float = 0.5 * PCT_POINT_TO_MM
    max_total_decrease_mm: float = 15.0 * PCT_POINT_TO_MM
    max_val_low_thr_mm: float = 1.5 * PCT_POINT_TO_MM
    min_val_thr_mm: float = 2.0 * PCT_POINT_TO_MM
    max_internal_range_mm: float = 15.0 * PCT_POINT_TO_MM
    std_dev_factor_for_upper_limit: float = 2.0
    outlier_window_size: int = 10
    outlier_threshold_mm: float = 5.0 * PCT_POINT_TO_MM
    apply_outlier_filter_in_detection: bool = False
    min_drop_rate_mm_per_step: float | None = None
    snap_start_to_peak: bool = False
    peak_search_before_h: float = 2.0
    peak_search_after_h: float = 0.5
    peak_min_gain_mm: float = 0.0
    peak_recession_break_mm: float = 0.05 * PCT_POINT_TO_MM
    peak_max_rewetting_fraction: float = 0.35
    peak_max_rewetting_mm: float = 0.25 * PCT_POINT_TO_MM


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_alachua() -> pd.DataFrame:
    df = pd.read_csv(ALACHUA_WITH_RAIN)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    df = df.drop_duplicates(subset=["DateTime"], keep="last")
    df["Rain"] = pd.to_numeric(df["Rain"], errors="coerce").fillna(0.0)

    moisture_cols = [c for c in df.columns if c.startswith("moisture_")]
    for col in moisture_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") * PCT_POINT_TO_MM
    return df.set_index("DateTime")


def outlier_detection(values: np.ndarray, threshold: float, window_size: int) -> bool:
    if len(values) < window_size:
        return False
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        if np.nanmax(window) - np.nanmin(window) > threshold:
            return True
    return False


def smooth_for_detection(series: pd.Series, cfg: DetectionConfig) -> pd.Series:
    series = series.dropna()
    if series.empty:
        return series

    window = int(cfg.smooth_window_length)
    poly_order = int(cfg.smooth_poly_order)
    if window <= 1 or len(series) <= poly_order + 1:
        return series

    window = min(window, len(series))
    if window % 2 == 0:
        window -= 1
    if window <= poly_order:
        return series

    smoothed = savgol_filter(series.to_numpy(dtype=float), window, poly_order, mode="nearest")
    return pd.Series(smoothed, index=series.index, name=series.name)


def detect_decreasing_events(
    series: pd.Series, cfg: DetectionConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    series = smooth_for_detection(series, cfg)
    if series.empty or len(series) < 2:
        return []

    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    count = 0
    start_timestamp: pd.Timestamp | None = None
    current_values: list[float] = []
    dynamic_upper_limit = float(series.mean() + cfg.std_dev_factor_for_upper_limit * series.std())

    for i in range(1, len(series)):
        if series.iloc[i] <= series.iloc[i - 1] + cfg.noise_tolerance_mm:
            count += 1
            if count == 1:
                start_timestamp = series.index[i - 1]
                current_values = [float(series.iloc[i - 1]), float(series.iloc[i])]
            else:
                current_values.append(float(series.iloc[i]))
        else:
            maybe_append_event(events, series.index[i - 1], start_timestamp, count, current_values, dynamic_upper_limit, cfg)
            count = 0
            start_timestamp = None
            current_values = []

    maybe_append_event(events, series.index[-1], start_timestamp, count, current_values, dynamic_upper_limit, cfg)
    return events


def snap_start_to_local_peak(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    previous_end: pd.Timestamp | None,
    cfg: DetectionConfig,
) -> tuple[pd.Timestamp, dict[str, object]]:
    original_start = pd.Timestamp(start)
    rec = {
        "detected_start": original_start,
        "peak_start": original_start,
        "peak_start_shift_h": 0.0,
        "peak_start_gain_mm": 0.0,
        "peak_connected_start": original_start,
        "peak_boundary_limited": False,
    }
    if not cfg.snap_start_to_peak:
        return original_start, rec

    raw = series.dropna()
    if raw.empty:
        return original_start, rec

    nominal_search_start = original_start - pd.Timedelta(hours=float(cfg.peak_search_before_h))
    search_start = nominal_search_start
    if previous_end is not None:
        search_start = max(search_start, pd.Timestamp(previous_end) + pd.Timedelta(microseconds=1))
    search_end = min(pd.Timestamp(end), original_start + pd.Timedelta(hours=float(cfg.peak_search_after_h)))
    window = raw.loc[search_start:search_end].dropna()
    if window.empty:
        return original_start, rec

    try:
        start_value = float(raw.loc[original_start])
    except KeyError:
        prior = raw.loc[:original_start]
        start_value = float(prior.iloc[-1]) if not prior.empty else float(window.iloc[-1])

    peak_time: pd.Timestamp | None = None
    peak_value = np.nan
    peak_rewetting = np.nan
    boundary_limited_seen = False
    ranked_candidates = window.sort_values(ascending=False)
    for candidate_time, candidate_value in ranked_candidates.items():
        candidate_time = pd.Timestamp(candidate_time)
        boundary_limited = (
            candidate_time == pd.Timestamp(window.index[0])
            and pd.Timestamp(window.index[0]) <= nominal_search_start + pd.Timedelta(seconds=1)
            and len(window) > 1
        )
        if boundary_limited:
            boundary_limited_seen = True
            continue

        gain_candidate = float(candidate_value) - start_value
        if gain_candidate < float(cfg.peak_min_gain_mm) - float(cfg.noise_tolerance_mm):
            continue

        if candidate_time <= original_start:
            path = raw.loc[candidate_time:original_start].dropna()
        else:
            path = raw.loc[original_start:candidate_time].dropna()
        if len(path) > 1 and candidate_time <= original_start:
            running_min = path.cummin()
            rewetting = float((path - running_min).max())
        else:
            rewetting = 0.0
        allowed_rewetting = max(
            float(cfg.peak_max_rewetting_mm),
            float(cfg.peak_max_rewetting_fraction) * max(gain_candidate, 0.0),
            float(cfg.peak_recession_break_mm),
        )
        if rewetting <= allowed_rewetting + float(cfg.noise_tolerance_mm):
            peak_time = candidate_time
            peak_value = float(candidate_value)
            peak_rewetting = rewetting
            break

    if peak_time is None:
        rec["peak_boundary_limited"] = bool(boundary_limited_seen)
        return original_start, rec

    gain = peak_value - start_value
    if gain < float(cfg.peak_min_gain_mm) - float(cfg.noise_tolerance_mm):
        return original_start, rec

    rec.update(
        {
            "peak_start": peak_time,
            "peak_start_shift_h": float((original_start - peak_time).total_seconds() / 3600.0),
            "peak_start_gain_mm": float(max(gain, 0.0)),
            "peak_connected_start": peak_time,
            "peak_boundary_limited": False,
            "peak_path_rewetting_mm": float(peak_rewetting),
        }
    )
    return peak_time, rec


def maybe_append_event(
    events: list[tuple[pd.Timestamp, pd.Timestamp]],
    end_timestamp: pd.Timestamp,
    start_timestamp: pd.Timestamp | None,
    count: int,
    current_values: list[float],
    dynamic_upper_limit: float,
    cfg: DetectionConfig,
) -> None:
    if count < cfg.min_steps or start_timestamp is None or len(current_values) < cfg.min_steps + 1:
        return

    values = np.array(current_values, dtype=float)
    total_decrease = float(values[0] - values[-1])
    duration_steps = max(len(values) - 1, 1)
    drop_rate = total_decrease / duration_steps

    if cfg.apply_outlier_filter_in_detection and outlier_detection(values, cfg.outlier_threshold_mm, cfg.outlier_window_size):
        return
    if (
        cfg.min_drop_rate_mm_per_step is not None
        and cfg.min_drop_rate_mm_per_step > 0
        and drop_rate < cfg.min_drop_rate_mm_per_step
    ):
        return
    if not (cfg.min_total_decrease_mm <= total_decrease <= cfg.max_total_decrease_mm):
        return
    if (values <= cfg.max_val_low_thr_mm).any():
        return
    if values.min() <= cfg.min_val_thr_mm:
        return
    if (values.max() - values.min()) > cfg.max_internal_range_mm:
        return
    if values.max() > dynamic_upper_limit:
        return

    events.append((pd.Timestamp(start_timestamp), pd.Timestamp(end_timestamp)))


def cumulative_rain(rain: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end < start:
        return 0.0
    window = rain.loc[start:end]
    if window.empty:
        return 0.0
    return float(window.sum())


def time_since_last_rain_hours(rain: pd.Series, start: pd.Timestamp, max_lookback_h: int = 96) -> float:
    begin = start - pd.Timedelta(hours=max_lookback_h)
    prior = rain.loc[begin:start]
    prior = prior[prior > 0]
    if prior.empty:
        return np.nan
    return float((start - prior.index.max()).total_seconds() / 3600.0)


def exp_model(t: np.ndarray, a: float, tau_h: float, c: float) -> np.ndarray:
    return c + a * np.exp(-t / tau_h)


def fit_exponential(t_h: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(t_h) & np.isfinite(y)
    t = np.asarray(t_h[ok], dtype=float)
    yy = np.asarray(y[ok], dtype=float)
    if len(yy) < 6 or np.nanmax(yy) <= np.nanmin(yy):
        return {"r2": np.nan, "tau_h": np.nan, "n": float(len(yy))}

    t = t - t[0]
    ymin = float(np.nanmin(yy))
    ymax = float(np.nanmax(yy))
    yrange = ymax - ymin
    p0 = [max(yrange, 0.01), max((t[-1] - t[0]) / 2.0, 0.5), max(ymin - 0.05 * yrange, 0.0)]
    lower = [0.0, 0.05, 0.0]
    upper = [max(yrange * 3.0, 1.0), 24.0 * 60.0, ymin]

    try:
        params, _ = curve_fit(exp_model, t, yy, p0=p0, bounds=(lower, upper), maxfev=20000)
        pred = exp_model(t, *params)
        ss_res = float(np.sum((yy - pred) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"r2": r2, "tau_h": float(params[1]), "n": float(len(yy))}
    except Exception:
        return {"r2": np.nan, "tau_h": np.nan, "n": float(len(yy))}


def event_loss_points(
    times: pd.DatetimeIndex, values: np.ndarray, event_id: str, layer: str
) -> pd.DataFrame:
    if len(values) < 3:
        return pd.DataFrame()
    dt_h = np.diff(times.view("int64")) / 1e9 / 3600.0
    dy = np.diff(values)
    ok = dt_h > 0
    if not ok.any():
        return pd.DataFrame()

    y_mid = (values[:-1] + values[1:]) / 2.0
    loss = -dy / dt_h
    elapsed_mid_h = ((times[:-1] - times[0]).total_seconds() / 3600.0) + dt_h / 2.0
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    denom = ymax - ymin
    if denom <= 0:
        x_norm = np.full_like(y_mid, np.nan, dtype=float)
    else:
        x_norm = (y_mid - ymin) / denom

    return pd.DataFrame(
        {
            "event_id": event_id,
            "layer": layer,
            "elapsed_mid_h": elapsed_mid_h[ok],
            "storage_norm": x_norm[ok],
            "loss_mm_h": loss[ok],
        }
    )


def slope_and_corr(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if len(xx) < 5 or np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan, np.nan
    slope = float(np.polyfit(xx, yy, 1)[0])
    corr = float(np.corrcoef(xx, yy)[0, 1])
    return slope, corr


def classify_regime_proxy(row: pd.Series) -> str:
    if row["early3_drop_share"] >= 0.4:
        return "early-transient-heavy"
    if row["trim3_r2"] >= 0.7 and row["post3_loss_storage_corr"] >= 0.2:
        return "stage-II-like"
    if abs(row["post3_loss_storage_corr"]) < 0.2 and row["post3_loss_cv"] < 0.6:
        return "stage-I-like"
    return "mixed_or_uncertain"


def build_event_tables(df: pd.DataFrame, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rain = df["Rain"].fillna(0.0)
    layers = [c for c in df.columns if c.startswith("moisture_")]
    event_records: list[dict[str, object]] = []
    loss_frames: list[pd.DataFrame] = []

    for layer in layers:
        events = detect_decreasing_events(df[layer], cfg)
        previous_adjusted_end: pd.Timestamp | None = None
        for idx, (start, end) in enumerate(events, start=1):
            original_start = pd.Timestamp(start)
            start, peak_rec = snap_start_to_local_peak(df[layer], start, end, previous_adjusted_end, cfg)
            segment = df.loc[start:end, [layer, "Rain"]].dropna(subset=[layer])
            if len(segment) < cfg.min_steps + 1:
                previous_adjusted_end = pd.Timestamp(end)
                continue
            y = segment[layer].to_numpy(dtype=float)
            times = segment.index
            duration_h = float((times[-1] - times[0]).total_seconds() / 3600.0)
            if duration_h <= 0:
                previous_adjusted_end = pd.Timestamp(end)
                continue

            total_drop = float(y[0] - y[-1])
            if total_drop <= 0:
                previous_adjusted_end = pd.Timestamp(end)
                continue
            event_id = f"{layer}_{idx:04d}"
            rain_before_12 = cumulative_rain(rain, start - pd.Timedelta(hours=12), start)
            rain_before_24 = cumulative_rain(rain, start - pd.Timedelta(hours=24), start)
            rain_before_48 = cumulative_rain(rain, start - pd.Timedelta(hours=48), start)
            rain_during = cumulative_rain(rain, start + pd.Timedelta(minutes=1), end)
            lag_h = time_since_last_rain_hours(rain, start, 96)

            t_h = (times - times[0]).total_seconds().to_numpy() / 3600.0
            fit_full = fit_exponential(t_h, y)
            fit_1 = fit_exponential(t_h[t_h >= 1.0], y[t_h >= 1.0])
            fit_3 = fit_exponential(t_h[t_h >= 3.0], y[t_h >= 3.0])
            fit_6 = fit_exponential(t_h[t_h >= 6.0], y[t_h >= 6.0])

            early3_mask = t_h <= min(3.0, t_h[-1])
            early3_drop = float(y[0] - y[early3_mask][-1]) if early3_mask.any() else np.nan
            early3_drop_share = early3_drop / total_drop if total_drop > 0 else np.nan

            loss_df = event_loss_points(times, y, event_id, layer)
            if not loss_df.empty:
                loss_frames.append(loss_df)
            post3 = loss_df[loss_df["elapsed_mid_h"] >= 3.0] if not loss_df.empty else pd.DataFrame()
            slope, corr = slope_and_corr(post3.get("storage_norm", []), post3.get("loss_mm_h", []))
            post3_loss = post3["loss_mm_h"].to_numpy(dtype=float) if not post3.empty else np.array([])
            post3_cv = float(np.nanstd(post3_loss) / np.nanmean(post3_loss)) if len(post3_loss) >= 5 and np.nanmean(post3_loss) > 0 else np.nan

            event_records.append(
                {
                    "event_id": event_id,
                    "layer": layer,
                    "detected_start": peak_rec["detected_start"],
                    "peak_start": peak_rec["peak_start"],
                    "peak_start_shift_h": peak_rec["peak_start_shift_h"],
                    "peak_start_gain_mm": peak_rec["peak_start_gain_mm"],
                    "start": start,
                    "end": end,
                    "points": len(segment),
                    "duration_h": duration_h,
                    "start_mm": float(y[0]),
                    "end_mm": float(y[-1]),
                    "total_drop_mm": total_drop,
                    "mean_loss_mm_h": total_drop / duration_h,
                    "rain_before_12": rain_before_12,
                    "rain_before_24": rain_before_24,
                    "rain_before_48": rain_before_48,
                    "rain_during": rain_during,
                    "associated_12h": rain_before_12 > ASSOCIATION_RAIN_THRESHOLD_IN,
                    "associated_24h": rain_before_24 > ASSOCIATION_RAIN_THRESHOLD_IN,
                    "associated_48h": rain_before_48 > ASSOCIATION_RAIN_THRESHOLD_IN,
                    "interrupted_by_rain": rain_during > INTERRUPTION_RAIN_THRESHOLD_IN,
                    "rain_lag_h": lag_h,
                    "full_r2": fit_full["r2"],
                    "full_tau_h": fit_full["tau_h"],
                    "trim1_r2": fit_1["r2"],
                    "trim1_tau_h": fit_1["tau_h"],
                    "trim3_r2": fit_3["r2"],
                    "trim3_tau_h": fit_3["tau_h"],
                    "trim6_r2": fit_6["r2"],
                    "trim6_tau_h": fit_6["tau_h"],
                    "early3_drop_mm": early3_drop,
                    "early3_drop_share": early3_drop_share,
                    "post3_loss_storage_slope": slope,
                    "post3_loss_storage_corr": corr,
                    "post3_loss_cv": post3_cv,
                }
            )
            previous_adjusted_end = pd.Timestamp(end)

    events_df = pd.DataFrame(event_records)
    if not events_df.empty:
        events_df["audit_class"] = np.select(
            [
                events_df["associated_24h"] & ~events_df["interrupted_by_rain"],
                events_df["associated_48h"] & ~events_df["interrupted_by_rain"],
                events_df["associated_48h"] & events_df["interrupted_by_rain"],
                ~events_df["associated_48h"],
            ],
            [
                "rain-associated clean <=24h",
                "rain-associated clean 24-48h",
                "rain-associated but interrupted",
                "not associated within 48h",
            ],
            default="ambiguous",
        )
        events_df["regime_proxy"] = events_df.apply(classify_regime_proxy, axis=1)
    loss_points_df = pd.concat(loss_frames, ignore_index=True) if loss_frames else pd.DataFrame()
    return events_df, loss_points_df


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    grouped = events.groupby("layer", dropna=False)
    summary = grouped.agg(
        events=("event_id", "count"),
        median_duration_h=("duration_h", "median"),
        median_total_drop_mm=("total_drop_mm", "median"),
        associated_24h_rate=("associated_24h", "mean"),
        associated_48h_rate=("associated_48h", "mean"),
        interrupted_rate=("interrupted_by_rain", "mean"),
        median_lag_h=("rain_lag_h", "median"),
        median_full_r2=("full_r2", "median"),
        median_trim3_r2=("trim3_r2", "median"),
        median_full_tau_h=("full_tau_h", "median"),
        median_trim3_tau_h=("trim3_tau_h", "median"),
        median_early3_drop_share=("early3_drop_share", "median"),
        stageII_like_rate=("regime_proxy", lambda x: float((x == "stage-II-like").mean())),
        stageI_like_rate=("regime_proxy", lambda x: float((x == "stage-I-like").mean())),
        early_transient_heavy_rate=("regime_proxy", lambda x: float((x == "early-transient-heavy").mean())),
    )
    return summary.reset_index()


def plot_detection_audit(events: pd.DataFrame) -> None:
    if events.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    class_counts = events.groupby(["layer", "audit_class"]).size().reset_index(name="n")
    sns.barplot(data=class_counts, x="layer", y="n", hue="audit_class", ax=axes[0])
    axes[0].set_title("SMDE rain-association audit")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Detected SMDE count")
    axes[0].tick_params(axis="x", rotation=30)

    clean = events[events["associated_48h"] & np.isfinite(events["rain_lag_h"])]
    sns.histplot(data=clean, x="rain_lag_h", hue="layer", bins=24, multiple="stack", ax=axes[1])
    axes[1].set_title("Lag from previous rain to SMDE start")
    axes[1].set_xlabel("Lag (h)")

    sns.scatterplot(data=events, x="duration_h", y="total_drop_mm", hue="audit_class", style="layer", s=45, ax=axes[2])
    axes[2].set_title("Event duration and total dry-down")
    axes[2].set_xlabel("Duration (h)")
    axes[2].set_ylabel("Total drop (mm)")

    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.set_title("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_detection_audit_alachua.png", dpi=300)
    plt.close(fig)


def plot_regime_diagnostics(events: pd.DataFrame, loss_points: pd.DataFrame) -> None:
    if events.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    if not loss_points.empty:
        loss_plot = loss_points[(loss_points["loss_mm_h"] > 0) & (loss_points["loss_mm_h"] < loss_points["loss_mm_h"].quantile(0.99))]
        loss_plot = loss_plot.copy()
        loss_plot["storage_bin"] = pd.cut(loss_plot["storage_norm"], bins=np.linspace(0, 1, 16), include_lowest=True)
        binned = (
            loss_plot.groupby(["layer", "storage_bin"], observed=True)
            .agg(storage_norm=("storage_norm", "mean"), loss_mm_h=("loss_mm_h", "median"))
            .reset_index()
        )
        sns.lineplot(data=binned, x="storage_norm", y="loss_mm_h", hue="layer", marker="o", ax=axes[0])
    axes[0].set_title("Observed loss-rate relation")
    axes[0].set_xlabel("Normalized storage within SMDE")
    axes[0].set_ylabel("Median loss rate (mm h-1)")

    fit_long = events.melt(
        id_vars=["event_id", "layer"],
        value_vars=["full_r2", "trim1_r2", "trim3_r2", "trim6_r2"],
        var_name="fit",
        value_name="r2",
    )
    sns.boxplot(data=fit_long, x="fit", y="r2", hue="layer", ax=axes[1])
    axes[1].set_title("Exponential fit sensitivity")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R2")
    axes[1].tick_params(axis="x", rotation=25)

    regime_counts = events.groupby(["layer", "regime_proxy"]).size().reset_index(name="n")
    sns.barplot(data=regime_counts, x="layer", y="n", hue="regime_proxy", ax=axes[2])
    axes[2].set_title("Event-level regime proxy")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Event count")
    axes[2].tick_params(axis="x", rotation=30)

    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.set_title("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_regime_diagnostics_alachua.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_out_dir()
    cfg = DetectionConfig()
    df = read_alachua()
    events, loss_points = build_event_tables(df, cfg)
    summary = summarize(events)

    events.to_csv(OUT_DIR / "alachua_smde_event_audit.csv", index=False)
    loss_points.to_csv(OUT_DIR / "alachua_smde_loss_points.csv", index=False)
    summary.to_csv(OUT_DIR / "alachua_smde_summary_by_layer.csv", index=False)
    plot_detection_audit(events)
    plot_regime_diagnostics(events, loss_points)

    print(f"Input rows: {len(df):,}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Detected events: {len(events):,}")
    print(f"Output directory: {OUT_DIR}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
