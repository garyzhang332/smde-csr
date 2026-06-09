from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from smde_regime_audit import DetectionConfig, build_event_tables, summarize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FAWN_EXPORT_DIR", ROOT / "data" / "fawn_exports"))
OUT_DIR = Path(os.environ.get("SMDE_AUDIT_OUT_DIR", ROOT / "_analysis" / "fawn_full_smde_audit"))

YEARS = [2023, 2024, 2025]
PCT_TO_MM_4IN = 4.0 * 25.4 / 100.0

MOISTURE_RENAME = {
    "moisture_sms_4_inch_pct": "moisture_4in",
    "moisture_sms_8_inch_pct": "moisture_8in",
    "moisture_sms_12_inch_pct": "moisture_12in",
    "moisture_sms_16_inch_pct": "moisture_16in",
    "moisture_sms_20_inch_pct": "moisture_20in",
}
MOISTURE_SOURCE_COLS = list(MOISTURE_RENAME.keys())
MOISTURE_LAYERS = list(MOISTURE_RENAME.values())

WX_VALUE_COLS = [
    "rain_2m_inches",
    "rain_backup_2m_inches",
    "temp_air_2m_C",
    "rh_2m_pct",
    "wind_speed_10m_mph",
    "rfd_2m_wm2",
    "trf_2m_kJm2",
    "vp_2m_kPa",
    "vp_sat_2m_kPa",
    "vp_def_2m_kPa",
    "temp_dp_2m_C",
    "temp_wb_2m_C",
    "temp_soil_10cm_C",
]
WX_CONTEXT_COLS = [
    "vp_def_2m_kPa",
    "temp_air_2m_C",
    "rh_2m_pct",
    "wind_speed_10m_mph",
    "rfd_2m_wm2",
    "trf_2m_kJm2",
    "temp_soil_10cm_C",
]


def make_config() -> DetectionConfig:
    return DetectionConfig(
        min_steps=4,
        smooth_window_length=49,
        smooth_poly_order=3,
        noise_tolerance_mm=0.001 * PCT_TO_MM_4IN,
        min_total_decrease_mm=0.1 * PCT_TO_MM_4IN,
        max_total_decrease_mm=15.0 * PCT_TO_MM_4IN,
        max_val_low_thr_mm=2.0 * PCT_TO_MM_4IN,
        min_val_thr_mm=2.0 * PCT_TO_MM_4IN,
        max_internal_range_mm=15.0 * PCT_TO_MM_4IN,
        outlier_threshold_mm=5.0 * PCT_TO_MM_4IN,
        apply_outlier_filter_in_detection=False,
        min_drop_rate_mm_per_step=None,
        snap_start_to_peak=True,
        peak_search_before_h=48.0,
        peak_search_after_h=0.5,
        peak_min_gain_mm=0.0,
        peak_recession_break_mm=0.05 * PCT_TO_MM_4IN,
        peak_max_rewetting_fraction=0.35,
        peak_max_rewetting_mm=0.25 * PCT_TO_MM_4IN,
    )


def read_station_parquet(paths: list[Path], site_id: int, columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, columns=columns, filters=[("ID", "==", int(site_id))])
        except Exception:
            frame = pd.read_parquet(path, columns=columns)
            frame = frame[frame["ID"] == site_id]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def normalize_moisture_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.mask(numeric <= 0)
    positive = numeric.dropna()
    if positive.empty:
        return numeric

    if positive.median() <= 1.0 and positive.quantile(0.95) <= 1.5:
        numeric = numeric * 100.0
    return numeric


def get_station_ids() -> list[int]:
    ids = []
    for path in sorted(DATA_DIR.glob("soil_moisture_*.parquet")):
        frame = pd.read_parquet(path, columns=["ID"])
        ids.extend(frame["ID"].dropna().astype(int).unique().tolist())
    return sorted(set(ids))


def prepare_soil(site_id: int) -> pd.DataFrame:
    paths = [DATA_DIR / f"soil_moisture_{year}.parquet" for year in YEARS]
    cols = ["ID", "UTC", *MOISTURE_SOURCE_COLS]
    soil = read_station_parquet(paths, site_id, cols)
    if soil.empty:
        return soil

    soil["UTC"] = pd.to_datetime(soil["UTC"], errors="coerce")
    soil = soil.dropna(subset=["UTC"]).sort_values("UTC")
    soil = soil.drop_duplicates(subset=["UTC"], keep="last")
    for col in MOISTURE_SOURCE_COLS:
        soil[col] = normalize_moisture_pct(soil[col]) * PCT_TO_MM_4IN
    return soil.rename(columns=MOISTURE_RENAME)


def prepare_wx(site_id: int) -> pd.DataFrame:
    paths = [DATA_DIR / f"wx_selected_{year}.parquet" for year in YEARS]
    cols = ["ID", "UTC", *WX_VALUE_COLS]
    wx = read_station_parquet(paths, site_id, cols)
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


def merge_station(soil: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    soil_indexed = soil.set_index("UTC")[MOISTURE_LAYERS].sort_index()
    wx_indexed = wx.set_index("UTC").sort_index()
    merged = soil_indexed.join(wx_indexed, how="left")
    merged["Rain"] = merged["Rain"].fillna(0.0)
    merged["rain_backup_used"] = merged["rain_backup_used"].astype("boolean").fillna(False).astype(bool)
    return merged


def safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() == 0:
        return float("nan")
    return float(clean.mean())


def safe_max(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() == 0:
        return float("nan")
    return float(clean.max())


def add_weather_context(events: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events

    records = []
    for row in events.itertuples(index=False):
        start = pd.Timestamp(row.start)
        end = pd.Timestamp(row.end)
        during = merged.loc[start:end]
        pre6 = merged.loc[start - pd.Timedelta(hours=6) : start]
        pre24 = merged.loc[start - pd.Timedelta(hours=24) : start]

        rec = {
            "event_id": row.event_id,
            "start_year": int(start.year),
            "start_month": int(start.month),
            "start_doy": int(start.dayofyear),
            "rain_backup_used_during": bool(during["rain_backup_used"].any()) if "rain_backup_used" in during else False,
            "pre6_rain_inches": float(pre6["Rain"].sum()) if "Rain" in pre6 else float("nan"),
            "pre24_rain_inches": float(pre24["Rain"].sum()) if "Rain" in pre24 else float("nan"),
        }
        for col in WX_CONTEXT_COLS:
            if col in merged.columns:
                rec[f"during_{col}_mean"] = safe_mean(during[col])
                rec[f"during_{col}_max"] = safe_max(during[col])
                rec[f"pre6_{col}_mean"] = safe_mean(pre6[col])
            else:
                rec[f"during_{col}_mean"] = float("nan")
                rec[f"during_{col}_max"] = float("nan")
                rec[f"pre6_{col}_mean"] = float("nan")
        records.append(rec)

    context = pd.DataFrame(records)
    return events.merge(context, on="event_id", how="left")


def process_site(site_id: int, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    soil = prepare_soil(site_id)
    wx = prepare_wx(site_id)
    status = {
        "site_id": site_id,
        "soil_rows": int(len(soil)),
        "wx_rows": int(len(wx)),
        "events": 0,
        "loss_points": 0,
    }
    if soil.empty or wx.empty:
        return pd.DataFrame(), pd.DataFrame(), status

    merged = merge_station(soil, wx)
    events, loss_points = build_event_tables(merged[[*MOISTURE_LAYERS, "Rain"]], cfg)
    if events.empty:
        return events, loss_points, status

    events = events.copy()
    loss_points = loss_points.copy()
    events["event_id"] = events["event_id"].map(lambda x: f"S{site_id}_{x}")
    loss_points["event_id"] = loss_points["event_id"].map(lambda x: f"S{site_id}_{x}")
    events.insert(0, "site_id", site_id)
    loss_points.insert(0, "site_id", site_id)
    events = add_weather_context(events, merged)

    status["events"] = int(len(events))
    status["loss_points"] = int(len(loss_points))
    return events, loss_points, status


def summarize_by_site_layer(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["site_id", "layer"], dropna=False)
        .agg(
            events=("event_id", "count"),
            associated_24h_rate=("associated_24h", "mean"),
            associated_48h_rate=("associated_48h", "mean"),
            interrupted_rate=("interrupted_by_rain", "mean"),
            clean_24h_rate=("event_id", lambda x: float("nan")),
            median_duration_h=("duration_h", "median"),
            median_drop_mm=("total_drop_mm", "median"),
            median_trim3_r2=("trim3_r2", "median"),
            median_trim3_tau_h=("trim3_tau_h", "median"),
            median_early3_drop_share=("early3_drop_share", "median"),
            stageII_like_rate=("regime_proxy", lambda x: float((x == "stage-II-like").mean())),
            stageI_like_rate=("regime_proxy", lambda x: float((x == "stage-I-like").mean())),
            early_transient_heavy_rate=("regime_proxy", lambda x: float((x == "early-transient-heavy").mean())),
            during_vpd_mean=("during_vp_def_2m_kPa_mean", "mean"),
        )
        .reset_index()
    )


def add_clean_rates(events: pd.DataFrame, summary: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    clean = events.assign(
        clean_24h=events["associated_24h"] & ~events["interrupted_by_rain"],
        clean_48h=events["associated_48h"] & ~events["interrupted_by_rain"],
        stageII_clean_48h=(events["regime_proxy"] == "stage-II-like")
        & events["associated_48h"]
        & ~events["interrupted_by_rain"],
    )
    rates = (
        clean.groupby(keys, dropna=False)
        .agg(
            clean_24h_rate=("clean_24h", "mean"),
            clean_48h_rate=("clean_48h", "mean"),
            stageII_clean_48h_rate=("stageII_clean_48h", "mean"),
        )
        .reset_index()
    )
    return summary.drop(columns=[c for c in rates.columns if c in summary.columns and c not in keys], errors="ignore").merge(
        rates, on=keys, how="left"
    )


def summarize_by_year_layer(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby(["start_year", "layer"], dropna=False)
    summary = grouped.agg(
        events=("event_id", "count"),
        stations=("site_id", "nunique"),
        associated_24h_rate=("associated_24h", "mean"),
        associated_48h_rate=("associated_48h", "mean"),
        interrupted_rate=("interrupted_by_rain", "mean"),
        median_duration_h=("duration_h", "median"),
        median_drop_mm=("total_drop_mm", "median"),
        median_trim3_r2=("trim3_r2", "median"),
        stageII_like_rate=("regime_proxy", lambda x: float((x == "stage-II-like").mean())),
        early_transient_heavy_rate=("regime_proxy", lambda x: float((x == "early-transient-heavy").mean())),
        during_vpd_mean=("during_vp_def_2m_kPa_mean", "mean"),
    )
    return add_clean_rates(events, summary.reset_index(), ["start_year", "layer"])


def summarize_demand(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "during_vp_def_2m_kPa_mean",
        "during_temp_air_2m_C_mean",
        "during_rh_2m_pct_mean",
        "during_wind_speed_10m_mph_mean",
        "during_rfd_2m_wm2_mean",
        "mean_loss_mm_h",
        "post3_loss_storage_corr",
    ]
    available = [c for c in cols if c in events.columns]
    return (
        events.groupby(["layer", "regime_proxy"], dropna=False)
        .agg(
            events=("event_id", "count"),
            **{f"median_{col}": (col, "median") for col in available},
        )
        .reset_index()
    )


def summarize_loss_bins(loss_points: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if loss_points.empty:
        return pd.DataFrame()

    regimes = events[["event_id", "regime_proxy"]].drop_duplicates()
    loss = loss_points.merge(regimes, on="event_id", how="left")
    loss = loss[(loss["loss_mm_h"] > 0) & loss["storage_norm"].between(0, 1)]
    if loss.empty:
        return pd.DataFrame()
    q995 = loss.groupby("layer")["loss_mm_h"].transform(lambda s: s.quantile(0.995))
    loss = loss[loss["loss_mm_h"] <= q995]
    bins = np.linspace(0, 1, 16)
    loss["storage_bin"] = pd.cut(loss["storage_norm"], bins=bins, include_lowest=True)
    binned = (
        loss.groupby(["layer", "regime_proxy", "storage_bin"], observed=True)
        .agg(
            storage_norm=("storage_norm", "mean"),
            loss_mm_h_median=("loss_mm_h", "median"),
            loss_mm_h_q25=("loss_mm_h", lambda x: float(np.quantile(x, 0.25))),
            loss_mm_h_q75=("loss_mm_h", lambda x: float(np.quantile(x, 0.75))),
            points=("loss_mm_h", "count"),
        )
        .reset_index()
    )
    binned["storage_bin"] = binned["storage_bin"].astype(str)
    return binned


def plot_detection(events: pd.DataFrame, by_site_layer: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    class_counts = events.groupby(["layer", "audit_class"]).size().reset_index(name="events")
    sns.barplot(data=class_counts, x="layer", y="events", hue="audit_class", errorbar=None, ax=axes[0, 0])
    axes[0, 0].set_title("SMDE detection audit by rain association")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Detected event count")
    axes[0, 0].tick_params(axis="x", rotation=20)

    heat = by_site_layer.pivot(index="site_id", columns="layer", values="events").fillna(0)
    sns.heatmap(heat, cmap="YlGnBu", linewidths=0.2, ax=axes[0, 1])
    axes[0, 1].set_title("Detected SMDE count by station and depth")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("FAWN site ID")

    clean = events[events["associated_48h"] & events["rain_lag_h"].notna()]
    sns.histplot(data=clean, x="rain_lag_h", hue="layer", bins=32, multiple="stack", ax=axes[1, 0])
    axes[1, 0].set_title("Lag from previous rain to SMDE start")
    axes[1, 0].set_xlabel("Lag (h)")
    axes[1, 0].set_ylabel("Event count")

    sample = events.sample(n=min(len(events), 6000), random_state=7) if len(events) > 6000 else events
    sns.scatterplot(
        data=sample,
        x="duration_h",
        y="total_drop_mm",
        hue="audit_class",
        style="layer",
        s=22,
        alpha=0.65,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Event duration and total soil-moisture decrease")
    axes[1, 1].set_xlabel("Duration (h)")
    axes[1, 1].set_ylabel("Total drop (mm)")
    if events["duration_h"].notna().any():
        xmax = float(events["duration_h"].quantile(0.995))
        axes[1, 1].set_xlim(0, max(24.0, xmax * 1.05))

    for ax in axes.flat:
        legend = ax.get_legend()
        if legend:
            legend.set_title("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_full_detection_audit.png", dpi=300)
    plt.close(fig)


def plot_regime(events: pd.DataFrame, binned_loss: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if not binned_loss.empty:
        layer_loss = (
            binned_loss.groupby(["layer", "storage_bin"])
            .agg(storage_norm=("storage_norm", "mean"), loss_mm_h_median=("loss_mm_h_median", "median"))
            .reset_index()
        )
        sns.lineplot(data=layer_loss, x="storage_norm", y="loss_mm_h_median", hue="layer", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Observed loss-rate relation")
    axes[0, 0].set_xlabel("Normalized storage within SMDE")
    axes[0, 0].set_ylabel("Median loss rate (mm h-1)")

    fit_long = events.melt(
        id_vars=["event_id", "layer"],
        value_vars=["full_r2", "trim1_r2", "trim3_r2", "trim6_r2"],
        var_name="fit",
        value_name="r2",
    )
    sns.boxplot(data=fit_long, x="fit", y="r2", hue="layer", showfliers=False, ax=axes[0, 1])
    axes[0, 1].set_title("Exponential fit sensitivity")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("R2")
    axes[0, 1].tick_params(axis="x", rotation=20)

    regime = events.groupby(["layer", "regime_proxy"]).size().reset_index(name="events")
    totals = regime.groupby("layer")["events"].transform("sum")
    regime["share"] = regime["events"] / totals
    sns.barplot(data=regime, x="layer", y="share", hue="regime_proxy", errorbar=None, ax=axes[1, 0])
    axes[1, 0].set_title("Regime proxy composition")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Share of events")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis="x", rotation=20)

    demand_sample = events[
        events["during_vp_def_2m_kPa_mean"].notna() & events["mean_loss_mm_h"].notna()
    ]
    demand_sample = demand_sample.sample(n=min(len(demand_sample), 6000), random_state=11) if len(demand_sample) > 6000 else demand_sample
    sns.scatterplot(
        data=demand_sample,
        x="during_vp_def_2m_kPa_mean",
        y="mean_loss_mm_h",
        hue="regime_proxy",
        style="layer",
        s=24,
        alpha=0.6,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Atmospheric demand context")
    axes[1, 1].set_xlabel("Mean VPD during event (kPa)")
    axes[1, 1].set_ylabel("Mean loss rate (mm h-1)")
    if demand_sample["mean_loss_mm_h"].notna().any():
        ymax = float(demand_sample["mean_loss_mm_h"].quantile(0.995))
        axes[1, 1].set_ylim(0, max(1.0, ymax * 1.05))

    for ax in axes.flat:
        legend = ax.get_legend()
        if legend:
            legend.set_title("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_full_regime_composition.png", dpi=300)
    plt.close(fig)


def plot_site_heatmaps(by_site_layer: pd.DataFrame) -> None:
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    metrics = [
        ("associated_48h_rate", "Rain-associated within 48h"),
        ("interrupted_rate", "Interrupted by rain"),
        ("stageII_like_rate", "Stage-II-like proxy"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        heat = by_site_layer.pivot(index="site_id", columns="layer", values=metric)
        sns.heatmap(heat, vmin=0, vmax=1, cmap="viridis", linewidths=0.2, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("FAWN site ID")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_full_site_layer_heatmaps.png", dpi=300)
    plt.close(fig)


def write_outputs(
    events: pd.DataFrame,
    loss_points: pd.DataFrame,
    statuses: list[dict[str, object]],
    cfg: DetectionConfig,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events.to_csv(OUT_DIR / "full_smde_event_audit.csv", index=False)
    loss_points.to_parquet(OUT_DIR / "full_smde_loss_points.parquet", index=False)
    pd.DataFrame(statuses).to_csv(OUT_DIR / "processing_status_by_site.csv", index=False)

    by_layer = add_clean_rates(events, summarize(events.drop(columns=["site_id"], errors="ignore")), ["layer"])
    by_site_layer = add_clean_rates(events, summarize_by_site_layer(events), ["site_id", "layer"])
    by_year_layer = summarize_by_year_layer(events)
    demand = summarize_demand(events)
    binned_loss = summarize_loss_bins(loss_points, events)
    audit_counts = events.groupby(["layer", "audit_class"]).size().reset_index(name="events")
    regime_counts = events.groupby(["layer", "regime_proxy"]).size().reset_index(name="events")

    by_layer.to_csv(OUT_DIR / "full_smde_summary_by_layer.csv", index=False)
    by_site_layer.to_csv(OUT_DIR / "full_smde_summary_by_site_layer.csv", index=False)
    by_year_layer.to_csv(OUT_DIR / "full_smde_summary_by_year_layer.csv", index=False)
    demand.to_csv(OUT_DIR / "full_smde_demand_regime_summary.csv", index=False)
    binned_loss.to_csv(OUT_DIR / "full_smde_binned_loss_by_layer_regime.csv", index=False)
    audit_counts.to_csv(OUT_DIR / "full_smde_audit_class_counts.csv", index=False)
    regime_counts.to_csv(OUT_DIR / "full_smde_regime_counts.csv", index=False)

    plot_detection(events, by_site_layer)
    plot_regime(events, binned_loss)
    plot_site_heatmaps(by_site_layer)

    manifest = {
        "detection_version": "smde_regime_audit.build_event_tables+fawn_full_make_config_v2",
        "detection_config": asdict(cfg),
        "years": YEARS,
        "script_mtime": {
            "fawn_full_smde_audit.py": pd.Timestamp(Path(__file__).stat().st_mtime, unit="s").isoformat(),
            "smde_regime_audit.py": pd.Timestamp((BASE_DIR / "smde_regime_audit.py").stat().st_mtime, unit="s").isoformat(),
        },
        "events": int(len(events)),
        "loss_points": int(len(loss_points)),
        "sites": int(events["site_id"].nunique()),
        "layers": sorted(events["layer"].unique().tolist()),
        "start": str(events["start"].min()),
        "end": str(events["end"].max()),
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
    }
    (OUT_DIR / "full_smde_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print("\nSummary by layer:")
    print(by_layer.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--sites", nargs="*", type=int, default=None)
    args = parser.parse_args()

    cfg = make_config()
    station_ids = args.sites if args.sites else get_station_ids()
    station_ids = sorted(set(station_ids))
    if args.max_sites:
        station_ids = station_ids[: args.max_sites]

    all_events = []
    all_loss_points = []
    statuses = []
    for idx, site_id in enumerate(station_ids, start=1):
        events, loss_points, status = process_site(site_id, cfg)
        statuses.append(status)
        if not events.empty:
            all_events.append(events)
        if not loss_points.empty:
            all_loss_points.append(loss_points)
        print(
            f"[{idx:02d}/{len(station_ids):02d}] site {site_id}: "
            f"soil={status['soil_rows']:,}, wx={status['wx_rows']:,}, "
            f"events={status['events']:,}, loss_points={status['loss_points']:,}"
        )

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    loss_points_df = pd.concat(all_loss_points, ignore_index=True) if all_loss_points else pd.DataFrame()
    if events_df.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(statuses).to_csv(OUT_DIR / "processing_status_by_site.csv", index=False)
        print("No events detected.")
        return

    write_outputs(events_df, loss_points_df, statuses, cfg)


if __name__ == "__main__":
    main()
