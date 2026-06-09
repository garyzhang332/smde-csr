from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
AUDIT = ANALYSIS / "fawn_full_smde_audit"
EXP3_SOURCE = ANALYSIS / "experiment3_adaptive_regime_csr" / "source_data"
EXP4_SOURCE = ANALYSIS / "experiment4_regime_conditioned_forecast" / "source_data"
OUT = ANALYSIS / "experiment2_loss_function_regime_diagnosis"
SOURCE = OUT / "source_data"
SUBMISSION_FIGURES = ROOT / "figures"

EVENTS_FILE = AUDIT / "full_smde_event_audit.csv"
BINNED_FILE = AUDIT / "full_smde_binned_loss_by_layer_regime.csv"
CONSTRUCTION_POOL_FILE = EXP3_SOURCE / "adaptive_regime_construction_pool_summary.csv"
SEGMENT_POINTS_FILE = EXP3_SOURCE / "adaptive_segment_points_all.parquet"

REP_SITE = 270
REP_LAYER = "moisture_4in"
REP_EVENT_ID = "S270_moisture_4in_0034"

LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
EVENT_REGIME_ORDER = ["stage-II-like", "stage-I-like", "early-transient-heavy", "mixed_or_uncertain"]
EVENT_REGIME_LABELS = {
    "stage-II-like": "Stage-II-like",
    "stage-I-like": "Stage-I-like",
    "early-transient-heavy": "Early transient",
    "mixed_or_uncertain": "Mixed/uncertain",
}
SEGMENT_ORDER = ["early_transient", "stageI_like", "stageII_like"]
SEGMENT_LABELS = {
    "early_transient": "Early transient",
    "stageI_like": "Stage I-like",
    "stageII_like": "Stage II-like",
}
REGIME_COLORS = {
    "stage-II-like": "#6FA38A",
    "stage-I-like": "#4E7E9E",
    "early-transient-heavy": "#C78D4B",
    "mixed_or_uncertain": "#B7B7B7",
    "early_transient": "#C78D4B",
    "stageI_like": "#4E7E9E",
    "stageII_like": "#6FA38A",
}
COLORS = {
    "neutral_dark": "#303030",
    "neutral_mid": "#737373",
    "neutral_light": "#E9EDF1",
    "grid": "#E6EBEF",
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
    SUBMISSION_FIGURES.mkdir(parents=True, exist_ok=True)


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    shutil.copyfile(OUT / f"{stem}.pdf", SUBMISSION_FIGURES / "Figure_5_loss_function_regime_diagnosis.pdf")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )


def read_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])
    events["layer"] = pd.Categorical(events["layer"], categories=LAYER_ORDER, ordered=True)
    events["regime_proxy"] = pd.Categorical(events["regime_proxy"], categories=EVENT_REGIME_ORDER, ordered=True)
    events["clean_48h"] = events["associated_48h"] & ~events["interrupted_by_rain"]
    events["clean_interpretable_48h"] = events["clean_48h"] & events["regime_proxy"].astype(str).isin(
        ["stage-II-like", "stage-I-like", "early-transient-heavy"]
    )
    return events.sort_values(["layer", "site_id", "start"])


def read_binned() -> pd.DataFrame:
    binned = pd.read_csv(BINNED_FILE)
    binned["layer"] = pd.Categorical(binned["layer"], categories=LAYER_ORDER, ordered=True)
    binned["regime_proxy"] = pd.Categorical(binned["regime_proxy"], categories=EVENT_REGIME_ORDER, ordered=True)
    return binned.sort_values(["layer", "regime_proxy", "storage_norm"])


def read_construction_pool() -> pd.DataFrame:
    pool = pd.read_csv(CONSTRUCTION_POOL_FILE)
    pool["layer"] = pd.Categorical(pool["layer"], categories=LAYER_ORDER, ordered=True)
    pool["segment"] = pd.Categorical(pool["segment"], categories=SEGMENT_ORDER, ordered=True)
    return pool.sort_values(["layer", "segment"])


def read_segment_points() -> pd.DataFrame:
    points = pd.read_parquet(SEGMENT_POINTS_FILE)
    points["layer"] = pd.Categorical(points["layer"], categories=LAYER_ORDER, ordered=True)
    points["segment_regime"] = pd.Categorical(points["segment_regime"], categories=SEGMENT_ORDER, ordered=True)
    return points.sort_values(["site_id", "layer", "parent_event_id", "t_h"])


def select_representative_event(events: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    chosen = points[
        (points["site_id"].astype(int) == REP_SITE)
        & (points["layer"].astype(str) == REP_LAYER)
        & (points["parent_event_id"].astype(str) == REP_EVENT_ID)
    ].copy()
    if chosen.empty:
        required = set(SEGMENT_ORDER)
        candidates = (
            points.groupby(["site_id", "layer", "parent_event_id"], observed=False)["segment_regime"]
            .agg(lambda s: set(str(v) for v in s.dropna()))
            .reset_index(name="regimes")
        )
        candidates = candidates[candidates["regimes"].map(lambda s: required.issubset(s))]
        candidates = candidates.merge(
            events[["site_id", "layer", "event_id", "start_mm", "end_mm", "duration_h"]],
            left_on=["site_id", "layer", "parent_event_id"],
            right_on=["site_id", "layer", "event_id"],
            how="left",
        )
        candidates["drop_mm"] = candidates["start_mm"] - candidates["end_mm"]
        if candidates.empty:
            raise RuntimeError("No representative event with all three segment regimes was found.")
        best = candidates.sort_values(["drop_mm", "duration_h"], ascending=False).iloc[0]
        chosen = points[
            (points["site_id"].astype(int) == int(best["site_id"]))
            & (points["layer"].astype(str) == str(best["layer"]))
            & (points["parent_event_id"].astype(str) == str(best["parent_event_id"]))
        ].copy()

    event_id = str(chosen["parent_event_id"].iloc[0])
    event_rows = events[events["event_id"].astype(str) == event_id]
    if event_rows.empty:
        raise RuntimeError(f"Representative event {event_id} was not found in the event audit table.")
    meta = event_rows.iloc[0].copy()
    chosen["timestamp"] = meta["start"] + pd.to_timedelta(chosen["t_h"], unit="h")
    return meta, chosen.sort_values(["t_h", "adaptive_segment_id"])


def calculate_segment_loss_points(event_points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for segment_id, segment in event_points.groupby("adaptive_segment_id", observed=False):
        segment = segment.sort_values("t_h")
        if len(segment) < 2:
            continue
        t = segment["t_h"].to_numpy(dtype=float)
        s = segment["moisture_mm"].to_numpy(dtype=float)
        x = segment["event_storage_norm"].to_numpy(dtype=float)
        dt = np.diff(t)
        keep = dt > 0
        if not np.any(keep):
            continue
        loss = -np.diff(s)[keep] / dt[keep]
        t_mid = (t[:-1][keep] + t[1:][keep]) / 2
        s_mid = (s[:-1][keep] + s[1:][keep]) / 2
        x_mid = (x[:-1][keep] + x[1:][keep]) / 2
        rows.extend(
            {
                "adaptive_segment_id": str(segment_id),
                "segment_order": int(segment["adaptive_segment_order"].iloc[0]),
                "segment_regime": str(segment["segment_regime"].iloc[0]),
                "t_mid_h": float(t_mid[i]),
                "storage_mid_mm": float(s_mid[i]),
                "storage_norm_mid": float(x_mid[i]),
                "loss_mm_h": float(loss[i]),
            }
            for i in range(len(loss))
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["storage_norm_mid", "loss_mm_h"])
        out = out[out["loss_mm_h"] >= 0]
    return out


def _segment_summary(event_points: pd.DataFrame) -> pd.DataFrame:
    return (
        event_points.groupby("adaptive_segment_id", observed=False)
        .agg(
            segment_regime=("segment_regime", "first"),
            segment_order=("adaptive_segment_order", "first"),
            start_t_h=("t_h", "min"),
            end_t_h=("t_h", "max"),
            start_mm=("moisture_mm", "first"),
            end_mm=("moisture_mm", "last"),
            start_norm=("event_storage_norm", "first"),
            end_norm=("event_storage_norm", "last"),
        )
        .sort_values("segment_order")
        .reset_index()
    )


def _style_segment_axis(ax: plt.Axes) -> None:
    ax.grid(color=COLORS["grid"], lw=0.65)
    ax.tick_params(labelsize=6.8)


def plot_representative_loss_storage(
    ax: plt.Axes, event_meta: pd.Series, event_points: pd.DataFrame, loss_points: pd.DataFrame
) -> None:
    _style_segment_axis(ax)
    summaries = _segment_summary(event_points)
    for row in summaries.itertuples(index=False):
        ax.axvspan(row.end_norm, row.start_norm, color=REGIME_COLORS[str(row.segment_regime)], alpha=0.06, lw=0)

    for regime in SEGMENT_ORDER:
        d = loss_points[loss_points["segment_regime"] == regime].sort_values("storage_norm_mid")
        if d.empty:
            continue
        color = REGIME_COLORS[regime]
        ax.scatter(
            d["storage_norm_mid"],
            d["loss_mm_h"],
            s=14,
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )
        ax.plot(d["storage_norm_mid"], d["loss_mm_h"], color=color, lw=1.4, alpha=0.95, zorder=4)

    ymax = float(np.nanpercentile(loss_points["loss_mm_h"], 97) * 1.25) if not loss_points.empty else 1.0
    ax.set_ylim(0, max(0.35, ymax))
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Normalized storage, x (0 = driest, 1 = wettest)")
    ax.set_ylabel(r"Loss rate, $L=-dS/dt$ (mm h$^{-1}$)")
    ax.set_title("Segment diagnosis in loss-storage space", loc="left", fontsize=8)

    for regime, xpos, ypos in [
        ("early_transient", 0.88, 0.86),
        ("stageI_like", 0.52, 0.70),
        ("stageII_like", 0.20, 0.52),
    ]:
        ax.text(
            xpos,
            ypos,
            SEGMENT_LABELS[regime],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.1,
            color=REGIME_COLORS[regime],
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        )
    ax.annotate(
        "drying direction",
        xy=(0.18, 0.06),
        xytext=(0.54, 0.06),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["neutral_mid"]),
        ha="center",
        va="center",
        fontsize=6,
        color=COLORS["neutral_mid"],
    )
    add_panel_label(ax, "a")


def plot_representative_smde(ax: plt.Axes, event_meta: pd.Series, event_points: pd.DataFrame) -> None:
    _style_segment_axis(ax)
    summaries = _segment_summary(event_points)
    for row in summaries.itertuples(index=False):
        ax.axvspan(row.start_t_h, row.end_t_h, color=REGIME_COLORS[str(row.segment_regime)], alpha=0.08, lw=0)
        if row.segment_order > 0:
            ax.axvline(row.start_t_h, color="#7B8790", lw=0.75, ls=(0, (3, 2)))

    event_points = event_points.sort_values("t_h")
    ax.plot(event_points["t_h"], event_points["moisture_mm"], color="#252525", lw=1.15, alpha=0.75, zorder=2)
    for segment_id, segment in event_points.groupby("adaptive_segment_id", observed=False):
        segment = segment.sort_values("t_h")
        regime = str(segment["segment_regime"].iloc[0])
        ax.plot(segment["t_h"], segment["moisture_mm"], color=REGIME_COLORS[regime], lw=2.0, zorder=3)

    label_handles = [
        patches.Patch(color=REGIME_COLORS[regime], alpha=0.55, label=SEGMENT_LABELS[regime])
        for regime in SEGMENT_ORDER
    ]
    ax.legend(handles=label_handles, loc="upper right", fontsize=6.2, ncol=1, borderaxespad=0.2)
    ax.set_xlabel("Time since SMDE start (h)")
    ax.set_ylabel("Soil water storage, S (mm)")
    start_label = event_meta["start"].strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"Observed SMDE segmentation, FAWN {int(event_meta['site_id'])}, 4 in", loc="left", fontsize=8)
    ax.text(
        0.02,
        0.05,
        start_label,
        transform=ax.transAxes,
        fontsize=6.1,
        color=COLORS["neutral_dark"],
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#D7DDE2", alpha=0.90),
    )
    add_panel_label(ax, "b")


def plot_representative_diagnosis_panel(
    ax: plt.Axes, event_meta: pd.Series, event_points: pd.DataFrame, loss_points: pd.DataFrame
) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "a", x=-0.038, y=1.125)
    ax.set_title(
        f"Observed SMDE decomposed by adaptive loss-function diagnosis, FAWN {int(event_meta['site_id'])}, {LAYER_LABELS[str(event_meta['layer'])]}, {event_meta['start'].strftime('%Y-%m-%d')}",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        pad=2,
    )

    ax_time = ax.inset_axes([0.035, 0.16, 0.435, 0.72])
    ax_loss = ax.inset_axes([0.555, 0.16, 0.405, 0.72])
    _style_segment_axis(ax_time)
    _style_segment_axis(ax_loss)

    summaries = _segment_summary(event_points)
    for row in summaries.itertuples(index=False):
        regime = str(row.segment_regime)
        ax_time.axvspan(row.start_t_h, row.end_t_h, color=REGIME_COLORS[regime], alpha=0.10, lw=0)
        ax_loss.axvspan(row.end_norm, row.start_norm, color=REGIME_COLORS[regime], alpha=0.07, lw=0)
        if row.segment_order > 0:
            ax_time.axvline(row.start_t_h, color="#7B8790", lw=0.65, ls=(0, (3, 2)))

    event_points = event_points.sort_values("t_h")
    ax_time.plot(event_points["t_h"], event_points["moisture_mm"], color="#222222", lw=1.05, alpha=0.82, zorder=2)
    for _, segment in event_points.groupby("adaptive_segment_id", observed=False):
        segment = segment.sort_values("t_h")
        regime = str(segment["segment_regime"].iloc[0])
        ax_time.plot(segment["t_h"], segment["moisture_mm"], color=REGIME_COLORS[regime], lw=1.95, zorder=3)

    for regime in SEGMENT_ORDER:
        d = loss_points[loss_points["segment_regime"] == regime].sort_values("storage_norm_mid")
        if d.empty:
            continue
        color = REGIME_COLORS[regime]
        ax_loss.scatter(
            d["storage_norm_mid"],
            d["loss_mm_h"],
            s=13,
            facecolor="white",
            edgecolor=color,
            linewidth=0.75,
            alpha=0.98,
            zorder=3,
        )
        ax_loss.plot(d["storage_norm_mid"], d["loss_mm_h"], color=color, lw=1.35, alpha=0.96, zorder=4)

    ymax = float(np.nanpercentile(loss_points["loss_mm_h"], 97) * 1.25) if not loss_points.empty else 1.0
    ax_loss.set_ylim(0, max(0.35, ymax))
    ax_loss.set_xlim(0, 1.03)
    ax_time.set_xlabel("Time since SMDE start (h)", fontsize=6.4)
    ax_time.set_ylabel("S (mm)", fontsize=6.4)
    ax_loss.set_xlabel("Normalized storage, x", fontsize=6.4)
    ax_loss.set_ylabel(r"$L=-dS/dt$ (mm h$^{-1}$)", fontsize=6.4)
    ax_time.set_title("time-domain segment boundaries", loc="left", fontsize=6.8, pad=2)
    ax_loss.set_title("loss-storage diagnostic view", loc="left", fontsize=6.8, pad=2)
    ax_loss.text(
        0.02,
        0.96,
        "wet states are at right;\npoints are interval loss rates",
        transform=ax_loss.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color=COLORS["neutral_mid"],
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.72},
    )
    ax_time.tick_params(labelsize=5.8)
    ax_loss.tick_params(labelsize=5.8)

    handles = [
        patches.Patch(color=REGIME_COLORS[regime], alpha=0.42, label=SEGMENT_LABELS[regime])
        for regime in SEGMENT_ORDER
    ]
    ax_time.legend(handles=handles, loc="upper right", fontsize=5.7, handlelength=0.9, borderaxespad=0.25)


def plot_local_segment_library(
    ax: plt.Axes, event_meta: pd.Series, segment_points: pd.DataFrame, events: pd.DataFrame
) -> None:
    site = int(event_meta["site_id"])
    layer = str(event_meta["layer"])
    event_year = int(event_meta["start"].year)
    local = segment_points[(segment_points["site_id"].astype(int) == site) & (segment_points["layer"].astype(str) == layer)].copy()
    local = local[local["segment_regime"].astype(str).isin(SEGMENT_ORDER)]
    local = local.merge(
        events[["event_id", "start"]],
        left_on="parent_event_id",
        right_on="event_id",
        how="left",
        suffixes=("", "_audit"),
    )
    local["timestamp"] = local["start"] + pd.to_timedelta(local["t_h"], unit="h")
    local = local[local["timestamp"].dt.year == event_year]
    _style_segment_axis(ax)
    if local.empty:
        ax.text(0.5, 0.5, "No local segment points found", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(local["timestamp"], local["moisture_mm"], s=2.2, color="#222222", alpha=0.16, linewidths=0, zorder=1)
        for _, segment in local.groupby("adaptive_segment_id", observed=False):
            segment = segment.sort_values("timestamp")
            regime = str(segment["segment_regime"].iloc[0])
            alpha = 0.95 if str(segment["parent_event_id"].iloc[0]) == str(event_meta["event_id"]) else 0.42
            lw = 1.35 if str(segment["parent_event_id"].iloc[0]) == str(event_meta["event_id"]) else 0.75
            ax.plot(segment["timestamp"], segment["moisture_mm"], color=REGIME_COLORS[regime], lw=lw, alpha=alpha, zorder=2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(pd.Timestamp(event_year, 1, 1), pd.Timestamp(event_year, 12, 31))
    ax.set_ylabel("Soil water storage, S (mm)")
    ax.set_xlabel(f"{event_year} local SMDE library")
    ax.set_title(f"Segmented local SMDE library, FAWN {site}, 4 in", loc="left", fontsize=8)
    ax.text(
        0.01,
        0.94,
        "colored traces = regime-diagnosed segments; black dots = segment observations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=COLORS["neutral_mid"],
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.78),
    )
    add_panel_label(ax, "c")


def read_library_coverage(events: pd.DataFrame) -> pd.DataFrame:
    all_sites = sorted(events["site_id"].dropna().astype(int).unique())
    full_index = pd.MultiIndex.from_product([all_sites, LAYER_ORDER], names=["site_id", "layer"]).to_frame(index=False)
    if not FORECAST_LIBRARY_FILE.exists():
        coverage = full_index.copy()
        coverage["regime_libraries"] = 0
        coverage["registered_segments"] = 0
        coverage["pairwise_edges"] = 0
        return coverage
    library = pd.read_csv(FORECAST_LIBRARY_FILE)
    grouped = (
        library.groupby(["site_id", "layer", "segment"], observed=False)
        .agg(
            registered_segments=("registered_segments", "max"),
            pairwise_edges=("pairwise_edges", "max"),
        )
        .reset_index()
    )
    coverage = (
        grouped.groupby(["site_id", "layer"], observed=False)
        .agg(
            regime_libraries=("segment", "nunique"),
            registered_segments=("registered_segments", "sum"),
            pairwise_edges=("pairwise_edges", "sum"),
        )
        .reset_index()
    )
    coverage = full_index.merge(coverage, on=["site_id", "layer"], how="left")
    coverage[["regime_libraries", "registered_segments", "pairwise_edges"]] = coverage[
        ["regime_libraries", "registered_segments", "pairwise_edges"]
    ].fillna(0)
    coverage["regime_libraries"] = coverage["regime_libraries"].astype(int)
    coverage["registered_segments"] = coverage["registered_segments"].astype(int)
    coverage["pairwise_edges"] = coverage["pairwise_edges"].astype(int)
    coverage["layer_label"] = coverage["layer"].map(LAYER_LABELS)
    return coverage.sort_values(["site_id", "layer"])


def build_source_tables(events: pd.DataFrame, binned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    clean_events = events[events["clean_48h"]].copy()
    regime_counts = (
        clean_events.groupby(["layer", "regime_proxy"], observed=False)
        .size()
        .reset_index(name="events")
        .query("events > 0")
    )
    regime_counts["layer_label"] = regime_counts["layer"].astype(str).map(LAYER_LABELS)
    regime_counts["regime_label"] = regime_counts["regime_proxy"].astype(str).map(EVENT_REGIME_LABELS)
    regime_counts["fraction_of_layer"] = regime_counts["events"] / regime_counts.groupby("layer", observed=False)["events"].transform("sum")

    diagnostic_summary = (
        clean_events.groupby(["layer", "regime_proxy"], observed=False)
        .agg(
            events=("event_id", "count"),
            clean_events=("clean_48h", "sum"),
            median_trim3_r2=("trim3_r2", "median"),
            median_early3_drop_share=("early3_drop_share", "median"),
            median_post3_loss_storage_corr=("post3_loss_storage_corr", "median"),
            median_post3_loss_cv=("post3_loss_cv", "median"),
            median_mean_loss_mm_h=("mean_loss_mm_h", "median"),
        )
        .reset_index()
        .query("events > 0")
    )
    diagnostic_summary["layer_label"] = diagnostic_summary["layer"].astype(str).map(LAYER_LABELS)
    diagnostic_summary["regime_label"] = diagnostic_summary["regime_proxy"].astype(str).map(EVENT_REGIME_LABELS)

    empirical_loss_primary = binned[
        (binned["layer"].astype(str) == "moisture_4in")
        & (binned["regime_proxy"].astype(str).isin(["stage-II-like", "stage-I-like", "early-transient-heavy"]))
    ].copy()
    empirical_loss_primary["layer_label"] = empirical_loss_primary["layer"].astype(str).map(LAYER_LABELS)
    empirical_loss_primary["regime_label"] = empirical_loss_primary["regime_proxy"].astype(str).map(EVENT_REGIME_LABELS)

    construction_pool = read_construction_pool()

    tables = {
        "experiment2_regime_composition_by_layer": regime_counts,
        "experiment2_event_diagnostic_summary": diagnostic_summary,
        "experiment2_empirical_loss_storage_4in": empirical_loss_primary,
        "experiment2_adaptive_regime_construction_pool": construction_pool,
    }
    for name, table in tables.items():
        table.to_csv(SOURCE / f"{name}.csv", index=False)
    return tables


def _fingerprint_curve(regime: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.08, 0.92, 120)
    if regime == "early-transient-heavy":
        y = 0.24 + 0.08 * x + 0.52 / (1 + np.exp(-18 * (x - 0.68)))
    elif regime == "stage-I-like":
        y = np.full_like(x, 0.52) + 0.015 * np.sin(np.linspace(0, np.pi, len(x)))
    else:
        y = 0.17 + 0.62 * np.power(x, 1.35)
    return x, np.clip(y, 0.08, 0.92)


def _event_shape(regime: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.10, 0.92, 90)
    if regime == "early-transient-heavy":
        s = 0.88 - 0.46 * (1 - np.exp(-5.0 * (t - 0.10))) - 0.08 * (t - 0.10)
    elif regime == "stage-I-like":
        s = 0.83 - 0.45 * (t - 0.10)
    else:
        s = 0.82 - 0.55 * (1 - np.exp(-1.75 * (t - 0.10)))
    return t, np.clip(s, 0.12, 0.90)


def plot_empirical_loss_storage(ax: plt.Axes, empirical: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_title("Diagnostic fingerprints used for regime labels", loc="left", fontsize=8, pad=5)
    regimes = [
        ("early-transient-heavy", "Early transient", "rapid wet-end\nloss"),
        ("stage-I-like", "Stage I-like", "near-constant\nloss"),
        ("stage-II-like", "Stage II-like", "storage-limited\nloss"),
    ]
    lefts = [0.02, 0.345, 0.67]
    for left, (regime, title, note) in zip(lefts, regimes):
        card = ax.inset_axes([left, 0.05, 0.305, 0.82])
        color = REGIME_COLORS[regime]
        card.set_xlim(0, 1)
        card.set_ylim(0, 1)
        card.set_xticks([])
        card.set_yticks([])
        for spine in card.spines.values():
            spine.set_visible(True)
            spine.set_color("#D7DDE2")
            spine.set_linewidth(0.65)
        card.set_facecolor("#FBFCFD")

        card.add_patch(
            patches.Rectangle((0, 0.82), 1, 0.18, transform=card.transAxes, color=color, alpha=0.14, lw=0)
        )
        card.text(0.5, 0.91, title, transform=card.transAxes, ha="center", va="center", fontsize=6.6, fontweight="bold", color=COLORS["neutral_dark"])
        card.text(0.5, 0.785, note, transform=card.transAxes, ha="center", va="top", fontsize=5.6, color=COLORS["neutral_mid"], linespacing=0.95)

        x, y = _fingerprint_curve(regime)
        card.plot(x, 0.13 + 0.42 * y, color=color, lw=1.8)
        card.fill_between(x, 0.13 + 0.42 * np.maximum(y - 0.08, 0), 0.13 + 0.42 * np.minimum(y + 0.08, 1), color=color, alpha=0.08, lw=0)

        t, s = _event_shape(regime)
        card.plot(t, 0.48 + 0.32 * s, color="#333333", lw=1.15)
        card.text(0.10, 0.78, "S(t)", transform=card.transAxes, fontsize=5.4, color="#333333")
        card.text(0.10, 0.43, "L(S)", transform=card.transAxes, fontsize=5.4, color=color)

        card.annotate("", xy=(0.92, 0.10), xytext=(0.10, 0.10), arrowprops=dict(arrowstyle="->", lw=0.6, color="#777777"))
        card.text(0.10, 0.02, "dry", transform=card.transAxes, ha="left", va="bottom", fontsize=5.2, color="#777777")
        card.text(0.92, 0.02, "wet", transform=card.transAxes, ha="right", va="bottom", fontsize=5.2, color="#777777")
        if left == lefts[0]:
            card.text(0.02, 0.31, "loss rate", rotation=90, transform=card.transAxes, ha="left", va="center", fontsize=5.2, color="#777777")
        card.text(0.50, 0.02, "storage", transform=card.transAxes, ha="center", va="bottom", fontsize=5.2, color="#777777")
    add_panel_label(ax, "a")


def plot_regime_composition(ax: plt.Axes, regime_counts: pd.DataFrame) -> None:
    pivot = (
        regime_counts.pivot(index="layer", columns="regime_proxy", values="fraction_of_layer")
        .reindex(LAYER_ORDER)
        .reindex(columns=EVENT_REGIME_ORDER)
        .fillna(0)
    )
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for regime in EVENT_REGIME_ORDER:
        vals = pivot[regime].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=REGIME_COLORS[regime],
            edgecolor="none",
            linewidth=0,
            width=0.72,
            label=EVENT_REGIME_LABELS[regime],
        )
        bottom += vals
    ax.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of clean rainfall-associated SMDEs")
    ax.set_title("Whole-event diagnostic composition", loc="left", fontsize=8)
    ax.legend(ncol=2, fontsize=6.0, handlelength=1.0, columnspacing=0.8, loc="upper right")

    stage2 = pivot["stage-II-like"].to_numpy(dtype=float)
    mixed = pivot["mixed_or_uncertain"].to_numpy(dtype=float)
    for i, (s2, mix) in enumerate(zip(stage2, mixed)):
        ax.text(i, s2 / 2, f"{s2:.0%}", ha="center", va="center", fontsize=6.0, color="white")
        early = pivot["early-transient-heavy"].to_numpy(dtype=float)[i]
        early_bottom = s2 + pivot["stage-I-like"].to_numpy(dtype=float)[i]
        if early > 0:
            ax.text(i, early_bottom + early / 2, f"{early:.0%}", ha="center", va="center", fontsize=4.8, color=COLORS["neutral_dark"])
        ax.text(i, 1 - mix / 2, f"{mix:.0%}", ha="center", va="center", fontsize=6.0, color=COLORS["neutral_dark"])
    add_panel_label(ax, "a", x=-0.095, y=1.10)


def plot_construction_pool(ax: plt.Axes, pool: pd.DataFrame) -> None:
    pivot = (
        pool.pivot(index="layer", columns="segment", values="segments")
        .reindex(LAYER_ORDER)
        .reindex(columns=SEGMENT_ORDER)
        .fillna(0)
    )
    totals = pivot.sum(axis=1).to_numpy(dtype=float)
    fractions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    x = np.arange(len(LAYER_ORDER))
    bottom = np.zeros(len(LAYER_ORDER), dtype=float)
    for segment in SEGMENT_ORDER:
        vals = fractions[segment].to_numpy(dtype=float)
        counts = pivot[segment].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.70,
            color=REGIME_COLORS[segment],
            edgecolor="none",
            linewidth=0,
            label=SEGMENT_LABELS[segment],
        )
        for xi, val, bot, count in zip(x, vals, bottom, counts):
            if val <= 0:
                continue
            label = f"{val:.0%}\n{int(count):,}"
            color = "white" if segment == "stageII_like" else COLORS["neutral_dark"]
            ax.text(
                xi,
                bot + val / 2,
                label,
                ha="center",
                va="center",
                fontsize=5.5 if val >= 0.12 else 4.8,
                color=color,
                linespacing=0.88,
            )
        bottom += vals
    for xi, total in zip(x, totals):
        ax.text(xi, 1.025, f"n={int(total):,}", ha="center", va="bottom", fontsize=5.7, color=COLORS["neutral_mid"])
    ax.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of regime-labeled segments")
    ax.set_title("Segment-level CSR construction pools", loc="left", fontsize=8, pad=8)
    ax.grid(False)
    ax.legend(
        ncol=3,
        fontsize=6.0,
        handlelength=1.0,
        columnspacing=0.75,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.16),
        borderaxespad=0.0,
    )
    add_panel_label(ax, "b", x=-0.045, y=1.16)


def plot_library_coverage(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    heat = (
        coverage.pivot_table(index="site_id", columns="layer", values="regime_libraries", observed=False)
        .reindex(columns=LAYER_ORDER)
        .sort_index()
    )
    arr = heat.to_numpy(dtype=float)
    cmap = ListedColormap(["#F2F2F2", "#DDE8F1", "#7FAED2", "#234E70"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
    y_labels = [str(int(s)) if i % 3 == 0 else "" for i, s in enumerate(heat.index)]
    ax.set_yticks(np.arange(len(heat.index)), y_labels, fontsize=5.8)
    ax.set_xticks(np.arange(len(LAYER_ORDER)), [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax.set_xlabel("Sensor depth")
    ax.set_ylabel("FAWN station")
    ax.set_title("Local regime-specific CSR libraries fitted per station-layer", loc="left", fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = int(arr[i, j]) if np.isfinite(arr[i, j]) else 0
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=4.8, color="white" if val == 3 else COLORS["neutral_dark"])
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=4.3, color="#AAAAAA")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015, ticks=[0, 1, 2, 3])
    cbar.set_label("local regime libraries")
    add_panel_label(ax, "d")


def write_report(events: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    clean_events = events[events["clean_48h"]].copy()
    total = len(clean_events)
    overall = (
        clean_events.groupby("regime_proxy", observed=False)
        .size()
        .reset_index(name="events")
        .query("events > 0")
    )
    overall["fraction"] = overall["events"] / total
    clean_interpretable = int(events["clean_interpretable_48h"].sum())
    pool = tables["experiment2_adaptive_regime_construction_pool"]

    lines = [
        "# Experiment 2: Loss-function regime diagnosis and CSR support",
        "",
        "Prepared: 2026-06-06",
        "",
        "## Purpose",
        "",
        "This figure now reports statistical diagnosis results only: whole-event regime composition for clean rainfall-associated SMDEs and segment-level construction pools for regime-specific CSR.",
        "",
        "## Core result",
        "",
        f"- Clean rainfall-associated SMDEs evaluated: {total:,}.",
        f"- Clean rainfall-associated events assigned to an interpretable event-level regime: {clean_interpretable:,} ({clean_interpretable / total:.1%}).",
        f"- Adaptive segment construction pool: {int(pool['segments'].sum()):,} regime-labeled segments from {int(pool['parent_events'].sum()):,} parent-event memberships across depth-regime combinations.",
        "",
        "## Overall regime composition",
        "",
        "| Regime proxy | Events | Fraction |",
        "|---|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(f"| {EVENT_REGIME_LABELS[str(row.regime_proxy)]} | {row.events:,} | {row.fraction:.1%} |")
    lines.extend(
        [
            "",
            "## Adaptive construction pool",
            "",
            pool.to_markdown(index=False),
            "",
            "## Draft figure legend",
            "",
            "Figure X. Loss-function diagnosis before CSR construction. (a) Whole-event diagnostic composition of clean rainfall-associated SMDEs by sensor depth. (b) Segment-level construction pools admitted to regime-specific CSR by sensor depth. Regime labels are diagnostic proxies inferred from soil moisture behavior and do not directly partition individual fluxes.",
            "",
            "## Interpretation boundary",
            "",
            "Regime labels are diagnostic proxies inferred from soil-moisture behavior. They organize the event library and forecast operator; they do not directly partition measured drainage, runoff, evaporation, transpiration, or redistribution fluxes.",
        ]
    )
    (OUT / "experiment2_loss_function_regime_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_figure() -> None:
    ensure_out()
    events = read_events()
    binned = read_binned()
    tables = build_source_tables(events, binned)

    fig = plt.figure(figsize=(7.4, 4.75), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.50)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    plot_regime_composition(ax_a, tables["experiment2_regime_composition_by_layer"])
    plot_construction_pool(ax_b, tables["experiment2_adaptive_regime_construction_pool"])

    fig.suptitle(
        "Loss-function diagnosis separates whole-event ambiguity from segment-level CSR support",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral_dark"],
    )
    fig.text(
        0.01,
        0.955,
        "Clean rainfall-associated SMDEs often remain mixed at the event level, while adaptive segmentation extracts interpretable regime-specific construction pools.",
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["neutral_mid"],
    )
    fig.subplots_adjust(top=0.89)
    save_pub(fig, "fig_experiment2_loss_function_regime_diagnosis")
    plt.close(fig)
    write_report(events, tables)


if __name__ == "__main__":
    build_figure()
