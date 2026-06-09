from __future__ import annotations

from pathlib import Path

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

try:
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
except ModuleNotFoundError:
    ccrs = None
    shpreader = None


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
OUT = ANALYSIS / "study_data_overview"
SOURCE = OUT / "source_data"
SUBMISSION_FIGS = ROOT / "figures"

STATUS = ANALYSIS / "fawn_full_smde_audit" / "processing_status_by_site.csv"
EVENTS = ANALYSIS / "fawn_full_smde_audit" / "full_smde_event_audit.csv"
SUMMARY_YEAR_LAYER = ANALYSIS / "fawn_full_smde_audit" / "full_smde_summary_by_year_layer.csv"
DB_COORDS = OUT / "fawn_station_coordinates_db_join.csv"
SOIL_PROFILE_IMAGE = SOURCE / "soil_profile_usda_wikimedia.jpg"
EXTERNAL_2026_EVENTS = ANALYSIS / "experiment4_2026_external_forecast" / "source_data" / "external_2026_detected_smde_events.csv"
EXTERNAL_2026_EVENTS_FALLBACK = (
    ANALYSIS / "experiment4c_all_train_rate_forecast" / "source_data" / "external_2026_detected_smde_events.csv"
)
SOIL_2026_META = ANALYSIS / "fawn_db_export" / "data" / "soil_moisture_2026.json"
WX_2026_META = ANALYSIS / "fawn_db_export" / "data" / "wx_selected_2026.json"

LAYER_ORDER = ["moisture_4in", "moisture_8in", "moisture_12in", "moisture_16in", "moisture_20in"]
LAYER_LABELS = {
    "moisture_4in": "4 in",
    "moisture_8in": "8 in",
    "moisture_12in": "12 in",
    "moisture_16in": "16 in",
    "moisture_20in": "20 in",
}
LAYER_COLORS = {
    "moisture_4in": "#4E7E9E",
    "moisture_8in": "#6FA38A",
    "moisture_12in": "#C78D4B",
    "moisture_16in": "#8E6AA9",
    "moisture_20in": "#777777",
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
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIGS.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coords = pd.read_csv(DB_COORDS)
    coords["Latitude_num"] = pd.to_numeric(coords["Latitude_num"], errors="coerce")
    coords["Longitude_num"] = pd.to_numeric(coords["Longitude_num"], errors="coerce")
    coords.loc[coords["Longitude_num"].between(70, 90), "Longitude_num"] *= -1
    events = pd.read_csv(EVENTS, parse_dates=["start", "end"])
    external_path = EXTERNAL_2026_EVENTS if EXTERNAL_2026_EVENTS.exists() else EXTERNAL_2026_EVENTS_FALLBACK
    external = pd.read_csv(external_path, parse_dates=["start", "end"])
    year_layer = pd.read_csv(SUMMARY_YEAR_LAYER)
    events["layer"] = events["layer"].astype(str)
    external["layer"] = external["layer"].astype(str)
    year_layer["layer"] = year_layer["layer"].astype(str)
    return coords, events, external, year_layer


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _label_bbox(
    x: float,
    y: float,
    label: str,
    dx_pt: float,
    dy_pt: float,
    dpi: float,
    fontsize: float,
) -> tuple[float, float, float, float]:
    scale = dpi / 72.0
    px = x + dx_pt * scale
    py = y + dy_pt * scale
    width = max(8.0, len(label) * fontsize * scale * 0.62)
    height = fontsize * scale * 1.08

    if dx_pt > 0:
        x0, x1 = px, px + width
    elif dx_pt < 0:
        x0, x1 = px - width, px
    else:
        x0, x1 = px - width / 2, px + width / 2

    if dy_pt > 0:
        y0, y1 = py, py + height
    elif dy_pt < 0:
        y0, y1 = py - height, py
    else:
        y0, y1 = py - height / 2, py + height / 2

    pad = 1.0
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _station_label_offsets(ax: plt.Axes, plot_df: pd.DataFrame, fontsize: float) -> dict[int, tuple[float, float]]:
    transform = ccrs.PlateCarree()._as_mpl_transform(ax) if ccrs is not None else ax.transData
    xy = transform.transform(plot_df[["Longitude_num", "Latitude_num"]].to_numpy(float))
    dpi = ax.figure.dpi
    axes_box = ax.get_window_extent()
    candidate_offsets = [
        (4.0, 3.0),
        (-4.0, 3.0),
        (4.0, -4.0),
        (-4.0, -4.0),
        (0.0, 5.5),
        (0.0, -5.5),
        (6.0, 0.0),
        (-6.0, 0.0),
        (6.0, 5.0),
        (-6.0, 5.0),
        (6.0, -5.0),
        (-6.0, -5.0),
    ]
    point_boxes = [(x - 3.0, y - 3.0, x + 3.0, y + 3.0) for x, y in xy]
    labels: dict[int, tuple[float, float]] = {}
    placed_boxes: list[tuple[float, float, float, float]] = []

    if len(xy) > 1:
        distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        order = np.argsort(distances.min(axis=1))
    else:
        order = np.arange(len(xy))

    for idx in order:
        row = plot_df.iloc[int(idx)]
        sid = int(row["site_id"])
        label = str(sid)
        x, y = xy[int(idx)]
        best_offset = candidate_offsets[0]
        best_score = np.inf
        for dx_pt, dy_pt in candidate_offsets:
            box = _label_bbox(x, y, label, dx_pt, dy_pt, dpi, fontsize)
            overlap = sum(_bbox_overlap(box, other) for other in placed_boxes)
            point_overlap = sum(_bbox_overlap(box, other) for other in point_boxes)
            outside = (
                max(0.0, axes_box.x0 - box[0])
                + max(0.0, box[2] - axes_box.x1)
                + max(0.0, axes_box.y0 - box[1])
                + max(0.0, box[3] - axes_box.y1)
            )
            score = overlap * 12.0 + point_overlap * 0.5 + outside * 60.0 + abs(dx_pt) + abs(dy_pt)
            if score < best_score:
                best_score = score
                best_offset = (dx_pt, dy_pt)
        labels[sid] = best_offset
        placed_boxes.append(_label_bbox(x, y, label, best_offset[0], best_offset[1], dpi, fontsize))
    return labels


def read_meta_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("rows", 0))


def florida_geometry():
    if shpreader is None:
        return None
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
    )
    for rec in reader.records():
        attrs = rec.attributes
        name = str(attrs.get("name", ""))
        adm0 = str(attrs.get("admin", attrs.get("adm0_name", "")))
        if name.lower() == "florida" and adm0.lower() == "united states of america":
            return rec.geometry
    return None


def panel_map(ax: plt.Axes, coords: pd.DataFrame) -> None:
    geom = florida_geometry()
    if geom is not None:
        ax.add_geometries([geom], ccrs.PlateCarree(), facecolor="#F7F8F6", edgecolor="#333333", linewidth=0.75)
    if ccrs is not None:
        ax.set_extent([-87.8, -79.6, 24.3, 31.2], crs=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.35,
            color="#C9CED3",
            alpha=0.9,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 5.7, "color": "#555555"}
        gl.ylabel_style = {"size": 5.7, "color": "#555555"}
        map_transform = {"transform": ccrs.PlateCarree()}
    else:
        florida_lon = [-87.6, -86.4, -85.1, -83.3, -81.6, -80.2, -80.0, -80.4, -81.2, -82.0, -82.6, -83.0, -84.0, -85.3, -86.7, -87.6]
        florida_lat = [30.99, 30.6, 30.2, 29.5, 28.4, 26.0, 25.2, 24.8, 24.6, 25.2, 27.0, 28.8, 30.0, 30.6, 30.95, 30.99]
        ax.fill(florida_lon, florida_lat, color="#F7F8F6", edgecolor="#333333", linewidth=0.75, zorder=1)
        ax.set_xlim(-87.8, -79.6)
        ax.set_ylim(24.3, 31.2)
        ax.set_xlabel("Longitude", fontsize=5.9)
        ax.set_ylabel("Latitude", fontsize=5.9)
        ax.grid(True, linewidth=0.35, color="#C9CED3", alpha=0.9)
        ax.tick_params(labelsize=5.5)
        map_transform = {}

    plot_df = coords.dropna(subset=["Latitude_num", "Longitude_num"]).copy()
    with_events = plot_df[plot_df["events"].gt(0)].copy()
    no_events = plot_df[plot_df["events"].le(0)].copy()
    sizes = 5 + 10 * np.sqrt(with_events["events"].to_numpy(float) / with_events["events"].max())
    sc = ax.scatter(
        with_events["Longitude_num"],
        with_events["Latitude_num"],
        c=with_events["events"],
        s=sizes,
        cmap="YlGnBu",
        norm=LogNorm(vmin=max(1, with_events["events"].min()), vmax=with_events["events"].max()),
        edgecolors="#222222",
        linewidths=0.18,
        zorder=4,
        **map_transform,
    )
    if not no_events.empty:
        ax.scatter(
            no_events["Longitude_num"],
            no_events["Latitude_num"],
            s=7,
            facecolors="white",
            edgecolors="#777777",
            linewidths=0.35,
            zorder=5,
            **map_transform,
        )

    label_fontsize = 4.3
    label_offsets = _station_label_offsets(ax, plot_df, label_fontsize)
    for _, r in plot_df.sort_values("site_id").iterrows():
        sid = int(r["site_id"])
        dx, dy = label_offsets.get(sid, (4.0, 3.0))
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
        ax.annotate(
            str(sid),
            xy=(float(r["Longitude_num"]), float(r["Latitude_num"])),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax) if ccrs is not None else "data",
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=label_fontsize,
            color="#222222",
            zorder=8,
            path_effects=[pe.withStroke(linewidth=1.15, foreground="white")],
        )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.042, pad=0.015)
    cbar.ax.set_title("SMDEs", fontsize=5.7, pad=3)
    cbar.ax.tick_params(labelsize=5.3, width=0.5)
    ax.set_title("a  FAWN stations used for SMDE analysis", loc="left", fontweight="bold", fontsize=8.0)


def panel_sensor_schematic(ax: plt.Axes, coords: pd.DataFrame, events: pd.DataFrame, external: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.text(0.00, 0.98, "b  Soil profile and sensor depths", fontweight="bold", fontsize=8.0, va="top")

    img = plt.imread(SOIL_PROFILE_IMAGE)
    left, bottom, width, height = 0.14, 0.08, 0.52, 0.78
    max_depth_in = 48
    depth_top_y = bottom + height * 0.74
    depth_bottom_y = bottom + height * 0.09

    def y_from_depth(depth_in: float) -> float:
        return depth_top_y - (depth_top_y - depth_bottom_y) * (depth_in / max_depth_in)

    ax.imshow(img, extent=[left, left + width, bottom, bottom + height], aspect="auto", interpolation="lanczos", zorder=1)
    ax.text(0.73, bottom + height - 0.02, "Sensor depth", ha="left", va="top", fontsize=6.1, color="#555555")
    for i, depth in enumerate([4, 8, 12, 16, 20]):
        y = y_from_depth(depth)
        layer = LAYER_ORDER[i]
        x_probe = left + width * 0.74
        ax.scatter([x_probe], [y], s=28, color=LAYER_COLORS[layer], edgecolor="white", linewidth=0.75, zorder=4)
        ax.plot([x_probe + 0.012, 0.715], [y, y], color="#AEB7BD", lw=0.62, zorder=3)
        ax.text(0.735, y, LAYER_LABELS[layer], ha="left", va="center", fontsize=6.3, color="#333333")


def panel_year_depth(ax: plt.Axes, year_layer: pd.DataFrame, external: pd.DataFrame) -> None:
    pivot = (
        year_layer.pivot_table(index="start_year", columns="layer", values="events", aggfunc="sum")
        .reindex(columns=LAYER_ORDER)
        .fillna(0)
        .astype(int)
    )
    external_counts = external.groupby("layer", observed=True).size().reindex(LAYER_ORDER).fillna(0).astype(int)
    pivot.loc[2026, LAYER_ORDER] = external_counts.to_numpy(int)
    pivot = pivot.sort_index()
    bottoms = np.zeros(len(pivot), dtype=float)
    years = pivot.index.astype(int).to_numpy()
    x = np.arange(len(years))
    for layer in LAYER_ORDER:
        values = pivot[layer].to_numpy(float)
        bars = ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.62,
            color=LAYER_COLORS[layer],
            edgecolor="none",
            lw=0,
            label=LAYER_LABELS[layer],
        )
        if 2026 in years:
            idx = int(np.where(years == 2026)[0][0])
            bars[idx].set_hatch("///")
            bars[idx].set_edgecolor("#333333")
            bars[idx].set_linewidth(0.25)
        bottoms += values
    for xi, total in zip(x, bottoms):
        ax.text(xi, total + max(bottoms) * 0.018, f"{int(total):,}", ha="center", va="bottom", fontsize=5.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}*" if y == 2026 else str(y) for y in years])
    ax.set_ylabel("Detected SMDEs")
    ax.set_xlabel("Start year")
    ax.set_title("c  SMDE counts by year and sensor depth", loc="left", fontweight="bold", fontsize=8.0)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=5.2, ncol=1, frameon=True, framealpha=0.92, borderpad=0.25)
    if 2026 in years:
        idx = int(np.where(years == 2026)[0][0])
        ax.text(
            idx,
            -0.17,
            "validation set",
            ha="center",
            va="top",
            fontsize=5.8,
            color="#333333",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    pivot.reset_index().to_csv(SOURCE / "figure1_panel_c_year_depth_counts.csv", index=False)


def _boxplot_by_layer(ax: plt.Axes, data: pd.DataFrame, value_col: str, title: str, xlabel: str, clip_value: float) -> None:
    box_data = []
    labels = []
    for layer in LAYER_ORDER:
        vals = pd.to_numeric(data.loc[data["layer"].eq(layer), value_col], errors="coerce").dropna()
        vals = vals[vals > 0].clip(upper=clip_value)
        if len(vals) > 0:
            box_data.append(vals.to_numpy(float))
            labels.append(LAYER_LABELS[layer])
    bp = ax.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=0.9),
        boxprops=dict(linewidth=0.55, color="#555555"),
        whiskerprops=dict(linewidth=0.55, color="#555555"),
        capprops=dict(linewidth=0.55, color="#555555"),
    )
    for patch, layer in zip(bp["boxes"], LAYER_ORDER):
        patch.set_facecolor(LAYER_COLORS[layer])
        patch.set_alpha(0.72)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=6.6, pad=3)
    ax.grid(axis="x", color="#E8EDF1", lw=0.45)


def panel_event_properties(ax_duration: plt.Axes, ax_drop: plt.Axes, events: pd.DataFrame) -> None:
    use = events.dropna(subset=["duration_h", "total_drop_mm"]).copy()
    use = use[(use["duration_h"] > 0) & (use["total_drop_mm"] > 0)]
    ax_duration.text(
        0.0,
        1.15,
        "d  SMDE duration and total soil water loss by depth",
        transform=ax_duration.transAxes,
        fontweight="bold",
        fontsize=8.0,
        va="bottom",
    )
    _boxplot_by_layer(
        ax_duration,
        use,
        "duration_h",
        "Duration",
        "Duration (h)",
        clip_value=72,
    )
    _boxplot_by_layer(
        ax_drop,
        use,
        "total_drop_mm",
        "Total soil water loss",
        "Total soil water loss (mm)",
        clip_value=8,
    )
    ax_drop.set_yticklabels([])
    ax_drop.set_ylabel("")
    use[["site_id", "event_id", "layer", "start", "duration_h", "total_drop_mm", "audit_class", "regime_proxy"]].to_csv(
        SOURCE / "figure1_panel_d_duration_drop_points.csv", index=False
    )


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(SUBMISSION_FIGS / "Figure_1_study_area_and_data_overview.pdf", bbox_inches="tight")


def main() -> None:
    ensure_dirs()
    coords, events, external, year_layer = load_data()
    coords.to_csv(SOURCE / "figure1_panel_a_station_coordinates.csv", index=False)
    events.groupby(["site_id", "layer"], observed=True).size().rename("events").reset_index().to_csv(
        SOURCE / "figure1_station_layer_event_counts.csv", index=False
    )

    fig = plt.figure(figsize=(7.2, 6.55))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 0.95],
        height_ratios=[1.10, 0.92],
        left=0.07,
        right=0.97,
        bottom=0.08,
        top=0.90,
        wspace=0.30,
        hspace=0.34,
    )
    map_subplot_kwargs = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    ax_map = fig.add_subplot(gs[0, 0], **map_subplot_kwargs)
    ax_schematic = fig.add_subplot(gs[0, 1])
    ax_year = fig.add_subplot(gs[1, 0])
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.26)
    ax_dur = fig.add_subplot(gs_d[0, 0])
    ax_drop = fig.add_subplot(gs_d[0, 1])

    panel_map(ax_map, coords)
    panel_sensor_schematic(ax_schematic, coords, events, external)
    panel_year_depth(ax_year, year_layer, external)
    panel_event_properties(ax_dur, ax_drop, events)

    fig.suptitle("FAWN soil moisture data and SMDE library", x=0.02, y=0.985, ha="left", fontsize=10.3, fontweight="bold")
    save_pub(fig, "fig1_study_area_and_data_overview")
    plt.close(fig)
    print(f"Wrote study/data overview figure to {OUT}")


if __name__ == "__main__":
    main()
