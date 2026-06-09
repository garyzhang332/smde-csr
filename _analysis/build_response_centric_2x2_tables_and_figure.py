from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_analysis"
OUT = ANALYSIS / "response_centric_2x2"
SOURCE = OUT / "source_data"

SUMMARY_FILE = OUT / "SMDE_CSR_response_centric_2x2_preliminary_summary.csv"
DELTA_FILE = OUT / "SMDE_CSR_response_centric_2x2_preliminary_deltas.csv"
LIBRARY_SUMMARY_FILE = OUT / "SMDE_CSR_response_library_2x2_sensitivity_summary.csv"
LIBRARY_DELTA_FILE = OUT / "SMDE_CSR_response_library_2x2_sensitivity_deltas.csv"
STATIC_THRESHOLD_SUMMARY_FILE = OUT / "SMDE_CSR_static_threshold_2x2_summary.csv"
STATIC_THRESHOLD_DELTA_FILE = OUT / "SMDE_CSR_static_threshold_2x2_deltas.csv"
STATIC_THRESHOLD_LIBRARY_SUMMARY_FILE = OUT / "SMDE_CSR_static_threshold_response_library_sensitivity_summary.csv"
STATIC_THRESHOLD_LIBRARY_DELTA_FILE = OUT / "SMDE_CSR_static_threshold_response_library_sensitivity_deltas.csv"

COLORS = {
    "FAWN rainfall-only": "#4C78A8",
    "On-farm managed": "#D55E00",
    "static": "#777777",
    "response": "#1B9E77",
    "grid": "#D8D8D8",
    "text": "#222222",
}


def ensure_out() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)


def build_manuscript_table(summary: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (setting, horizon), group in summary.groupby(["data_setting", "horizon_h"], sort=False):
        static = group[group["model_family"].eq("static_or_less_local")].iloc[0]
        response = group[group["model_family"].eq("response_centric")].iloc[0]
        drow = delta[(delta["data_setting"].eq(setting)) & (delta["horizon_h"].eq(horizon))].iloc[0]
        rows.append(
            {
                "data_setting": setting,
                "horizon_h": int(horizon),
                "n_forecasts": int(response["forecasts"]),
                "n_events": int(response["events"]),
                "static_model": static["model_name"],
                "response_model": response["model_name"],
                "static_r2": float(static["r2"]),
                "response_r2": float(response["r2"]),
                "delta_r2": float(drow["delta_r2_response_minus_static"]),
                "static_ccc": float(static["ccc"]),
                "response_ccc": float(response["ccc"]),
                "delta_ccc": float(drow["delta_ccc_response_minus_static"]),
                "rmse_ratio_response_over_static": float(drow["rmse_ratio_response_over_static"]),
                "unit_label": response["unit_label"],
                "status": response["status"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["horizon_h", "data_setting"]).reset_index(drop=True)


def build_compact_delta_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table[
        [
            "data_setting",
            "horizon_h",
            "n_forecasts",
            "n_events",
            "static_r2",
            "response_r2",
            "delta_r2",
            "static_ccc",
            "response_ccc",
            "delta_ccc",
            "rmse_ratio_response_over_static",
        ]
    ].copy()
    out["static_r2"] = out["static_r2"].round(6)
    out["response_r2"] = out["response_r2"].round(6)
    out["delta_r2"] = out["delta_r2"].round(6)
    out["static_ccc"] = out["static_ccc"].round(6)
    out["response_ccc"] = out["response_ccc"].round(6)
    out["delta_ccc"] = out["delta_ccc"].round(6)
    out["rmse_ratio_response_over_static"] = out["rmse_ratio_response_over_static"].round(3)
    return out


def build_pair_table(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    *,
    evidence_tier: str,
    recommended_location: str,
    static_family: str,
    delta_r2_col: str,
    delta_ccc_col: str,
    rmse_ratio_col: str,
    interpretation_note: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (setting, horizon), group in summary.groupby(["data_setting", "horizon_h"], sort=False):
        static_rows = group[group["model_family"].eq(static_family)]
        response_rows = group[group["model_family"].eq("response_centric")]
        delta_rows = delta[(delta["data_setting"].eq(setting)) & (delta["horizon_h"].eq(horizon))]
        if static_rows.empty or response_rows.empty or delta_rows.empty:
            continue
        static = static_rows.iloc[0]
        response = response_rows.iloc[0]
        drow = delta_rows.iloc[0]
        rows.append(
            {
                "evidence_tier": evidence_tier,
                "recommended_location": recommended_location,
                "data_setting": setting,
                "horizon_h": int(horizon),
                "n_forecasts": int(response["forecasts"]),
                "n_events": int(response["events"]),
                "static_family": static_family,
                "static_model": static["model_name"],
                "response_model": response["model_name"],
                "static_r2": float(static["r2"]),
                "response_r2": float(response["r2"]),
                "delta_r2": float(drow[delta_r2_col]),
                "static_ccc": float(static["ccc"]),
                "response_ccc": float(response["ccc"]),
                "delta_ccc": float(drow[delta_ccc_col]),
                "rmse_ratio_response_over_static": float(drow[rmse_ratio_col]),
                "unit_label": response["unit_label"],
                "interpretation_note": interpretation_note,
            }
        )
    return pd.DataFrame(rows).sort_values(["evidence_tier", "horizon_h", "data_setting"]).reset_index(drop=True)


def _format_integrated_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ["static_r2", "response_r2", "delta_r2", "static_ccc", "response_ccc", "delta_ccc"]:
        out[col] = out[col].astype(float).round(6)
    out["rmse_ratio_response_over_static"] = out["rmse_ratio_response_over_static"].astype(float).round(3)
    return out


def build_integrated_evidence_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = build_pair_table(
        pd.read_csv(SUMMARY_FILE),
        pd.read_csv(DELTA_FILE),
        evidence_tier="1_main_conservative_static_recession",
        recommended_location="Main Results",
        static_family="static_or_less_local",
        delta_r2_col="delta_r2_response_minus_static",
        delta_ccc_col="delta_ccc_response_minus_static",
        rmse_ratio_col="rmse_ratio_response_over_static",
        interpretation_note="Best current interaction evidence: FAWN gains are tiny; on-farm gains increase at longer horizons.",
    )
    library = build_pair_table(
        pd.read_csv(LIBRARY_SUMMARY_FILE),
        pd.read_csv(LIBRARY_DELTA_FILE),
        evidence_tier="2_sensitivity_response_library_analog",
        recommended_location="Supplement or sensitivity paragraph",
        static_family="static_or_less_local",
        delta_r2_col="delta_r2_response_minus_static",
        delta_ccc_col="delta_ccc_response_minus_static",
        rmse_ratio_col="rmse_ratio_response_over_static",
        interpretation_note="Response-memory sensitivity closer to CSR-library logic; not full on-farm CSR registration.",
    )
    threshold = build_pair_table(
        pd.read_csv(STATIC_THRESHOLD_SUMMARY_FILE),
        pd.read_csv(STATIC_THRESHOLD_DELTA_FILE),
        evidence_tier="3_lower_bound_static_threshold",
        recommended_location="Supplement or short lower-bound paragraph",
        static_family="property_centric_static_threshold",
        delta_r2_col="delta_r2_response_minus_static_threshold",
        delta_ccc_col="delta_ccc_response_minus_static_threshold",
        rmse_ratio_col="rmse_ratio_response_over_static_threshold",
        interpretation_note="Tests insufficiency of fixed normalized storage; too weak in FAWN to be main interaction evidence.",
    )
    threshold_library = build_pair_table(
        pd.read_csv(STATIC_THRESHOLD_LIBRARY_SUMMARY_FILE),
        pd.read_csv(STATIC_THRESHOLD_LIBRARY_DELTA_FILE),
        evidence_tier="4_lower_bound_static_threshold_response_library",
        recommended_location="Supplement only",
        static_family="property_centric_static_threshold",
        delta_r2_col="delta_r2_response_minus_static_threshold",
        delta_ccc_col="delta_ccc_response_minus_static_threshold",
        rmse_ratio_col="rmse_ratio_response_over_static_threshold",
        interpretation_note="Combines property-centric lower bound with response-library analog sensitivity.",
    )
    integrated = pd.concat([main, library, threshold, threshold_library], ignore_index=True)
    compact = _format_integrated_table(integrated)
    hierarchy = pd.DataFrame(
        [
            {
                "evidence_tier": "1_main_conservative_static_recession",
                "recommended_location": "Main Results",
                "source_files": f"{SUMMARY_FILE.name}; {DELTA_FILE.name}",
                "role": "Primary 2 x 2 theory test.",
                "caution": "On-farm model is response-centric SMDE, not full on-farm CSR registration.",
            },
            {
                "evidence_tier": "2_sensitivity_response_library_analog",
                "recommended_location": "Supplement or sensitivity paragraph",
                "source_files": f"{LIBRARY_SUMMARY_FILE.name}; {LIBRARY_DELTA_FILE.name}",
                "role": "Tests response-memory/CSR-library direction.",
                "caution": "Nearest-neighbor response analog, not full CSR registration.",
            },
            {
                "evidence_tier": "3_lower_bound_static_threshold",
                "recommended_location": "Supplement or short lower-bound paragraph",
                "source_files": f"{STATIC_THRESHOLD_SUMMARY_FILE.name}; {STATIC_THRESHOLD_DELTA_FILE.name}",
                "role": "Shows fixed normalized storage is insufficient by itself.",
                "caution": "Do not call it a full Richards or measured FC/WP model.",
            },
            {
                "evidence_tier": "4_lower_bound_static_threshold_response_library",
                "recommended_location": "Supplement only",
                "source_files": f"{STATIC_THRESHOLD_LIBRARY_SUMMARY_FILE.name}; {STATIC_THRESHOLD_LIBRARY_DELTA_FILE.name}",
                "role": "Combines property-centric lower bound with response-memory sensitivity.",
                "caution": "Useful as sensitivity only.",
            },
        ]
    )
    return integrated, compact, hierarchy


def write_integrated_markdown(compact: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    lines = [
        "# SMDE-CSR response-centric 2 x 2 manuscript tables",
        "",
        "Material Passport:",
        "",
        "- Type: manuscript and supplement table package",
        "- Main evidence: conservative `static_recession` comparison",
        "- Sensitivity evidence: `response_library_analog` and `static_threshold_recession`",
        "- Cross-setting RMSE caution: FAWN uses mm; on-farm uses source soil moisture units",
        "- On-farm CSR caution: response-library analog is not full on-farm CSR registration",
        "",
        "Evidence hierarchy:",
        "",
        hierarchy.to_markdown(index=False),
        "",
        "Integrated compact table:",
        "",
        compact[
            [
                "evidence_tier",
                "recommended_location",
                "data_setting",
                "horizon_h",
                "n_forecasts",
                "n_events",
                "static_model",
                "response_model",
                "static_r2",
                "response_r2",
                "delta_r2",
                "static_ccc",
                "response_ccc",
                "delta_ccc",
                "rmse_ratio_response_over_static",
            ]
        ].to_markdown(index=False),
    ]
    (OUT / "SMDE_CSR_response_centric_2x2_integrated_tables.md").write_text("\n".join(lines), encoding="utf-8")


def plot_preliminary_delta(table: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24, wspace=0.34)
    horizons = sorted(table["horizon_h"].unique())

    ax = axes[0]
    for setting, group in table.groupby("data_setting", sort=False):
        group = group.sort_values("horizon_h")
        ax.plot(
            group["horizon_h"],
            group["delta_r2"],
            marker="o",
            lw=1.6,
            ms=4.2,
            color=COLORS.get(setting, "#222222"),
            label=setting,
        )
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Response minus static R2")
    ax.set_title("a  Predictive gain from response structure", loc="left", fontweight="bold")
    ax.grid(True, color=COLORS["grid"], lw=0.5)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for setting, group in table.groupby("data_setting", sort=False):
        group = group.sort_values("horizon_h")
        ax.plot(
            group["horizon_h"],
            group["rmse_ratio_response_over_static"],
            marker="s",
            lw=1.6,
            ms=4.0,
            color=COLORS.get(setting, "#222222"),
            label=setting,
        )
    ax.axhline(1, color="#555555", lw=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_ylim(0.35, 1.08)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Response/static RMSE ratio")
    ax.set_title("b  Error reduction is larger on farm", loc="left", fontweight="bold")
    ax.grid(True, color=COLORS["grid"], lw=0.5)

    note = (
        "Preliminary: FAWN uses mm; on-farm RMSE uses source units. "
        "Use R2/CCC for cross-setting comparison."
    )
    fig.text(0.08, 0.055, note, ha="left", va="bottom", fontsize=6.5, color="#444444")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"fig_response_centric_2x2_preliminary.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_outputs() -> None:
    ensure_out()
    summary = pd.read_csv(SUMMARY_FILE)
    delta = pd.read_csv(DELTA_FILE)
    table = build_manuscript_table(summary, delta)
    compact = build_compact_delta_table(table)
    integrated, integrated_compact, hierarchy = build_integrated_evidence_tables()
    table.to_csv(OUT / "SMDE_CSR_response_centric_2x2_manuscript_table.csv", index=False)
    compact.to_csv(OUT / "SMDE_CSR_response_centric_2x2_compact_table.csv", index=False)
    integrated.to_csv(OUT / "SMDE_CSR_response_centric_2x2_integrated_evidence_table.csv", index=False)
    integrated_compact.to_csv(OUT / "SMDE_CSR_response_centric_2x2_integrated_compact_table.csv", index=False)
    hierarchy.to_csv(OUT / "SMDE_CSR_response_centric_2x2_evidence_hierarchy.csv", index=False)
    write_integrated_markdown(integrated_compact, hierarchy)
    plot_preliminary_delta(table)
    print(
        {
            "manuscript_table_rows": int(len(table)),
            "compact_table_rows": int(len(compact)),
            "integrated_table_rows": int(len(integrated)),
            "evidence_hierarchy_rows": int(len(hierarchy)),
            "figure_png": str(OUT / "fig_response_centric_2x2_preliminary.png"),
        }
    )


if __name__ == "__main__":
    write_outputs()
