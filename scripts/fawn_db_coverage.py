from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


OUT_DIR = Path(__file__).resolve().parents[1] / "fawn_db_export" / "coverage"
START = "2023-01-01"
END = "2026-01-01"

WX_DIAGNOSTIC_COLS = [
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
]


def main() -> None:
    db_url = os.environ.get("FAWN_DB_URL")
    if not db_url:
        raise RuntimeError("Set FAWN_DB_URL before running this script.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url)

    with engine.connect() as conn:
        soil_coverage = pd.read_sql(
            text(
                """
                SELECT
                    ID,
                    YEAR(UTC) AS year,
                    MIN(UTC) AS first_utc,
                    MAX(UTC) AS last_utc,
                    COUNT(*) AS rows_n,
                    COUNT(moisture_sms_4_inch_pct) AS moisture_4_n,
                    COUNT(moisture_sms_8_inch_pct) AS moisture_8_n,
                    COUNT(moisture_sms_12_inch_pct) AS moisture_12_n,
                    COUNT(moisture_sms_16_inch_pct) AS moisture_16_n,
                    COUNT(moisture_sms_20_inch_pct) AS moisture_20_n
                FROM soil_moisture
                WHERE UTC >= :start AND UTC < :end
                GROUP BY ID, YEAR(UTC)
                ORDER BY ID, year
                """
            ),
            conn,
            params={"start": START, "end": END},
        )

        count_exprs = ",\n                    ".join(
            [f"COUNT({col}) AS {col}_n" for col in WX_DIAGNOSTIC_COLS]
        )
        wx_sql = f"""
                SELECT
                    wx.ID,
                    YEAR(wx.UTC) AS year,
                    MIN(wx.UTC) AS first_utc,
                    MAX(wx.UTC) AS last_utc,
                    COUNT(*) AS rows_n,
                    {count_exprs},
                    SUM(CASE WHEN rain_2m_inches > 0 THEN 1 ELSE 0 END) AS rain_positive_rows,
                    SUM(COALESCE(rain_2m_inches, 0)) AS rain_total_inches
                FROM wx
                INNER JOIN (
                    SELECT DISTINCT ID
                    FROM soil_moisture
                    WHERE UTC >= :start AND UTC < :end
                ) sm_ids ON sm_ids.ID = wx.ID
                WHERE wx.UTC >= :start AND wx.UTC < :end
                GROUP BY wx.ID, YEAR(wx.UTC)
                ORDER BY wx.ID, year
                """
        wx_coverage = pd.read_sql(
            text(wx_sql),
            conn,
            params={"start": START, "end": END},
        )

    soil_coverage.to_csv(OUT_DIR / "soil_moisture_coverage_2023_2025.csv", index=False)
    wx_coverage.to_csv(OUT_DIR / "wx_selected_coverage_2023_2025.csv", index=False)

    soil_ids = set(soil_coverage["ID"].astype(int).unique())
    wx_ids = set(wx_coverage["ID"].astype(int).unique())
    overlap_ids = sorted(soil_ids.intersection(wx_ids))
    coverage_join = soil_coverage.merge(
        wx_coverage[["ID", "year", "rows_n", "rain_2m_inches_n", "rain_total_inches"]],
        on=["ID", "year"],
        how="left",
        suffixes=("_soil", "_wx"),
    )
    coverage_join.to_csv(OUT_DIR / "soil_wx_coverage_join_2023_2025.csv", index=False)

    print(f"Soil station IDs: {len(soil_ids):,}")
    print(f"WX station IDs with soil overlap: {len(wx_ids):,}")
    print(f"Station overlap IDs: {len(overlap_ids):,}")
    print(f"Soil rows 2023-2025: {soil_coverage['rows_n'].sum():,}")
    print(f"WX rows 2023-2025 for soil IDs: {wx_coverage['rows_n'].sum():,}")
    print("\nSoil coverage by year:")
    print(soil_coverage.groupby("year").agg(stations=("ID", "nunique"), rows=("rows_n", "sum")).to_string())
    print("\nWX coverage by year for soil IDs:")
    print(wx_coverage.groupby("year").agg(stations=("ID", "nunique"), rows=("rows_n", "sum")).to_string())
    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()


