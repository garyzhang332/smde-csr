from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text


OUT_DIR = Path(__file__).resolve().parent / "fawn_db_export" / "data"

SOIL_COLS = [
    "ID",
    "UTC",
    "temp_sms_4_inch_C",
    "temp_sms_8_inch_C",
    "temp_sms_12_inch_C",
    "temp_sms_16_inch_C",
    "temp_sms_20_inch_C",
    "moisture_sms_4_inch_pct",
    "moisture_sms_8_inch_pct",
    "moisture_sms_12_inch_pct",
    "moisture_sms_16_inch_pct",
    "moisture_sms_20_inch_pct",
]

WX_COLS = [
    "ID",
    "UTC",
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


def year_window(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year + 1}-01-01"


def select_expr(table_alias: str, cols: list[str]) -> str:
    return ", ".join([f"{table_alias}.`{col}`" for col in cols])


def build_sql(table: str) -> str:
    if table == "soil_moisture":
        return f"""
            SELECT {select_expr("soil_moisture", SOIL_COLS)}
            FROM soil_moisture
            WHERE UTC >= :start AND UTC < :end
            ORDER BY ID, UTC
        """
    if table == "wx":
        return f"""
            SELECT {select_expr("wx", WX_COLS)}
            FROM wx
            INNER JOIN (
                SELECT DISTINCT ID
                FROM soil_moisture
                WHERE UTC >= :overall_start AND UTC < :overall_end
            ) sm_ids ON sm_ids.ID = wx.ID
            WHERE wx.UTC >= :start AND wx.UTC < :end
            ORDER BY wx.ID, wx.UTC
        """
    raise ValueError(f"Unknown table: {table}")


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["ID"] = pd.to_numeric(chunk["ID"], errors="coerce").astype("Int64")
    chunk["UTC"] = pd.to_datetime(chunk["UTC"], errors="coerce")
    for col in chunk.columns:
        if col not in {"ID", "UTC"}:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    return chunk


def write_parquet_from_query(
    engine,
    sql: str,
    params: dict[str, str],
    out_path: Path,
    chunksize: int,
) -> dict[str, object]:
    if out_path.exists():
        out_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    first_utc = None
    last_utc = None
    station_ids: set[int] = set()

    with engine.connect() as conn:
        for chunk in pd.read_sql_query(text(sql), conn, params=params, chunksize=chunksize):
            chunk = normalize_chunk(chunk)
            chunk = chunk.dropna(subset=["ID", "UTC"])
            if chunk.empty:
                continue

            total_rows += len(chunk)
            first_utc = chunk["UTC"].min() if first_utc is None else min(first_utc, chunk["UTC"].min())
            last_utc = chunk["UTC"].max() if last_utc is None else max(last_utc, chunk["UTC"].max())
            station_ids.update(chunk["ID"].dropna().astype(int).unique().tolist())

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)

    if writer is not None:
        writer.close()

    return {
        "path": str(out_path),
        "rows": total_rows,
        "stations": len(station_ids),
        "first_utc": None if first_utc is None else str(first_utc),
        "last_utc": None if last_utc is None else str(last_utc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["soil_moisture", "wx"], required=True)
    parser.add_argument("--year", type=int, choices=[2023, 2024, 2025, 2026], required=True)
    parser.add_argument("--start", default=None, help="Optional inclusive UTC start, e.g. 2026-01-01")
    parser.add_argument("--end", default=None, help="Optional exclusive UTC end, e.g. 2026-06-06")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    db_url = os.environ.get("FAWN_DB_URL")
    if not db_url:
        raise RuntimeError("Set FAWN_DB_URL before running this script.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url)
    start, end = year_window(args.year)
    if args.start:
        start = args.start
    if args.end:
        end = args.end
    out_name = f"{args.table if args.table == 'soil_moisture' else 'wx_selected'}_{args.year}.parquet"
    out_path = OUT_DIR / out_name

    summary = write_parquet_from_query(
        engine=engine,
        sql=build_sql(args.table),
        params={
            "start": start,
            "end": end,
            "overall_start": start,
            "overall_end": end,
        },
        out_path=out_path,
        chunksize=args.chunksize,
    )
    summary.update({"table": args.table, "year": args.year, "columns": SOIL_COLS if args.table == "soil_moisture" else WX_COLS})

    manifest_path = OUT_DIR / f"{args.table if args.table == 'soil_moisture' else 'wx_selected'}_{args.year}.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
