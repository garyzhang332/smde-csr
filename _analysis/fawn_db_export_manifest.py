from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent / "fawn_db_export"
DATA_DIR = BASE_DIR / "data"
OUT_CSV = BASE_DIR / "export_manifest_2023_2025.csv"
OUT_JSON = BASE_DIR / "export_manifest_2023_2025.json"


def summarize_file(path: Path) -> dict[str, object]:
    pf = pq.ParquetFile(path)
    cols = pf.schema.names
    id_utc = pd.read_parquet(path, columns=["ID", "UTC"])
    return {
        "file": str(path),
        "name": path.name,
        "rows": int(pf.metadata.num_rows),
        "columns_n": len(cols),
        "columns": cols,
        "stations": int(id_utc["ID"].nunique()),
        "first_utc": str(id_utc["UTC"].min()),
        "last_utc": str(id_utc["UTC"].max()),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def main() -> None:
    records = []
    for path in sorted(DATA_DIR.glob("*.parquet")):
        records.append(summarize_file(path))

    manifest = pd.DataFrame(records)
    flat = manifest.drop(columns=["columns"]).copy()
    flat.to_csv(OUT_CSV, index=False)

    OUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(flat.to_string(index=False))
    print(f"\nTotal rows: {flat['rows'].sum():,}")
    print(f"Total size: {flat['size_mb'].sum():,.2f} MB")
    print(f"\nManifest CSV: {OUT_CSV}")
    print(f"Manifest JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
