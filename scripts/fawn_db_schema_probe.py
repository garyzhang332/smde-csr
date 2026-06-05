from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


OUT_DIR = Path(__file__).resolve().parents[1] / "fawn_db_export" / "schema"


def main() -> None:
    db_url = os.environ.get("FAWN_DB_URL")
    if not db_url:
        raise RuntimeError("Set FAWN_DB_URL before running this script.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url)

    with engine.connect() as conn:
        schema = pd.read_sql(
            text(
                """
                SELECT
                    table_name,
                    ordinal_position,
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name IN ('soil_moisture', 'wx')
                ORDER BY table_name, ordinal_position
                """
            ),
            conn,
        )

        samples = {}
        for table_name in ["soil_moisture", "wx"]:
            samples[table_name] = pd.read_sql(text(f"SELECT * FROM `{table_name}` LIMIT 5"), conn)

    schema.to_csv(OUT_DIR / "soil_moisture_wx_schema.csv", index=False)
    for table_name, sample in samples.items():
        sample.to_csv(OUT_DIR / f"{table_name}_sample.csv", index=False)

    print(f"Schema rows: {len(schema):,}")
    for table_name, sample in samples.items():
        print(f"\n{table_name} sample columns:")
        print(", ".join(sample.columns.tolist()))
        print(sample.head(2).to_string(index=False))
    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()


