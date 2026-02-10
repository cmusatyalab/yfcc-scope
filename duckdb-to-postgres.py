#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb"]
# ///
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import psycopg2, psycopg2.extras
import os
import time
import webdataset as wds
import json

"""
Export DuckDB tables into your existing Postgres DB (yfcc).
- Uses DuckDB postgres core extension
- Uses DuckDB spatial extension to handle POINT_2D gps_coords
"""

import duckdb

DUCKDB_PATH = "yfcc100m.duckdb"
def make_pg_conn_str_from_env() -> str:
    return (
        f"host={os.environ.get('DB_HOST', '127.0.0.1')} "
        f"port={os.environ.get('DB_PORT', '5432')} "
        f"dbname={os.environ.get('DB_NAME', 'yfcc')} "
        f"user={os.environ.get('DB_USER', 'postgres')} "
        f"password={os.environ.get('DB_PASSWORD', 'postgres')} "
        "sslmode=disable"
    )
PG_CONN_STR = make_pg_conn_str_from_env() 
# # Matches your psycopg2 connection:
# PG_CONN_STR = (
#     "host=127.0.0.1 "
#     "port=5432 "
#     "dbname=yfcc "
#     "user=postgres "
#     "password=postgres "
#     "sslmode=disable"
# )

TARGET_SCHEMA = "public"   # change if you want a custom schema


def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect(DUCKDB_PATH)

    print("Installing/loading postgres + spatial extensions...")
    # Postgres extension
    con.sql("INSTALL postgres;")
    con.sql("LOAD postgres;")

    # Spatial extension (needed for ST_X/ST_Y on POINT_2D)
    con.sql("INSTALL spatial;")
    con.sql("LOAD spatial;")

    print("Attaching to your existing Postgres DB...")
    con.sql(f"ATTACH '{PG_CONN_STR}' AS pg (TYPE postgres);")

    print("Setting null-byte replacement for Postgres...")
    # Strip NULL bytes from strings before sending to Postgres
    con.sql("SET pg_null_byte_replacement='';")

    print(f"Ensuring schema pg.{TARGET_SCHEMA} exists...")
    con.sql(f"CREATE SCHEMA IF NOT EXISTS pg.{TARGET_SCHEMA};")

    # ---- Drop old tables so we get a clean overwrite ----
    for t in ["metadata", "users", "tags", "image_tags"]:
        print(f"Dropping old pg.{TARGET_SCHEMA}.{t} (if exists)")
        con.sql(f"DROP TABLE IF EXISTS pg.{TARGET_SCHEMA}.{t};")

    # ---- Export dataset, converting POINT_2D -> (gps_lat, gps_lon)
    #      and adding combined image_file_id = '<image_id>_<image_crc>'
    print(f"Exporting dataset → pg.{TARGET_SCHEMA}.metadata")
    con.sql(
        f"""
        CREATE TABLE pg.{TARGET_SCHEMA}.metadata AS
        SELECT
            image_id,
            CONCAT(CAST(image_id AS VARCHAR), '_', image_crc) AS image_file_id,
            user_uid,
            set_id,
            data_id,
            server_id,
            bucket_id,
            image_crc,
            capture_time,
            upload_time,
            camera_model,
            caption,
            description,
            ST_Y(gps_coords) AS gps_lat,
            ST_X(gps_coords) AS gps_lon,
            license_id
        FROM main.dataset;
        """
    )

    # ---- Export simpler tables as-is ----
    for t in ["users", "tags"]:
        print(f"Exporting {t} → pg.{TARGET_SCHEMA}.{t}")
        con.sql(
            f"""
            CREATE TABLE pg.{TARGET_SCHEMA}.{t} AS
            SELECT * FROM main.{t};
            """
        )

    print("✔ Done! Tables exported to your Postgres database 'yfcc'.")
    con.close()


if __name__ == "__main__":
    main()
