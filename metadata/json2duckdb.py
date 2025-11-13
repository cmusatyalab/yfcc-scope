#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb", "pydantic-settings"]
# ///

"""Load yfcc100m metadata from json file into a duckdb database.

There isn't a lot of python code here, I could probably have done this
all with the duckdb cli.

It may be possible to use duckdb's ability to layer on top of sqlite or
postgresql to import into those databases.
"""

from pathlib import Path

import duckdb
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_implicit_flags=True,
        cli_kebab_case=True,
        env_file=".env",
        env_prefix="scope_proxy_",
        extra="ignore",
    )

    json_file: Path = "yfcc100m.json"
    duckdb: Path = "yfcc100m.duckdb"


settings = Settings()


def db_connect(db: Path | str, progress=False):
    con = duckdb.connect(str(db))
    con.install_extension("spatial")
    con.load_extension("spatial")
    if progress:
        con.sql("PRAGMA enable_progress_bar")
    return con


con = db_connect(settings.duckdb, progress=True)

print("# Creating 'users' table")
con.sql(
    "CREATE TABLE IF NOT EXISTS users AS "
    f"SELECT DISTINCT user_uid, username FROM read_ndjson({settings.json_file}, "
    "    columns = {user_uid: 'VARCHAR', username: 'VARCHAR'});"
)

## This works also, and we need fewer joins, but the DB is larger and queries
## selecting all images with specific tags are a little slower.
# print("# Creating 'image_tags' table (image_id, tagname)")
# con.sql(
#     "CREATE TABLE IF NOT EXISTS image_tags AS "
#     f"SELECT image_id, unnest(tags) AS tag FROM read_ndjson({settings.json_file}, "
#     "    columns = {image_id: 'BIGINT', tags: 'VARCHAR[]'});"
# )

print("# Creating 'tags' table")
# Using a temp table so we can use the rowid on the temp table as unique identifiers
con.sql(
    "CREATE TEMP TABLE unique_tags AS "
    f"SELECT DISTINCT unnest(tags) AS tag FROM read_ndjson({settings.json_file}, "
    "    columns = {tags: 'VARCHAR[]'});"
)
con.sql(
    "CREATE TABLE IF NOT EXISTS tags AS SELECT rowid AS tag_id, tag FROM unique_tags"
)
con.sql("DROP TABLE unique_tags")

print("# Creating 'image_tags' table (image_id, tag_id)")
con.sql(
    "CREATE TEMP TABLE expanded_tags AS "
    f"SELECT image_id, unnest(tags) AS tag FROM read_ndjson({settings.json_file}, "
    "    columns = {image_id: 'BIGINT', tags: 'VARCHAR[]'});"
)
con.sql(
    "CREATE TABLE IF NOT EXISTS image_tags AS "
    f"SELECT image_id, tag_id FROM expanded_tags JOIN tags ON expanded_tags.tag = tags.tag"
)
con.sql("DROP TABLE expanded_tags")

## Ignoring the machine_tags for now.
## The idea is nice, but the use seems to be not standardized and inconsistent.
## lots of images actually placed the tags in the description so they were not
## picked up as separate machine_tags, the namespace:key values are somewhat
## consistent but I still see various inconsistencies like geo:lon, geo:long,
## and geo:longitude and the values are all over the place. Not sure how to
## effectively use these for scoping.

## This takes 4 minutes to save 7MB (in a 17GB database)
# print("# Creating 'license' enum")
# con.sql(f"CREATE TYPE license AS ENUM (SELECT license_id FROM read_ndjson({settings.json_file}))")

print("# Creating 'dataset' table")
con.sql(
    "CREATE TABLE IF NOT EXISTS dataset AS "
    f"SELECT * FROM read_ndjson({settings.json_file}, "
    """columns = {
        image_id: "BIGINT",
        user_uid: "VARCHAR",
        set_id: "UTINYINT",
        data_id: "UTINYINT",
        server_id: "UTINYINT",
        bucket_id: "USMALLINT",
        image_crc: "VARCHAR",
        capture_time: "TIMESTAMP_S",
        upload_time: "TIMESTAMP_S",
        camera_model: "VARCHAR",
        caption: "VARCHAR",
        description: "VARCHAR",
        gps_coords: "POINT_2D",
        license_id: "VARCHAR",
   });"""
)

print("# Summarizing 'dataset' table")
print(con.sql("SUMMARIZE dataset"))
con.close()
