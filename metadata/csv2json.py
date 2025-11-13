#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic", "pydantic-extra-types", "pydantic-settings", "tqdm"]
# ///

"""Parse and cleanup fields from a yfcc100m_dataset file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from urllib.parse import urlunsplit, unquote_plus
from typing import Annotated

from pydantic import BaseModel, PlainSerializer
from pydantic_extra_types.coordinate import Coordinate
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_implicit_flags=True,
        cli_kebab_case=True,
        env_file=".env",
        env_prefix="scope_proxy_",
        extra="ignore",
    )

    meta_file: Path = "yfcc100m_dataset"
    json_file: Path = "yfcc100m.json"

    check_data: bool = False
    data_dir: Path | None = None


# batch size for each on-disk set_N/data_M split (from geturls.py)
BATCH_SIZE = 8_400_000

# map license identifiers to their SPDX abbreviation
LICENSES = {
    "http://creativecommons.org/licenses/by/2.0/": "CC-BY-2.0",
    "http://creativecommons.org/licenses/by-nc/2.0/": "CC-BY-NC-2.0",
    "http://creativecommons.org/licenses/by-nc-nd/2.0/": "CC-BY-NC-ND-2.0",
    "http://creativecommons.org/licenses/by-nc-sa/2.0/": "CC-BY-NC-SA-2.0",
    "http://creativecommons.org/licenses/by-nd/2.0/": "CC-BY-ND-2.0",
    "http://creativecommons.org/licenses/by-sa/2.0/": "CC-BY-SA-2.0",
}

# columns in the original yfcc100m_dataset file
METADATA_COLUMNS = [
    "index",  # BIGINT NOT NULL  # use this to derive set_N/data_M? No, it doesn't track skips.
    "image_id",  # BIGINT PRIMARY KEY  # unique image identifier
    "md5sum",  # CHAR(32) NOT NULL
    "user_uid",  # VARCHAR NOT NULL  # to construct flickr_url, map to table of users?
    "username",  # VARCHAR NOT NULL  # map to table of users? (there are fewer names than ids, so there must be duplicate names)
    "capture_time",  # VARCHAR  # probably time of capture (unreliable, may depend on camera settings?) (really DATETIME, but NULL = null)
    "upload_time",  # BIGINT NOT NULL  # most likely, it is consistently later than timestamp
    "camera_model",  # VARCHAR  # spaces are replaced with '+', urlencoded?
    "caption",  # VARCHAR  # spaces are replaced with '+', urlencoded?
    "description",  # VARCHAR  # spaces are replaced with '+', urlencoded?
    "tags",  # VARCHAR[]  # comma separated, spaces are replaced with '+', urlencoded? map to table of tags?
    "machine_tags",  # VARCHAR[]  # comma separated, namespace:tag=value
    "latitude",  # DOUBLE
    "longitude",  # DOUBLE
    "gps_satellites",  # UTINYINT  # only exists when there are GPS coordinates, number varies between 1 and 16
    "flickr_url",  # VARCHAR NOT NULL  # http://www.flickr.com/{ "videos" if is_video else "photos" }/{ user_id }/{ image_id }/
    "image_url",  # VARCHAR NOT NULL  # http://farm{ server_id }.staticflickr.com/{ bucket_id }/{ image_id }_{ image_crc }.{ file_ext }
    "license_text",  # VARCHAR NOT NULL  # finite set, map to table of licenses (there are only 6 license_text/license_url pairs)
    "license_url",  # VARCHAR NOT NULL  # finite set, map to table of licenses
    "bucket_id",  # USMALLINT NOT NULL  # to construct image_url
    "server_id",  # UTINYINT NOT NULL  # to construct image_url
    "image_crc",  # CHAR(10) NOT NULL  # to construct image_url, some hash/crc of the image? (5 hex encoded bytes)
    "unknown_crc",  # CHAR(10) NOT NULL  # unknown hash/crc, not otherwise used? (5 hex encoded bytes)
    "file_ext",  # VARCHAR NOT NULL  # to construct image_url
    "is_video",  # BOOLEAN NOT NULL  # to construct flickr_url, 0 == photos (99,206,564x), 1 == videos (793,436x)
]


class ImageMeta(BaseModel):
    image_id: int
    user_uid: str
    username: str
    set_id: int
    data_id: int
    bucket_id: int
    server_id: int
    image_crc: str
    upload_time: datetime
    license_id: str

    capture_time: datetime | None = None
    camera_model: str | None = None
    caption: str | None = None
    description: str | None = None
    tags: list[str] = []
    machine_tags: list[str] = []
    gps_coords: (
        Annotated[
            Coordinate,
            # Pydantic Coordinate serializes as {"latitude":..., "longitude":...}
            # but duckdb Point2D expects {"x":...,"y":...}
            PlainSerializer(lambda coord: {"x": coord.latitude, "y": coord.longitude}),
        ]
        | None
    ) = None

    @classmethod
    def from_csv(cls, fields: list[str], set_id: int, data_id: int) -> ImageMeta:
        imagemeta = cls(
            image_id=fields[1],
            user_uid=fields[3],
            username=unquote_plus(fields[4]),
            capture_time=fields[5] if fields[5] != "null" else None,
            upload_time=fields[6],
            camera_model=unquote_plus(fields[7]) if fields[7] else None,
            caption=unquote_plus(fields[8]) if fields[8] else None,
            description=unquote_plus(fields[9]) if fields[9] else None,
            tags=[unquote_plus(tag) for tag in fields[10].split(",") if tag],
            machine_tags=[unquote_plus(tag) for tag in fields[11].split(",") if tag],
            gps_coords=Coordinate(float(fields[12]), float(fields[13]))
            if fields[14]
            else None,
            license_id=LICENSES[fields[18]],
            set_id=set_id,
            data_id=data_id,
            bucket_id=fields[19],
            server_id=fields[20],
            image_crc=fields[21],
        )
        # assert imagemeta.flickr_url == fields[15]
        # assert imagemeta.image_url == fields[16]
        return imagemeta

    def path(self, base: Path | str) -> Path:
        return Path(base).joinpath(
            f"set_{self.set_id}",
            f"data_{self.data_id}",
            "images",
            str(self.bucket_id),
            f"{self.image_id}_{self.image_crc}.jpg",
        )

    @property
    def flickr_url(self) -> Path:
        return urlunsplit(
            (
                "http",
                "www.flickr.com",
                f"/photos/{self.user_uid}/{self.image_id}/",
                "",
                "",
            )
        )

    @property
    def image_url(self) -> Path:
        return urlunsplit(
            (
                "http",
                f"farm{self.server_id}.staticflickr.com",
                f"/{self.bucket_id}/{self.image_id}_{self.image_crc}.jpg",
                "",
                "",
            )
        )


def import_meta(meta_file: Path, json_file: Path, check_data: Path | None = None):
    # The meta file is huge, so lets do line by line
    with meta_file.open() as f, json_file.open("w") as out:
        skipped = 0
        for counter, line in tqdm(enumerate(f, start=1), total=100_000_000):
            fields = line.strip().split("\t")
            if fields[24] != "0":  # is_video
                skipped += 1
                continue

            batch_size = (counter - skipped) // BATCH_SIZE
            set_id, data_id = divmod(batch_size, 4)

            imagemeta = ImageMeta.from_csv(fields, set_id=set_id, data_id=data_id)
            out.write(imagemeta.model_dump_json(indent=None) + "\n")

            # expensive check if each file exists on disk
            if check_data is not None:
                assert imagemeta.path(check_data).exists()


if __name__ == "__main__":
    settings = Settings()
    check_data = settings.data_dir if settings.check_data else None
    import_meta(settings.meta_file, settings.json_file, check_data)
