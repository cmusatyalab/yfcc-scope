# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import niquests

from .. import settings
from ..types import WebDatasetShard, WIDSDescriptor
from ..utils import get_object
from .db import count_items_in_scope, get_item_by_id


async def generate_wids_descriptor(scope: str, base_url: str) -> WIDSDescriptor:
    nitems = await count_items_in_scope(scope)

    shard_name = f"{scope}-%06d.tar"
    nshards = nitems // settings.SCOPE_BATCH_SIZE
    last_nitems = nitems % settings.SCOPE_BATCH_SIZE

    # build the WIDS descriptor
    shardlist = [
        WebDatasetShard(url=shard_name % index, nsamples=settings.SCOPE_BATCH_SIZE)
        for index in range(nshards)
    ]
    if last_nitems:
        shardlist.append(
            WebDatasetShard(url=shard_name % nshards, nsamples=last_nitems)
        )
    return WIDSDescriptor(
        name=scope,
        base=base_url,
        shardlist=shardlist,
    )


async def get_image_by_id(image_id: str) -> bytes:
    session = niquests.AsyncSession()
    url, start, end = await get_item_by_id(image_id)
    obj = await get_object(session, url, start, end)

    with tarfile.open(fileobj=io.BytesIO(obj), mode="r") as tar:
        for member in tar:
            if member.isfile() and Path(member.name).suffix in (".jpg",):
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    return file_obj.read()

    msg = f"{image_id} JPEG image not found"
    raise KeyError(msg)
