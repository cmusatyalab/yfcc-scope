# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import niquests
from tenacity import RetryError, TryAgain, retry, wait_exponential

from .. import settings
from .db import count_items_in_scope, get_item_by_id

logger = logging.getLogger(__name__)


async def generate_wids_descriptor(scope: str, base_url: str) -> dict[str, Any]:
    nitems = await count_items_in_scope(scope)

    shard_name = f"{scope}-%06d.tar"
    nshards = nitems // settings.SCOPE_BATCH_SIZE
    last_nitems = nitems % settings.SCOPE_BATCH_SIZE

    # build the WIDS descriptor
    shardlist = [
        {"url": shard_name % index, "nsamples": settings.SCOPE_BATCH_SIZE}
        for index in range(nshards)
    ]
    if last_nitems:
        shardlist.append(
            {
                "url": shard_name % nshards,
                "nsamples": last_nitems,
            }
        )

    wids_descriptor = {
        "wids_version": 1,
        "name": scope,
        "description": "",
        "base": base_url,
        "shardlist": shardlist,
    }
    return wids_descriptor


@retry(wait=wait_exponential(multiplier=1, max=10))
async def get_object(
    session: niquests.AsyncSession, url: str, offset: int, end: int
) -> bytes:
    # shard = url.rsplit("/", 1)[1]
    # url = f"http://blue1.satyalab/yfcc100m/{shard}"
    headers = {"Range": f"bytes={offset}-{end}"}
    response = await session.get(url, headers=headers)
    response.raise_for_status()
    if response.content is None:
        logger.error(f"Received empty content from {url} ({offset}-{end})")
        raise TryAgain
    return response.content


async def generate_shard(
    items: AsyncIterator[tuple[str, int, int]],
) -> AsyncIterator[bytes]:
    session = niquests.AsyncSession()

    # sliding window for url range-request fetches
    tasks = []
    for _ in range(settings.SCOPE_FETCH_WINDOW):
        try:
            url, start, end = await anext(items)
            tasks.append(asyncio.create_task(get_object(session, url, start, end)))
        except StopAsyncIteration:
            break

    while tasks:
        try:
            yield await tasks.pop(0)
        except RetryError:
            logger.error("Timed out trying to fetch object")

        with suppress(StopAsyncIteration):
            url, start, end = await anext(items)
            tasks.append(asyncio.create_task(get_object(session, url, start, end)))


async def get_image_by_id(image_id: str) -> bytes:
    session = niquests.AsyncSession()
    url, start, end = await get_item_by_id(image_id)
    obj = await get_object(session, url, start, end)

    with tarfile.open(fileobj=io.BytesIO(obj), mode="r") as tar:
        for member in tar:
            if member.isfile() and Path(member.name).suffix in (".jpg",):
                with tar.extractfile(member) as file_obj:
                    return file_obj.read()

    msg = f"{image_id} JPEG image not found"
    raise KeyError(msg)
