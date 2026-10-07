# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio

import typer
from rich import print
from tqdm.asyncio import tqdm

from ..types import (
    ClusterID,
    EmbeddingType,
)
from ..utils import generate_webdataset_shard
from .clusters import (
    get_items_in_cluster,
    scoped_wids_descriptor,
)

app = typer.Typer(
    help="Test cluster-specific wids and shard generators", no_args_is_help=True
)


async def create_wids(
    embedding: EmbeddingType, clusters: list[int], limit: int
) -> None:
    base_url = "http://localhost:8000/"
    cluster_ids = [ClusterID(cluster) for cluster in clusters]
    wids_descriptor = await scoped_wids_descriptor(
        embedding, base_url, cluster_ids, limit
    )
    print(wids_descriptor.model_dump())


@app.command()
def wids(embedding: EmbeddingType, cluster: list[int], limit: int = 0) -> None:
    """Generate a 'WIDS' json descriptor for the embedding.

    This would normally be the response to URL_BASE/<embedding>.json
    """
    asyncio.run(create_wids(embedding, cluster, limit))


async def create_shard(embedding: EmbeddingType, cluster: int) -> None:
    items = get_items_in_cluster(embedding, ClusterID(cluster))
    items = tqdm(items)

    with open(f"{embedding}-{cluster:06d}.tar", "wb") as f:
        async for obj in generate_webdataset_shard(items):
            f.write(obj)


@app.command()
def shard(embedding: EmbeddingType, cluster: int) -> None:
    """Generate a shard tar archive for the embedding.

    This would normally be the response to accessing
    URL_BASE/<embedding>-{cluster:06d}.tar
    """
    asyncio.run(create_shard(embedding, cluster))


if __name__ == "__main__":
    app()
