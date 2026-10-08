# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio
from pathlib import Path

import polars as pl
import typer
from rich import print
from tqdm.asyncio import tqdm

from ..settings import EMBEDDING_BASE_DIR
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


@app.command()
def build_cluster_index(embedding: EmbeddingType, shard_index: Path) -> None:
    """Build a cluster index ('clusters.parquet') for all objects in assignments."""
    assignments = Path(EMBEDDING_BASE_DIR, embedding.value, "assignments.parquet")
    clusters = Path(EMBEDDING_BASE_DIR, embedding.value, "clusters.parquet")

    shard_index_df = pl.scan_ndjson(shard_index)
    cluster_index = (
        pl.scan_parquet(assignments)
        .select(pl.col("image_id").cast(pl.String), pl.col("cluster"))
        .join(shard_index_df, on="image_id")
        .sort("cluster", "shard", "offset")
        .select("cluster", "image_id", "shard", "offset", "length")
    )
    # print(cluster_index.explain())
    cluster_index.sink_parquet(clusters)


if __name__ == "__main__":
    app()
