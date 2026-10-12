# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from fastapi import HTTPException

from ..settings import EMBEDDING_BASE_DIR
from ..types import (
    ClusterID,
    Embeddings,
    EmbeddingType,
    WebDatasetShard,
    WIDSDescriptor,
)


def scan(embedding_type: EmbeddingType, datafile: str) -> pl.LazyFrame:
    data_path = Path(EMBEDDING_BASE_DIR, embedding_type.value, datafile).with_suffix(".parquet")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Unknown embedding")
    return pl.scan_parquet(data_path)


def scan_clusters(embedding_type: EmbeddingType) -> pl.LazyFrame:
    clusters_file = Path(EMBEDDING_BASE_DIR, embedding_type.value, "clusters.parquet")
    if not clusters_file.exists():
        raise HTTPException(status_code=404, detail="Missing cluster index")
    return pl.scan_parquet(clusters_file)


def embeddings_to_clusters(
    embedding_type: EmbeddingType, image_embeddings: Embeddings
) -> list[ClusterID]:
    # load the cluster centroids
    centroids = (
        scan(embedding_type, "centroids")
        .select(pl.col("centroid"))
        .collect()
    ).get_column("centroid").to_numpy()
    dims = centroids.shape[1]

    # convert image_embeddings to numpy array and verify they are correctly sized
    image_features = np.asarray(image_embeddings.embeddings, dtype=np.float32)
    if image_features.shape[1] != dims:
        raise HTTPException(
            status_code=400, detail=f"Mismatched embedding size, expected {dims}"
        )

    # normalize the image_features
    feature_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
    normalized_features = image_features / np.maximum(feature_norms, 1e-12)

    # score the image embeddings
    scores = (centroids @ normalized_features.T).max(axis=1)
    return cast(list[ClusterID], np.argsort(-scores).astype(int).tolist())


async def cluster_wids_descriptor(
    embedding_type: EmbeddingType, base_url: str
) -> WIDSDescriptor:
    """When we decide on our clustering methods, this can become a static file."""
    # load the cluster centroids
    clusters = await (
        scan(embedding_type, "centroids")
        .select(pl.col("nsamples"))
        .with_row_index()
    ).collect_async()

    # build the WIDS descriptor
    shard_name = f"{embedding_type.value}-%06d.tar"
    shardlist = [
        WebDatasetShard(url=shard_name % cluster, nsamples=length)
        for cluster, length in clusters.iter_rows()
    ]

    return WIDSDescriptor(
        name=f"yfcc100m-{embedding_type.value}",
        description=f"YFCC100M clustered by {embedding_type.value}",
        base=base_url,
        shardlist=shardlist,
    )


async def scoped_wids_descriptor(
    embedding_type: EmbeddingType,
    base_url: str,
    clusters: list[ClusterID],
    limit: int = 0,
) -> WIDSDescriptor:
    """Take an existing full dataset descriptor and return only the scoped subset."""
    # here we could fetch an existing (pre-computed) wids descriptor instead
    wids_descriptor = await cluster_wids_descriptor(embedding_type, base_url)

    # build a new shardlist reordered based on the list of clusters
    shardlist = []
    collected = 0
    for cluster in clusters:
        shard = wids_descriptor.shardlist[cluster]
        shardlist.append(shard)
        collected += shard.nsamples
        if limit and collected >= limit:
            break

    # and rebuild the scoped WIDS descriptor
    wids_descriptor.name += "-scoped"
    wids_descriptor.shardlist = shardlist
    return wids_descriptor


async def get_items_in_cluster(
    embedding_type: EmbeddingType, cluster: ClusterID
) -> AsyncIterator[tuple[str, int, int]]:
    """Return object indices to build a cluster specific webdataset shard."""

    cluster_offsets = await (
        scan(embedding_type, "centroids")
        .select(
            pl.col("nsamples").shift(1, fill_value=0).cum_sum().alias("offset"),
            pl.col("nsamples").alias("length"),
        )
        .slice(cluster, 1)
    ).collect_async()
    cluster_offset, cluster_length = cluster_offsets.row(0)

    items = await (
        scan(embedding_type, "clusters")
        .select(pl.col("url"), pl.col("offset"), pl.col("length"))
        .slice(cluster_offset, cluster_length)
    ).collect_async()
    for url, offset, length in items.iter_rows():
        yield url, offset, offset + length - 1
