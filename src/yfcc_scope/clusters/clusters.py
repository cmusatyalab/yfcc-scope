# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from fastapi import HTTPException

from ..cluster_centroids_pca3d.app import BASE_DIR
from ..scope.db import get_items_by_id
from ..types import (
    ClusterID,
    Embeddings,
    EmbeddingType,
    WebDatasetShard,
    WIDSDescriptor,
)


# should probably use a config setting to locate the directory containing
# clustered embeddings instead of using cluster_centroids_pca3d.app.BASE_DIR
def read_centroids(embedding_type: EmbeddingType) -> pl.DataFrame:
    centroids_file = Path(BASE_DIR, embedding_type.value, "centroids.parquet")
    if not centroids_file.exists():
        raise HTTPException(status_code=404, detail="Unknown embedding")
    return pl.read_parquet(centroids_file)


def scan_assignments(embedding_type: EmbeddingType) -> pl.LazyFrame:
    assignments_file = Path(BASE_DIR, embedding_type.value, "assignments.parquet")
    if not assignments_file.exists():
        raise HTTPException(status_code=404, detail="Unknown embedding")
    return pl.scan_parquet(assignments_file)


def embeddings_to_clusters(
    embedding_type: EmbeddingType, image_embeddings: Embeddings
) -> list[ClusterID]:
    # load the cluster centroids
    centroids = read_centroids(embedding_type).get_column("centroid").to_numpy()
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
    scores = centroids @ normalized_features.T
    return cast(list[ClusterID], np.argsort(-scores.max(axis=1)).astype(int).tolist())


async def cluster_wids_descriptor(
    embedding_type: EmbeddingType, base_url: str
) -> WIDSDescriptor:
    """When we decide on our clustering methods, this can become a static file."""
    # load the cluster centroids
    clusters = read_centroids(embedding_type).with_row_index()

    # build the WIDS descriptor
    shard_name = f"{embedding_type.value}-%06d.tar"
    shardlist = [
        WebDatasetShard(url=shard_name % cluster, nsamples=length)
        for cluster, _, length in clusters.iter_rows()
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
    """Build a cluster specific webdataset shard."""
    # Extract the image ids from the cluster assignments
    image_ids = await (
        scan_assignments(embedding_type)
        .filter(pl.col("cluster") == cluster)
        .select(pl.col("image_id"))
    ).collect_async()

    async def item_ids(image_ids: pl.DataFrame) -> AsyncIterator[str]:
        for row in image_ids.iter_rows():
            yield str(row[0])

    # Map item ids to shard urls and locations.
    async for result in get_items_by_id(item_ids(image_ids)):
        yield result
