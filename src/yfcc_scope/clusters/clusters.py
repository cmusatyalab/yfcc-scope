# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NewType

import numpy as np
import polars as pl
from starlette.exceptions import HTTPException

from ..cluster_centroids_pca3d.app import BASE_DIR
from ..scope.db import get_items_by_id

ClusterID = NewType("ClusterID", int)
Embedding = NewType("Embedding", list[float])
ItemID = NewType("ItemID", str)


def embeddings_to_clusters(
    embedding_type: str, image_embeddings: list[Embedding] | None
) -> list[ClusterID] | None:
    if image_embeddings is None:
        return None

    # load the cluster centroids
    centroids_file = Path(BASE_DIR, embedding_type, "centroids.parquet")
    if not centroids_file.exists():
        raise HTTPException(status_code=404, detail="Unknown embedding")

    centroids = pl.read_parquet(centroids_file).get_column("centroids").to_numpy()

    # convert image_embeddings to numpy array and verify they are correctly sized
    image_features = np.asarray(image_embeddings, dtype=np.float32)
    if centroids.shape[1] != image_features.shape[1]:
        raise HTTPException(status_code=400, detail="Mismatched embedding size")

    # normalize the centroids
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / np.maximum(centroid_norms, 1e-12)

    # normalize the image_features
    feature_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
    normalized_features = image_features / np.maximum(feature_norms, 1e-12)

    # score the image embeddings
    scores = normalized_centroids @ normalized_features.T
    return np.argsort(-scores.max(axis=1)).astype(int).tolist()


# should probably use a config setting to locate the directory containing
# clustered embeddings instead of using cluster_centroids_pca3d.app.BASE_DIR
def assignments(embedding_type: str) -> pl.LazyFrame:
    assignments_file = Path(BASE_DIR, embedding_type, "assignments.parquet")
    if not assignments_file.exists():
        raise HTTPException(status_code=404, detail="Unknown embedding")
    return pl.scan_parquet(assignments_file)


async def cluster_wids_descriptor(embedding_type: str, base_url: str) -> dict[str, Any]:
    """When we decide on our clustering methods, this can become a static file."""
    shard_name = f"{embedding_type}-%06d.tar"

    clusters = await (
        assignments(embedding_type).group_by("centroid").len().sort("centroid")
    ).collect_async()

    # build the WIDS descriptor
    shardlist = [
        {"url": shard_name % cluster, "nsamples": length}
        for cluster, length in clusters.iter_rows()
    ]
    wids_descriptor = {
        "wids_version": 1,
        "name": f"yfcc100m-{embedding_type}",
        "description": f"YFCC100M clustered by {embedding_type}",
        "base": base_url,
        "shardlist": shardlist,
    }
    return wids_descriptor


async def scoped_wids_descriptor(
    embedding_type: str, base_url: str, clusters: list[ClusterID] | None = None
) -> dict[str, Any]:
    """Take an existing full dataset descriptor and return only the scoped subset."""
    # here we could fetch an existing (pre-computed) wids descriptor instead
    wids_descriptor = await cluster_wids_descriptor(embedding_type, base_url)

    if clusters is None:
        return wids_descriptor

    # copy the original shard list
    shards = wids_descriptor["shardlist"]

    # and rebuild the scoped WIDS descriptor
    wids_descriptor["name"] = f"yfcc100m-{embedding_type}-scoped"
    wids_descriptor["shardlist"] = [shards[cluster] for cluster in clusters]
    return wids_descriptor


async def get_items_in_cluster(
    embedding_type: str, cluster: ClusterID
) -> AsyncIterator[tuple[str, int, int]]:
    """Build a cluster specific webdataset shard."""
    # Extract the image ids from the cluster assignments
    image_ids = await (
        assignments(embedding_type)
        .filter(pl.col("centroid") == cluster)
        .select(pl.col("image_id"))
    ).collect_async()

    # Map image ids (int) to an async list of item ids (str)
    async def get_item_ids(image_ids: pl.DataFrame) -> AsyncIterator[str]:
        for row in image_ids.iter_rows():
            yield str(row[0])

    item_ids = get_item_ids(image_ids)

    # Map itemp ids to shard urls and locations.
    async for result in get_items_by_id(item_ids):
        yield result
