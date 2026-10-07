# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, File, Form, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..embedding import image_model
from ..log import log
from ..types import ClusterID, EmbeddingType

METHOD = "faiss_kmeans"
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class ClusterData:
    embedding: EmbeddingType
    centroids: npt.NDArray
    pca3d_centroids: npt.NDArray
    inverted_index_indptr: npt.NDArray
    inverted_index_order: npt.NDArray
    counts: npt.NDArray
    image_id_list: list[int]


pca_cluster_data: dict[EmbeddingType, ClusterData] = {}


def _load_pca_data(embedding: EmbeddingType) -> None:
    if pca_cluster_data.get(embedding):
        return

    embedding_type = embedding.value

    log.info(f"Loading {embedding_type} data")
    centroids = np.load(BASE_DIR / embedding_type / f"{METHOD}_centroids.npy")
    pca3d_centroids = np.load(
        BASE_DIR / embedding_type / f"{METHOD}_pca3d_centroids.npy"
    )
    assignments = np.load(BASE_DIR / embedding_type / f"{METHOD}_assignments.npy")
    inverted_index_indptr = np.load(
        BASE_DIR / embedding_type / f"{METHOD}_inverted_index_indptr.npy"
    )
    inverted_index_order = np.load(
        BASE_DIR / embedding_type / f"{METHOD}_inverted_index_order.npy"
    )
    with open(BASE_DIR / embedding_type / "image_id_list.json") as f:
        image_id_list = json.load(f)

    n_clusters = pca3d_centroids.shape[0]
    counts = np.bincount(assignments, minlength=n_clusters)

    pca_cluster_data[embedding] = ClusterData(
        embedding=embedding,
        centroids=centroids,
        pca3d_centroids=pca3d_centroids,
        inverted_index_indptr=inverted_index_indptr,
        inverted_index_order=inverted_index_order,
        counts=counts,
        image_id_list=image_id_list,
    )


async def load_pca_data(embedding: EmbeddingType) -> ClusterData:
    if embedding not in pca_cluster_data:
        try:
            await run_in_threadpool(_load_pca_data, embedding)
        except FileNotFoundError:
            log.exception("load pca data failed")
            raise HTTPException(
                status_code=404,
                detail=f"Data for embedding '{embedding.value}' not found",
            ) from None
    return pca_cluster_data[embedding]


router = APIRouter(tags=["cluster explorer"])


@router.get("/centroids_pca3d")
async def centroids_pca3d(
    cluster_data: Annotated[ClusterData, Depends(load_pca_data)],
) -> list[tuple[float, float, float]]:
    pca3d_centroids = cluster_data.pca3d_centroids
    return pca3d_centroids.tolist()


@router.get("/cluster_sizes")
async def cluster_sizes(
    cluster_data: Annotated[ClusterData, Depends(load_pca_data)],
) -> list[int]:
    counts = cluster_data.counts
    return counts.tolist()


@router.get("/cluster_image_id")
async def cluster_image_id(
    cluster_data: Annotated[ClusterData, Depends(load_pca_data)], cluster: ClusterID
) -> list[int]:
    inverted_index_indptr = cluster_data.inverted_index_indptr
    inverted_index_order = cluster_data.inverted_index_order
    image_id_list = cluster_data.image_id_list

    n_clusters = len(inverted_index_indptr) - 1
    if not (0 <= cluster < n_clusters):
        raise HTTPException(
            status_code=400, detail={"error": "cluster index out of range"}
        )

    start = inverted_index_indptr[cluster]
    end = inverted_index_indptr[cluster + 1]
    image_ids = [image_id_list[i] for i in inverted_index_order[start:end]]
    return image_ids


class NearestCentroids(BaseModel):
    row_ids: list[ClusterID]


@router.post("/image_nearest_centroids")
def image_nearest_centroids(
    cluster_data: Annotated[ClusterData, Depends(load_pca_data)],
    image: Annotated[bytes, File()],
    limit: Annotated[int, Form(ge=1, le=100)] = 5,
) -> NearestCentroids:
    image_feature = (
        image_model(cluster_data.embedding).image_features(image)[0].astype(np.float32)
    )

    centroids = cluster_data.centroids.astype(np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / np.maximum(centroid_norms, 1e-12)

    scores = normalized_centroids @ image_feature
    row_ids = np.argsort(-scores)[:limit].astype(int).tolist()

    return NearestCentroids(row_ids=row_ids)
