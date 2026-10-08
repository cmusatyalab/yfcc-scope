# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request

from ..types import (
    ClusterID,
    Embeddings,
    EmbeddingType,
    TarStreamingResponse,
    WIDSDescriptor,
)
from ..utils import generate_webdataset_shard
from .clusters import (
    cluster_wids_descriptor,
    embeddings_to_clusters,
    get_items_in_cluster,
    scoped_wids_descriptor,
)

router = APIRouter()


def get_cluster_base_url(request: Request, embedding_type: EmbeddingType) -> str:
    """Base URL for the clustered webdataset shards served by this router."""
    url = request.url_for("cluster_wids", embedding_type=embedding_type)
    return str(url).rsplit("/", 1)[0] + "/"


ClusterBaseUrl = Annotated[str, Depends(get_cluster_base_url)]


@router.get("/{embedding_type}.json", tags=["webdataset index"], name="cluster_wids")
async def scope_all_clusters(
    base_url: ClusterBaseUrl, embedding_type: EmbeddingType
) -> WIDSDescriptor:
    """Return a WIDS descriptor for all clustered webdataset shards."""
    return await cluster_wids_descriptor(embedding_type, base_url)


@router.post("/{embedding_type}.json", tags=["webdataset index"])
async def scoped_clusters(
    base_url: ClusterBaseUrl,
    embedding_type: EmbeddingType,
    embeddings: Embeddings,
    limit: Annotated[int, Query(ge=0)] = 0,
) -> WIDSDescriptor:
    """Return a WIDS descriptor where the clustered webdataset shards are
    ordered based on the similarity score for the provided embeddings."""
    clusters = embeddings_to_clusters(embedding_type, embeddings)
    return await scoped_wids_descriptor(embedding_type, base_url, clusters, limit)


@router.get(
    "/{embedding_type}-{cluster}.tar",
    response_class=TarStreamingResponse,
    tags=["webdataset shard"],
)
async def get_cluster_shard(
    embedding_type: EmbeddingType,
    cluster: Annotated[ClusterID, Path(title="Cluster ID", ge=0)],
) -> AsyncIterable[bytes]:
    items = get_items_in_cluster(embedding_type, cluster)
    async for chunk in generate_webdataset_shard(items):
        yield chunk
