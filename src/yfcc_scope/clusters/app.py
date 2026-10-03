# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, FastAPI, Path, Query, Request

from ..scope.slicer import generate_shard
from ..types import (
    ClusterID,
    Embeddings,
    EmbeddingType,
    TarStreamingResponse,
    WIDSDescriptor,
)
from .clusters import (
    cluster_wids_descriptor,
    embeddings_to_clusters,
    get_items_in_cluster,
    scoped_wids_descriptor,
)

router = APIRouter()


@router.get("/{embedding_type}.json", tags=["webdataset index"])
async def scope_all_clusters(
    request: Request, embedding_type: EmbeddingType
) -> WIDSDescriptor:
    """Return a WIDS descriptor for all clustered webdataset shards."""
    base_url = str(request.base_url)
    return await cluster_wids_descriptor(embedding_type, base_url)


@router.post("/{embedding_type}.json", tags=["webdataset index"])
async def scoped_clusters(
    request: Request,
    embedding_type: EmbeddingType,
    embeddings: Embeddings,
    limit: Annotated[int, Query(ge=0)] = 0,
) -> WIDSDescriptor:
    """Return a WIDS descriptor where the clustered webdataset shards are
    ordered based on the similarity score for the provided embeddings."""
    base_url = str(request.base_url)
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
    async for chunk in generate_shard(items):
        yield chunk


app = FastAPI()
app.include_router(router)
