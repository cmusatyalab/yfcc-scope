# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .. import settings
from ..scope.slicer import generate_shard
from .clusters import (
    embeddings_to_clusters,
    get_items_in_cluster,
    scoped_wids_descriptor,
)

if TYPE_CHECKING:
    from starlette.requests import Request


async def get_cluster_scope(request: Request) -> JSONResponse:
    embedding = request.path_params["embedding"]
    base_url = str(request.base_url)

    try:
        body = await request.json()
        embeddings = body.get("embeddings")
    except json.decoder.JSONDecodeError:
        embeddings = None

    clusters = embeddings_to_clusters(embedding, embeddings)

    wids_descriptor = await scoped_wids_descriptor(embedding, base_url, clusters)
    return JSONResponse(wids_descriptor)


async def get_cluster_shard(request: Request) -> StreamingResponse:
    embedding = request.path_params["embedding"]
    cluster = request.path_params["cluster"]

    items = get_items_in_cluster(embedding, cluster)
    object_stream = generate_shard(items)
    return StreamingResponse(object_stream, media_type="application/x-tar")


clusters_routes = [
    Route("/{embedding}.json", get_cluster_scope, methods=["GET", "POST", "PUT"]),
    Route("/{embedding}-{cluster:int}.tar", get_cluster_shard),
]

app = Starlette(
    debug=settings.DEBUG,
    routes=clusters_routes,
)
