# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.routing import Mount, Route

from .bboxviewer.app import router as bboxviewer_router
from .cluster_centroids_pca3d.app import router as cluster_centroids_router
from .clusters.app import router as clusters_router
from .dashboard.app import router as dashboard_router
from .log import setup_logging
from .pcaviewer.app import router as pcaviewer_router
from .routes import (
    clip_image_query,
    clip_text_query,
    create_scope_clip_image,
    create_scope_clip_text,
    create_scope_coco,
    download_zip,
    run_query,
    run_query_count,
)
from .scope.app import image_router, scope_router

setup_logging()


api_routes = [
    Route("/clip_text_query", clip_text_query, methods=["POST"]),
    Route("/clip_image_query", clip_image_query, methods=["POST"]),
    Route("/run_query", run_query, methods=["POST"]),
    Route("/run_query_count", run_query_count, methods=["POST"]),
    Route("/create_scope_coco", create_scope_coco, methods=["POST"]),
    Route("/create_scope_clip_text", create_scope_clip_text, methods=["POST"]),
    Route("/create_scope_clip_image", create_scope_clip_image, methods=["POST"]),
    Route("/download_zip", download_zip, methods=["POST"]),
    # pulling routes out of fastapi.APIRouter is a hack we need until we have
    # all api endpoints converted to FastAPI
    *bboxviewer_router.routes,
    *cluster_centroids_router.routes,
    *dashboard_router.routes,
    *pcaviewer_router.routes,
]
routes = [
    Mount("/api", routes=api_routes),
]

app = FastAPI(routes=routes)
app.include_router(image_router, prefix="/image")
app.include_router(clusters_router, prefix="/clustered-scope")
app.include_router(scope_router, prefix="/scope")
app.include_router(bboxviewer_router, prefix="/boxviewer")
app.include_router(cluster_centroids_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(pcaviewer_router, prefix="/api")


@app.get("/", response_class=RedirectResponse)
async def home() -> str:
    return "/image-viewer"


# Redirect React frontend application links
viewer_dir = Path(str(files("yfcc_scope") / "dist"))
app.frontend("/", directory=viewer_dir)
