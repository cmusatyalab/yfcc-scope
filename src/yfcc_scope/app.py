# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .bboxviewer.app import router as bboxviewer_router
from .cluster_centroids_pca3d.app import router as cluster_centroids_router
from .clustered.app import router as clustered_router
from .dashboard.app import router as dashboard_router
from .imageviewer.app import router as imageviewer_router
from .log import setup_logging
from .pcaviewer.app import router as pcaviewer_router
from .scope.app import image_router, scope_router

setup_logging()


app = FastAPI()
app.include_router(image_router, prefix="/image")
app.include_router(bboxviewer_router, prefix="/boxviewer")
app.include_router(clustered_router, prefix="/clustered")
app.include_router(scope_router, prefix="/scope")
app.include_router(cluster_centroids_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(imageviewer_router, prefix="/api")
app.include_router(pcaviewer_router, prefix="/api")


@app.get("/", response_class=RedirectResponse)
async def home() -> str:
    """Redirect to image viewer home"""
    return "/image-viewer"


# Redirect React frontend application links
viewer_dir = Path(str(files("yfcc_scope") / "dist"))
app.frontend("/", directory=viewer_dir)
