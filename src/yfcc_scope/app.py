# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from importlib.resources import files

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from .log import setup_logging
from .routes import (
    boxviewer,
    clip_image_query,
    clip_text_query,
    conf_hist,
    create_scope_clip_image,
    create_scope_clip_text,
    create_scope_coco,
    download_zip,
    freqs_api,
    image,
    images_api,
    recalc_freqs,
    recalc_vectors,
    run_query,
    run_query_count,
    vector_rows_api,
)

setup_logging()


async def home(request: Request) -> RedirectResponse:
    return RedirectResponse(url="/image-viewer", status_code=302)


app = Starlette(
    routes=[
        Route("/", home),
        Route("/boxviewer", boxviewer, name="boxviewer"),
        Route("/image", image),
        Route("/api/images", images_api),
        Route("/api/conf_hist", conf_hist, methods=["GET"]),
        Route("/api/vector_rows", vector_rows_api),
        Route("/api/clip_text_query", clip_text_query, methods=["POST"]),
        Route("/api/clip_image_query", clip_image_query, methods=["POST"]),
        Route("/api/run_query", run_query, methods=["POST"]),
        Route("/api/run_query_count", run_query_count, methods=["POST"]),
        Route("/api/create_scope_coco", create_scope_coco, methods=["POST"]),
        Route("/api/create_scope_clip_text", create_scope_clip_text, methods=["POST"]),
        Route(
            "/api/create_scope_clip_image", create_scope_clip_image, methods=["POST"]
        ),
        Route("/api/download_zip", download_zip, methods=["POST"]),
        Route("/freqs", freqs_api),
        Route("/recalc", recalc_freqs, methods=["POST"]),
        Route("/recalc_vectors", recalc_vectors, methods=["POST"]),
    ]
)

app.mount("/static", StaticFiles(packages=[("yfcc_scope", "static")]), name="static")

# Redirect React frontend application links
viewer_dir = files("yfcc_scope") / "dist"
app.mount("/assets", StaticFiles(directory=str(viewer_dir / "assets"), html=True))
app.mount(
    "/image-viewer", StaticFiles(directory=str(viewer_dir), html=True), name="viewer"
)
app.mount("/dashboard", StaticFiles(directory=str(viewer_dir), html=True))
app.mount("/pca3d", StaticFiles(directory=str(viewer_dir), html=True))
