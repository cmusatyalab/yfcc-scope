from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from .log import setup_logging
from .routes import (
    conf_hist,
    download_zip,
    freqs_api,
    home,
    image,
    images_api,
    recalc_freqs,
    recalc_vectors,
    redirect_to_viewer,
    run_query,
    vector_rows_api,
    viewer_app,
    viewer_index,
)

setup_logging()


app = Starlette(
    routes=[
        Route("/", home),
        Route("/image-viewer", redirect_to_viewer),
        Route("/dashboard", redirect_to_viewer),
        Route("/pca3d", redirect_to_viewer),
        Route("/viewer", viewer_index),
        Route("/viewer/{path:path}", viewer_app),
        Route("/image", image),
        Route("/api/images", images_api),
        Route("/api/conf_hist", conf_hist, methods=["GET"]),
        Route("/api/vector_rows", vector_rows_api),
        Route("/api/run_query", run_query, methods=["POST"]),
        Route("/api/download_zip", download_zip, methods=["POST"]),
        Route("/freqs", freqs_api),
        Route("/recalc", recalc_freqs, methods=["POST"]),
        Route("/recalc_vectors", recalc_vectors, methods=["POST"]),
    ]
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://cloudlet032.elijah.cs.cmu.edu:5173",
        "http://128.2.212.50:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
