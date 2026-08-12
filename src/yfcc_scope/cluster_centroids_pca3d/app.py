# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from io import BytesIO
from pathlib import Path

import numpy as np
import wids
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..log import log

METHOD = "faiss_kmeans"
BASE_DIR = Path(__file__).resolve().parent
WIDS_JSON_URL = "https://storage.cmusatyalab.org/yfcc100m/yfcc100m.json"

pca_cluster_data = {}
ds = wids.ShardListDataset(WIDS_JSON_URL)


def _load_pca_data(embedding_type: str):
    if pca_cluster_data.get(embedding_type):
        return

    if not embedding_type in ["clip", "dinov3"]:
        raise FileNotFoundError(f"Unknown embedding type: {embedding_type}")

    log.info(f"Loading {embedding_type} data")
    pca3d_centroids = np.load(BASE_DIR / embedding_type / f"{METHOD}_pca3d_centroids.npy")
    assignments = np.load(BASE_DIR / embedding_type / f"{METHOD}_assignments.npy")
    inverted_index_indptr = np.load(BASE_DIR / embedding_type / f"{METHOD}_inverted_index_indptr.npy")
    inverted_index_order = np.load(BASE_DIR / embedding_type / f"{METHOD}_inverted_index_order.npy")

    n_clusters = pca3d_centroids.shape[0]
    counts = np.bincount(assignments, minlength=n_clusters)

    pca_cluster_data[embedding_type] = {
        "pca3d_centroids": pca3d_centroids,
        "assignments": assignments,
        "inverted_index_indptr": inverted_index_indptr,
        "inverted_index_order": inverted_index_order,
        "counts": counts,
    }


def load_pca_data_required(func):
    async def wrapper(request: Request):
        try:
            embedding_type = request.query_params["embedding"]
        except KeyError:
            return JSONResponse({"error": "Missing 'embedding' query parameter"}, status_code=400)

        if embedding_type not in pca_cluster_data:
            try:
                await run_in_threadpool(_load_pca_data, embedding_type)
            except FileNotFoundError:
                return JSONResponse({"error": f"Data for embedding '{embedding_type}' not found"}, status_code=404)
        return await func(request)

    return wrapper


@load_pca_data_required
async def centroids_pca3d(request: Request):
    embedding = request.query_params["embedding"]
    pca3d_centroids = pca_cluster_data[embedding]["pca3d_centroids"]
    return JSONResponse(pca3d_centroids.tolist())


@load_pca_data_required
async def cluster_sizes(request: Request):
    embedding = request.query_params["embedding"]
    counts = pca_cluster_data[embedding]["counts"]
    return JSONResponse(counts.tolist())


@load_pca_data_required
async def cluster_image_indexes(request: Request):
    try:
        cluster_index = int(request.query_params["cluster"])
    except (ValueError, KeyError):
        return JSONResponse({"error": "Invalid or missing 'cluster' query parameter"}, status_code=400)

    embedding = request.query_params["embedding"]
    n_clusters = len(pca_cluster_data[embedding]["inverted_index_indptr"]) - 1
    if not (0 <= cluster_index < n_clusters):
        return JSONResponse({"error": "cluster index out of range"}, status_code=400)

    inverted_index_indptr = pca_cluster_data[embedding]["inverted_index_indptr"]
    inverted_index_order = pca_cluster_data[embedding]["inverted_index_order"]
    start = inverted_index_indptr[cluster_index]
    end = inverted_index_indptr[cluster_index + 1]
    image_indexes = inverted_index_order[start:end]
    return JSONResponse(image_indexes.tolist())


def image_wds(request: Request):
    try:
        idx = int(request.query_params["image_idx"])
    except (ValueError, KeyError):
        return JSONResponse({"error": "Invalid or missing 'image_idx' query parameter"}, status_code=400)

    if not (0 <= idx < len(ds)):
        return JSONResponse({"error": "image_idx out of range"}, status_code=400)

    img = ds[idx][".jpg"]

    buffer = BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


api_routes = [
    Route("/centroids_pca3d", centroids_pca3d),
    Route("/cluster_sizes", cluster_sizes),
    Route("/cluster_image_indexes", cluster_image_indexes),
    Route("/image_wds", image_wds),
]
