# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from io import BytesIO
from pathlib import Path

import json
import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..log import log
from ..embedding import clip_image_features, dinov3_image_features

METHOD = "faiss_kmeans"
BASE_DIR = Path(__file__).resolve().parent

pca_cluster_data = {}


def _load_pca_data(embedding_type: str):
    if pca_cluster_data.get(embedding_type):
        return

    if not embedding_type in ["clip", "dinov3", "clip-2", "dinov3-2"]:
        raise FileNotFoundError(f"Unknown embedding type: {embedding_type}")

    log.info(f"Loading {embedding_type} data")
    centroids = np.load(BASE_DIR / embedding_type / f"{METHOD}_centroids.npy")
    pca3d_centroids = np.load(BASE_DIR / embedding_type / f"{METHOD}_pca3d_centroids.npy")
    assignments = np.load(BASE_DIR / embedding_type / f"{METHOD}_assignments.npy")
    inverted_index_indptr = np.load(BASE_DIR / embedding_type / f"{METHOD}_inverted_index_indptr.npy")
    inverted_index_order = np.load(BASE_DIR / embedding_type / f"{METHOD}_inverted_index_order.npy")
    with open(BASE_DIR / embedding_type / "image_id_list.json") as f:
        image_id_list = json.load(f)

    n_clusters = pca3d_centroids.shape[0]
    counts = np.bincount(assignments, minlength=n_clusters)

    pca_cluster_data[embedding_type] = {
        "centroids": centroids,
        "pca3d_centroids": pca3d_centroids,
        "assignments": assignments,
        "inverted_index_indptr": inverted_index_indptr,
        "inverted_index_order": inverted_index_order,
        "counts": counts,
        "image_id_list": image_id_list,
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
async def cluster_image_id(request: Request):
    try:
        cluster_index = int(request.query_params["cluster"])
    except (ValueError, KeyError):
        return JSONResponse({"error": "Invalid or missing 'cluster' query parameter"}, status_code=400)

    embedding = request.query_params["embedding"]
    inverted_index_indptr = pca_cluster_data[embedding]["inverted_index_indptr"]
    inverted_index_order = pca_cluster_data[embedding]["inverted_index_order"]
    image_id_list = pca_cluster_data[embedding]["image_id_list"]

    n_clusters = len(inverted_index_indptr) - 1
    if not (0 <= cluster_index < n_clusters):
        return JSONResponse({"error": "cluster index out of range"}, status_code=400)

    start = inverted_index_indptr[cluster_index]
    end = inverted_index_indptr[cluster_index + 1]
    image_ids = [image_id_list[i] for i in inverted_index_order[start:end]]
    return JSONResponse(image_ids)

@load_pca_data_required
async def image_nearest_centroids(request: Request):
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "Invalid form data"}, status_code=400)

    image_file = form.get("image")
    if not image_file:
        return JSONResponse({"error": "image file is required"}, status_code=400)

    try:
        limit = int(form.get("limit", "5"))
    except (ValueError, TypeError):
        return JSONResponse({"error": "limit must be an integer"}, status_code=400)
    limit = max(1, min(100, limit))

    embedding = request.query_params["embedding"]
    image_bytes = await image_file.read()

    def find_nearest():
        if embedding.startswith("dinov3"):
            image_feature = dinov3_image_features(image_bytes)[0].astype(np.float32)
        else:
            image_feature = clip_image_features(image_bytes)[0].astype(np.float32)
        centroids = pca_cluster_data[embedding]["centroids"].astype(np.float32)
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        normalized_centroids = centroids / np.maximum(centroid_norms, 1e-12)
        scores = normalized_centroids @ image_feature
        return np.argsort(-scores)[:limit].astype(int).tolist()

    try:
        row_ids = await run_in_threadpool(find_nearest)
    except Exception as error:
        log.exception("image_nearest_centroids failed")
        return JSONResponse({"error": str(error)}, status_code=500)

    return JSONResponse({"row_ids": row_ids})


api_routes = [
    Route("/centroids_pca3d", centroids_pca3d),
    Route("/cluster_sizes", cluster_sizes),
    Route("/cluster_image_id", cluster_image_id),
    Route("/image_nearest_centroids", image_nearest_centroids, methods=["POST"]),
]
