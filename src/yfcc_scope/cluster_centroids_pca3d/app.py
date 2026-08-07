# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from io import BytesIO
from pathlib import Path

import numpy as np
import wids
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..log import log

METHOD = "faiss_kmeans"
BASE_DIR = Path(__file__).resolve().parent
WIDS_JSON_URL = "https://storage.cmusatyalab.org/yfcc100m/yfcc100m.json"

pca3d_centroids = np.load(BASE_DIR / f"{METHOD}_pca3d_centroids.npy")
assignments = np.load(BASE_DIR / f"{METHOD}_assignments.npy")
inverted_index_indptr = np.load(BASE_DIR / f"{METHOD}_inverted_index_indptr.npy")
inverted_index_order = np.load(BASE_DIR / f"{METHOD}_inverted_index_order.npy")

n_clusters = pca3d_centroids.shape[0]
n_samples = assignments.shape[0]
counts = np.bincount(assignments, minlength=n_clusters)

ds = wids.ShardListDataset(WIDS_JSON_URL)


async def centroids_pca3d(request):
    return JSONResponse(pca3d_centroids.tolist())


async def cluster_sizes(request):
    return JSONResponse(counts.tolist())


async def cluster_image_indexes(request):
    cluster_index = int(request.query_params["cluster"])
    start = inverted_index_indptr[cluster_index]
    end = inverted_index_indptr[cluster_index + 1]
    image_indexes = inverted_index_order[start:end]
    return JSONResponse(image_indexes.tolist())


async def image_wds(request):
    idx = int(request.query_params["image_idx"])
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
