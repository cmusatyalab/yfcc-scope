import argparse
import faiss
import numpy as np
from pathlib import Path

DIR = "/home/ubuntu/yfcc-scope/embeddings"


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster image embeddings for the YFCC dataset")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["clip", "dinov3"],
        help="Embedding model (clip or dinov3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    emb_dir = Path(DIR) / args.method
    emb = np.load(emb_dir / "embeddings_cleaned.npy")
    assert len(emb) == 10016544 - 20719

    # Clustering
    dim = emb.shape[1]
    ncentroids = 10000
    niter = 20
    kmeans = faiss.Kmeans(dim, ncentroids, niter=niter, verbose=True, max_points_per_centroid=1100)

    kmeans.train(emb)

    # Save centroids and cluster assignments
    np.save(emb_dir / "faiss_kmeans_centroids.npy", kmeans.centroids)
    assignments = kmeans.index.search(emb, 1)[1].reshape(-1).astype(np.int32)
    np.save(emb_dir / "faiss_kmeans_assignments.npy", assignments)

    # Save inverted index
    order = np.argsort(assignments, kind="stable")
    sorted_cluster_ids = assignments[order]
    indptr = np.searchsorted(sorted_cluster_ids, np.arange(ncentroids + 1))
    # image indexes sorted by cluster id
    np.save(emb_dir / "faiss_kmeans_inverted_index_order.npy", order)
    # indptr[i] is the index in order where cluster i starts, and indptr[i+1] is where it ends
    np.save(emb_dir / "faiss_kmeans_inverted_index_indptr.npy", indptr)
