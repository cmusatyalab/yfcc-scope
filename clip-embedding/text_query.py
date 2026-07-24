import torch
import open_clip
import os
import psycopg2
import numpy as np
import time
import argparse
from pgvector.psycopg2 import register_vector


def open_conn():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "yfcc"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", 5432)),
    )


def compute_text_features(texts, model, tokenizer, device=None):
    text_input = tokenizer(texts)
    with torch.no_grad(), torch.autocast(device):
        text_feat = model.encode_text(text_input)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.cpu().numpy().astype(np.float16, copy=False)


def query_ivf(cur, text_feat, neighbor_num):
    cur.execute("LOAD 'pg_hint_plan';")
    cur.execute("SET ivfflat.probes = 500;")

    start_time = time.time()
    cur.execute(
        """
        /*+ IndexScan(clip_embeddings clip_embeddings_embedding_idx_ivf) */
        SELECT image_file_id FROM clip_embeddings
        ORDER BY embedding <=> %s
        LIMIT %s;
        """,
        (text_feat, neighbor_num),
    )
    query_time = time.time() - start_time

    return cur.fetchall(), query_time


def query_hnsw(cur, text_feat, neighbor_num):
    cur.execute("LOAD 'pg_hint_plan';")
    cur.execute("SET hnsw.ef_search = %s;", (neighbor_num,))

    start_time = time.time()
    cur.execute(
        """
        /*+ IndexScan(clip_embeddings clip_embeddings_embedding_idx_hnsw) */
        SELECT image_file_id FROM clip_embeddings
        ORDER BY embedding <=> %s
        LIMIT %s;
        """,
        (text_feat, neighbor_num),
    )
    query_time = time.time() - start_time

    return cur.fetchall(), query_time


def parse_args():
    parser = argparse.ArgumentParser(description="Run a text similarity query against CLIP embeddings")
    parser.add_argument("-q", "--query", required=True, type=str, help="Text query to search for")
    parser.add_argument("-n", "--neighbor", required=True, type=int, help="Number of nearest neighbors to retrieve")
    parser.add_argument("-m", "--method", type=str, default="ivf", help="Indexing method to use (ivf or hnsw)")
    parser.add_argument(
        "-p", "--print-ids", action="store_true", help="Print image file IDs of the retrieved results (default: False)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_feat = compute_text_features([args.query], model, tokenizer, device)[0]

    with open_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if args.method == "ivf":
                results, query_time = query_ivf(cur, text_feat, args.neighbor)
            elif args.method == "hnsw":
                results, query_time = query_hnsw(cur, text_feat, args.neighbor)
            else:
                raise ValueError("Invalid method. Use 'ivf' or 'hnsw'.")

    image_file_ids = [row[0] for row in results]
    if args.print_ids:
        print(image_file_ids)
    print(f"Number of results: {len(image_file_ids)}")
    print(f"Query execution time: {query_time:.4f} seconds")
