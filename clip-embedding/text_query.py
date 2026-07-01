import torch
import open_clip
import os
import psycopg2
import numpy as np
import time
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
    cur.execute("SET ivfflat.probes = 100;")

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


if __name__ == "__main__":
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    neighbor_num = input("Enter number of neighbors to retrieve: ")
    text = input("Enter text query: ")
    text_feat = compute_text_features([text], model, tokenizer, device)[0]

    with open_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            results, query_time = query_ivf(cur, text_feat, int(neighbor_num))

    image_file_ids = [row[0] for row in results]
    print(image_file_ids)
    print(f"Query execution time: {query_time:.4f} seconds")
