import os
import psycopg2
import wids
import numpy as np
from tqdm import tqdm
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_batch
from psycopg2.errors import ForeignKeyViolation

TOTAL_NUM = 10016544
YFCC_URL = "https://storage.cmusatyalab.org/yfcc100m/yfcc100m.json"
EMBEDDING_PATH = "/home/ubuntu/yfcc-scope/clip-embedding/yfcc_image_embeddings.npy"


def open_conn():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "yfcc"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", 5432)),
    )


def batch_insert_embeddings(conn, cur, batch_data, insert_query):
    try:
        execute_batch(cur, insert_query, batch_data, page_size=100)
        conn.commit()

    except ForeignKeyViolation:
        # The batch have keys that don't exist in yfcc_index
        conn.rollback()

        for single_id, single_emb in batch_data:
            try:
                cur.execute(insert_query, (single_id, single_emb))
                conn.commit()
            except ForeignKeyViolation:
                conn.rollback()
                print(f"Skipping {single_id} due to missing foreign key reference.")
                continue


if __name__ == "__main__":
    conn = open_conn()
    cur = conn.cursor()
    register_vector(conn)  # Register the vector type support

    cur.execute("""
        CREATE TABLE IF NOT EXISTS clip_embeddings (
            image_file_id TEXT PRIMARY KEY,
            embedding HALFVEC(512),
            FOREIGN KEY (image_file_id) REFERENCES yfcc_index(image_file_id) ON DELETE CASCADE
        );
    """)
    conn.commit()

    import wids.wids_decode

    ds = wids.ShardListDataset(YFCC_URL, transformations=[wids.wids_decode.decode_basic])
    emb = np.load(EMBEDDING_PATH)

    insert_query = """
        INSERT INTO clip_embeddings (image_file_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (image_file_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """

    BATCH_SIZE = 1000
    batch_data = []

    for i in tqdm(range(TOTAL_NUM)):
        image_json = ds[i][".json"]
        image_file_id = f"{image_json['image_id']}_{image_json['image_crc']}"
        batch_data.append((image_file_id, emb[i]))

        if len(batch_data) >= BATCH_SIZE:
            batch_insert_embeddings(conn, cur, batch_data, insert_query)
            batch_data = []

    if batch_data:
        batch_insert_embeddings(conn, cur, batch_data, insert_query)

    cur.close()
    conn.close()
