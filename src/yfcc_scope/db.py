# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time

import psycopg2
from pgvector.psycopg2 import register_vector

from .log import log
from .settings import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def open_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=str(DB_PASSWORD),
        host=DB_HOST,
        port=DB_PORT,
    )


def ensure_hist_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS yfcc_label_conf_hist (
            label TEXT NOT NULL,
            conf_bin SMALLINT NOT NULL,
            box_count BIGINT NOT NULL,
            updated_at INT NOT NULL,
            PRIMARY KEY(label, conf_bin)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS yfcc_images_maxbin_hist (
            max_bin SMALLINT PRIMARY KEY,
            image_count BIGINT NOT NULL,
            updated_at INT NOT NULL
        );
    """)
    conn.commit()
    cur.close()


def ensure_vector_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yfcc_image_label_counts (
                image_file_id TEXT PRIMARY KEY,
                total_bboxes INT NOT NULL,
                counts JSONB NOT NULL,
                updated_at INT NOT NULL
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_yfcc_image_label_counts_updated
            ON yfcc_image_label_counts(updated_at DESC);
        """)
    conn.commit()


def rebuild_vector_table():
    conn = open_conn()
    ensure_vector_table(conn)
    now = int(time.time())
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute("TRUNCATE yfcc_image_label_counts;")
            cur.execute(
                """
                WITH per_label AS (
                    SELECT image_file_id, label, COUNT(*)::int AS cnt
                    FROM bb_table
                    WHERE label IS NOT NULL
                    GROUP BY image_file_id, label
                ),
                per_image AS (
                    SELECT
                        image_file_id,
                        SUM(cnt)::int AS total_bboxes,
                        jsonb_object_agg(label, cnt) AS counts
                    FROM per_label
                    GROUP BY image_file_id
                )
                INSERT INTO yfcc_image_label_counts
                (image_file_id, total_bboxes, counts, updated_at)
                SELECT image_file_id, total_bboxes, counts, %s
                FROM per_image;
            """,
                (now,),
            )
            cur.execute("COMMIT;")
    except Exception as e:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK;")
        conn.close()
        log.exception("Vector table rebuild failed: %s", e)
        raise
    finally:
        conn.close()


def fetch_images_for_labels(labels, limit, offset, conf_ranges=None):
    conn = open_conn()
    cur = conn.cursor()
    conf_sql = ""
    params = []
    if conf_ranges:
        ors = []
        for lo, hi in conf_ranges:
            ors.append("(b.confidence_score >= %s AND b.confidence_score < %s)")
            params += [lo, hi]
        conf_sql = " AND (" + " OR ".join(ors) + ")"

    if labels:
        query = f"""
            SELECT y.image_file_id, y.path
            FROM yfcc_index y
            JOIN bb_table b ON b.image_file_id = y.image_file_id
            WHERE b.label = ANY(%s)
            {conf_sql}
            GROUP BY y.image_file_id, y.path
            HAVING COUNT(DISTINCT b.label) = %s
            ORDER BY MAX(y.ts) DESC
            LIMIT %s OFFSET %s;
        """
        all_params = [labels] + params + [len(set(labels)), limit, offset]
        cur.execute(query, all_params)
    else:
        cur.execute(
            "SELECT y.image_file_id, y.path FROM yfcc_index y "
            "ORDER BY y.ts DESC LIMIT %s OFFSET %s;",
            (limit, offset),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"image_file_id": r[0], "path": r[1]} for r in rows]


def conf_hist_sync(labels):
    sql = """
    WITH x AS (
      SELECT image_file_id,
        FLOOR(confidence_score * 100)::int AS bin,
        COUNT(DISTINCT label) AS labels_hit
      FROM bb_table
      WHERE confidence_score IS NOT NULL AND label = ANY(%s)
      GROUP BY image_file_id, FLOOR(confidence_score * 100)::int
    )
    SELECT bin, COUNT(*) AS image_count FROM x
    WHERE labels_hit = %s GROUP BY bin ORDER BY bin;
    """
    with open_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (labels, len(labels)))
        rows = cur.fetchall()
    counts = {b: c for (b, c) in rows if 0 <= b <= 100}
    return [{"bin": i, "image_count": int(counts.get(i, 0))} for i in range(101)]


def vector_rows_sync(limit, offset, min_total):
    conn = open_conn()
    try:
        with conn.cursor() as cur:
            ensure_vector_table(conn)
            cur.execute(
                """
                SELECT v.image_file_id, y.path, v.total_bboxes, v.counts
                FROM yfcc_image_label_counts v
                JOIN yfcc_index y ON y.image_file_id = v.image_file_id
                WHERE v.total_bboxes >= %s
                ORDER BY y.ts DESC
                LIMIT %s OFFSET %s;
            """,
                (min_total, limit, offset),
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT COALESCE(MAX(updated_at),0) FROM yfcc_image_label_counts;"
            )
            updated_at = cur.fetchone()[0] or 0
        return rows, updated_at
    finally:
        conn.close()


def execute_wrapped_query(raw_sql: str):
    wrapped_sql = f"""
        WITH q AS (
            {raw_sql}
        )
        SELECT
            q.image_file_id,
            y.path,
            0 AS total_bboxes,
            '{{}}'::jsonb AS counts
        FROM q
        JOIN yfcc_index y ON y.image_file_id = q.image_file_id
        LIMIT 1000
    """

    conn = open_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout TO 30000;")
            cur.execute(wrapped_sql)
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()


def execute_query(raw_sql: str):
    conn = open_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(raw_sql)
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()


def fetch_paths_by_ids(image_ids):
    conn = open_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT image_file_id, path FROM yfcc_index "
                "WHERE image_file_id = ANY(%s)",
                (image_ids,),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    except Exception as e:
        log.error("fetch_paths_by_ids failed: %s", e)
        return {}
    finally:
        conn.close()


def search_clip_images(text_feat, limit):
    conn = open_conn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("LOAD 'pg_hint_plan';")
            cur.execute("SET ivfflat.probes = 100;")
            cur.execute(
                """
                /*+ IndexScan(ce clip_embeddings_embedding_idx_ivf) */
                SELECT ce.image_file_id, y.path
                FROM clip_embeddings ce
                JOIN yfcc_index y ON y.image_file_id = ce.image_file_id
                ORDER BY ce.embedding <=> %s
                LIMIT %s;
                """,
                (text_feat, limit),
            )
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()


def search_clip_ids(text_feat, limit):
    conn = open_conn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("LOAD 'pg_hint_plan';")
            cur.execute("SET ivfflat.probes = 500;")
            cur.execute(
                """
                /*+ IndexScan(clip_embeddings clip_embeddings_embedding_idx_ivf) */
                SELECT image_file_id FROM clip_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (text_feat, limit),
            )
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()
