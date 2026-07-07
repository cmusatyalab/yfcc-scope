import os
import time

import psycopg2
from pgvector.psycopg2 import register_vector

from .constants import LABELS
from .log import log
from .utils import conf_to_bin


def open_conn():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "yfcc"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", 5432)),
    )


def fetch(image_file_id: str):
    try:
        conn = open_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT y.path,
                   b.bounding_box_number,
                   b.label,
                   b.center_x,
                   b.center_y,
                   b.width,
                   b.height,
                   b.confidence_score
            FROM yfcc_index y
            LEFT JOIN bb_table b
              ON y.image_file_id = b.image_file_id
            WHERE y.image_file_id = %s
            ORDER BY COALESCE(b.bounding_box_number, 0)
            """,
            (image_file_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        log.error("DB fetch failed for image_file_id=%r: %s", image_file_id, e)
        return (None, [])

    if not rows:
        log.warning("image_file_id %r not found in yfcc_index", image_file_id)
        return (None, [])

    path = rows[0][0]
    cleaned = [(n, label, cx, cy, w, h, conf) for _, n, label, cx, cy, w, h, conf in rows if label is not None]
    return (path, cleaned)


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


def rebuild_histograms():
    conn = open_conn()
    ensure_hist_tables(conn)
    cur = conn.cursor()
    now = int(time.time())
    try:
        cur.execute("BEGIN;")
        cur.execute("TRUNCATE yfcc_label_conf_hist;")
        cur.execute("TRUNCATE yfcc_images_maxbin_hist;")
        cur.execute(
            """
            INSERT INTO yfcc_label_conf_hist (label, conf_bin, box_count, updated_at)
            SELECT
                b.label,
                GREATEST(0, LEAST(100, FLOOR(b.confidence_score * 100.0 + 0.000001)))::SMALLINT AS conf_bin,
                COUNT(*) AS box_count,
                %s AS updated_at
            FROM bb_table b
            WHERE b.confidence_score IS NOT NULL
            GROUP BY b.label, conf_bin;
        """,
            (now,),
        )
        cur.execute(
            """
            WITH maxbin AS (
                SELECT
                    image_file_id,
                    GREATEST(0, LEAST(100, FLOOR(MAX(confidence_score) * 100.0 + 0.000001)))::SMALLINT AS max_bin
                FROM bb_table
                WHERE confidence_score IS NOT NULL
                GROUP BY image_file_id
            )
            INSERT INTO yfcc_images_maxbin_hist (max_bin, image_count, updated_at)
            SELECT max_bin, COUNT(*)::BIGINT, %s
            FROM maxbin
            GROUP BY max_bin;
        """,
            (now,),
        )
        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        conn.close()
        log.exception("Histogram rebuild failed: %s", e)
        raise
    finally:
        cur.close()
        conn.close()


def read_label_counts_at_threshold(min_conf: float):
    tb = conf_to_bin(min_conf)
    conn = open_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT label, COALESCE(SUM(box_count),0)
            FROM yfcc_label_conf_hist
            WHERE conf_bin >= %s
            GROUP BY label;
        """,
            (tb,),
        )
        label_rows = cur.fetchall()
        cur.execute("SELECT COALESCE(MAX(updated_at),0) FROM yfcc_label_conf_hist;")
        updated_at = cur.fetchone()[0] or 0
    finally:
        cur.close()
        conn.close()

    counts = {lab: 0 for lab in LABELS}
    for lab, cnt in label_rows:
        if lab in counts:
            counts[lab] = int(cnt)
    total_boxes = sum(counts.values())
    return counts, total_boxes, updated_at


def read_images_with_boxes_at_threshold(min_conf: float) -> int:
    tb = conf_to_bin(min_conf)
    conn = open_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT COALESCE(SUM(image_count),0)
            FROM yfcc_images_maxbin_hist
            WHERE max_bin >= %s;
        """,
            (tb,),
        )
        n = cur.fetchone()[0] or 0
    finally:
        cur.close()
        conn.close()
    return int(n)


def read_total_images_yfcc() -> int:
    conn = open_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM yfcc_index;")
        n = cur.fetchone()[0] or 0
    finally:
        cur.close()
        conn.close()
    return int(n)


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
                INSERT INTO yfcc_image_label_counts (image_file_id, total_bboxes, counts, updated_at)
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
            "SELECT y.image_file_id, y.path FROM yfcc_index y ORDER BY y.ts DESC LIMIT %s OFFSET %s;",
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
    SELECT bin, COUNT(*) AS image_count FROM x WHERE labels_hit = %s GROUP BY bin ORDER BY bin;
    """
    with open_conn() as conn:
        with conn.cursor() as cur:
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
            cur.execute("SELECT COALESCE(MAX(updated_at),0) FROM yfcc_image_label_counts;")
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
            cur.execute("SELECT image_file_id, path FROM yfcc_index WHERE image_file_id = ANY(%s)", (image_ids,))
            return {row[0]: row[1] for row in cur.fetchall()}
    except Exception as e:
        log.error("fetch_paths_by_ids failed: %s", e)
        return {}
    finally:
        conn.close()


def execute_clip_query(text_feat, limit):
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
