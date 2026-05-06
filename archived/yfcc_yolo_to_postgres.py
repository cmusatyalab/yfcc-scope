from ultralytics import YOLO
import re
from PIL import Image
from io import BytesIO
from IPython.display import display
import psycopg2, psycopg2.extras
import json, os, hashlib
import requests
import time  # <-- for timing / ETA

# -------------------------
# YOLO setup + helpers
# -------------------------

yolo = YOLO("yolo11s.pt") 

def space_to_underscore(label: str) -> str:
    return label.lower().replace(" ", "_")
    
def yolo_names():
    names = yolo.model.names
    return [space_to_underscore(names[i]) for i in range(len(names))]

YOLO_LABELS = yolo_names()
print(YOLO_LABELS)

def image_file_id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

# def path_to_link(p: str) -> str:
#     rel = os.path.relpath(p, IMG_DIR)  # relative to dataset root
#     return f"{BASE_URL}/{rel.replace(os.sep, '/')}"  # URL-style slashes

# def flickr_link(p: str) -> str:
#     photo_id = image_id_from_path(p)
#     return f"https://www.flickr.com/photo.gne?id={photo_id}"

# -------------------------
# DB connection + schema
# -------------------------
def open_conn():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "yfcc"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", 5432)),
    )

def make_upsert_sql():
    base = ['image_file_id', 'path', 'total_bboxes']
    all_cols = base + YOLO_LABELS
    insert_cols = ", ".join(f'"{c}"' for c in all_cols)
    placeholders = ", ".join(["%s"] * len(all_cols))

    set_parts = [f'"{c}"=EXCLUDED."{c}"' for c in base[1:]]  # path,total_bboxes
    set_parts += [f'"{c}"=EXCLUDED."{c}"' for c in YOLO_LABELS]
    set_parts.append('ts=now()')

    return f"""
    INSERT INTO yfcc_index ({insert_cols})
    VALUES ({placeholders})
    ON CONFLICT (image_file_id) DO UPDATE
    SET {", ".join(set_parts)};
    """

UPSERT_LABELS_SQL = make_upsert_sql()

def upsert_image_into_yfcc_index(cur, path, counts):
    upsert_sql = UPSERT_LABELS_SQL

    total_bboxes = sum(counts.values())
    row_vals = [int(counts.get(lbl, 0)) for lbl in YOLO_LABELS]
    cur.execute(
        upsert_sql,
        (
            image_file_id_from_path(path),
            path,
            total_bboxes,
            *row_vals,
        ),
    )

# NEW: bulk insert for all bboxes in a chunk
def bulk_insert_bboxes(cur, bbox_rows):
    """
    bbox_rows: list of tuples
      (image_file_id, bounding_box_number, label, confidence_score,
       center_x, center_y, width, height)
    """
    if not bbox_rows:
        return
    sql = """
        INSERT INTO bb_table (
            image_file_id, bounding_box_number, label, confidence_score,
            center_x, center_y, width, height
        )
        VALUES %s;
    """
    psycopg2.extras.execute_values(cur, sql, bbox_rows, page_size=1000)

# -------------------------
# YOLO inference (streaming)
# -------------------------

# global requests.Session for connection reuse
SESSION = requests.Session()

def yolo_counts_and_bboxes_from_url(url: str, conf: float = 0.4):
    """
    Download image bytes via HTTP and run YOLO in-memory (no disk writes).
    Uses a shared requests.Session to reuse TCP connections.
    """
    resp = SESSION.get(url, timeout=10)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")

    res = yolo(img, conf=conf, verbose=False)[0]

    names  = yolo.model.names
    xyxy   = res.boxes.xyxy.cpu().numpy()
    cls_id = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()

    labels = [space_to_underscore(names[int(c)]) for c in cls_id]

    h, w = res.orig_shape
    
    # Build counts dictionary
    counts = {}
    for n in labels:
        counts[n] = counts.get(n, 0) + 1

    # Build per-detection list
    bboxes = []
    for i in range(len(labels)):
        x_min, y_min, x_max, y_max = xyxy[i].tolist()
    
        center_x = float((x_min + x_max) / 2 / w)
        center_y = float((y_min + y_max) / 2 / h)
        width = float((x_max - x_min) / w)
        height = float((y_max - y_min) / h)
    
        bboxes.append(
            {
                "bbox_xywh": [center_x, center_y, width, height],
                "label": labels[i],
                "score": float(scores[i])
            }
        )


    return counts, bboxes

# -------------------------
# Small helper to pretty-print ETA
# -------------------------

def format_seconds(sec: float) -> str:
    if sec < 0:
        sec = 0
    m, s = divmod(int(sec + 0.5), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# -------------------------
# Schema creation
# -------------------------

conn = open_conn()
cur = conn.cursor()
print("creating/ensuring yfcc_index + bb_table")
# print("trying to truncate bbtable")
# cur.execute("TRUNCATE TABLE bb_table;")
# print("trying to truncate yfcc_index")
# cur.execute("TRUNCATE TABLE yfcc_index CASCADE;")
# print("done truncating")
cur.execute("""
CREATE TABLE IF NOT EXISTS yfcc_index (
  image_file_id   TEXT PRIMARY KEY,
  path       TEXT NOT NULL,
  ts         TIMESTAMPTZ DEFAULT now(),
  total_bboxes INT DEFAULT 0
);
""")
# make sure it exists even if table was already created before
cur.execute('ALTER TABLE yfcc_index ADD COLUMN IF NOT EXISTS total_bboxes INT DEFAULT 0;')

for col in YOLO_LABELS:
    cur.execute(f'ALTER TABLE yfcc_index ADD COLUMN IF NOT EXISTS "{col}" INT DEFAULT 0;')
cur.execute("""
CREATE TABLE IF NOT EXISTS bb_table (
  image_file_id TEXT REFERENCES yfcc_index(image_file_id) ON DELETE CASCADE,
  bounding_box_number    INT  NOT NULL,
  label                  TEXT NOT NULL,
  confidence_score        REAL  NOT NULL,
  center_x        REAL  NOT NULL, 
  center_y        REAL  NOT NULL, 
  width        REAL  NOT NULL, 
  height        REAL  NOT NULL  
);
""")
conn.commit()
print("schema ready")
print("v20mil")

# -------------------------
# Main loop over dataset
# -------------------------

# for path in imgs[:7000]:
#     counts, bboxes = yolo_counts_and_bboxes(path)
#     # display(Image.open(path))
#     # print(path, counts)
#     # if (len(counts) == 0):
#         # print("problem!! " + path)
    
#     upsert_image_into_yfcc_index(cur, path, counts)
#     upsert_all_bboxes_for_image_into_bb_table(cur, image_id_from_path(path), bboxes)
#     print(".")


BASE_URL      = "https://yfcc100m.cmusatyalab.org"
chunk_size    = 4000   # tune as needed
last_image_id = 19_999_927
MAX_IMAGE_ID  = 500_000_000  # <-- stop at x million

# Figure out how many images we *intend* to process total
cur.execute(
    """
    SELECT COUNT(*)
    FROM dataset
    WHERE image_id > %s
      AND image_id <= %s;
    """,
    (last_image_id, MAX_IMAGE_ID),
)
total_target = cur.fetchone()[0]
print(f"Total images in range ({last_image_id}, {MAX_IMAGE_ID}]: {total_target}")

processed = 0
start_time = time.time()

while True:
    cur.execute(
        """
        SELECT image_id, set_id, data_id, bucket_id, image_crc
        FROM dataset
        WHERE image_id > %s
          AND image_id <= %s
        ORDER BY image_id
        LIMIT %s;
        """,
        (last_image_id, MAX_IMAGE_ID, chunk_size),
    )
    rows = cur.fetchall()
    print(f"Fetched {len(rows)} rows")
    if not rows:
        break  # done

    bbox_buffer = []  # collect all bbox rows for this chunk

    for row in rows:
        image_id  = row[0]
        set_id    = row[1]
        data_id   = row[2]
        bucket_id = row[3]
        image_crc = row[4]

        image_file_id = f"{image_id}_{image_crc}"
        url = f"{BASE_URL}/set_{set_id}/data_{data_id}/images/{bucket_id}/{image_file_id}.jpg"

        try:
            counts, bboxes = yolo_counts_and_bboxes_from_url(url)
            upsert_image_into_yfcc_index(cur, url, counts)

            # add bboxes for bulk insert
            for i, box in enumerate(bboxes, start=1):
                center_x, center_y, width, height = box["bbox_xywh"]
                bbox_buffer.append(
                    (
                        image_file_id,
                        i,
                        box["label"],
                        box["score"],
                        center_x,
                        center_y,
                        width,
                        height,
                    )
                )

        except Exception as e:
            print(f"⚠️ error on image_id={image_id} ({url}): {e}")

    # bulk insert all bb_table rows for this chunk
    bulk_insert_bboxes(cur, bbox_buffer)
    conn.commit()

    # progress + ETA
    processed += len(rows)
    last_image_id = rows[-1][0]  # advance to last image_id
    elapsed = time.time() - start_time
    if elapsed > 0 and processed > 0:
        imgs_per_sec = processed / elapsed
        if total_target > 0:
            remaining = total_target - processed
            eta_sec = remaining / imgs_per_sec if imgs_per_sec > 0 else 0
            frac = processed / total_target
            print(
                f"✅ Up to image_id={last_image_id} | "
                f"{processed}/{total_target} ({frac:.2%}) | "
                f"{imgs_per_sec:.2f} img/s | ETA ~ {format_seconds(eta_sec)}"
            )
        else:
            print(
                f"✅ Up to image_id={last_image_id} | "
                f"processed={processed} | {imgs_per_sec:.2f} img/s"
            )
    else:
        print(f"✅ Up to image_id={last_image_id}")

    if last_image_id >= MAX_IMAGE_ID:
        print(f"🎯 Reached MAX_IMAGE_ID={MAX_IMAGE_ID}, stopping.")
        break

conn.commit()
cur.close()
conn.close()

print("✅ Finished upserting rows into yfcc_index and bb_table (up to 20,000,000)")
