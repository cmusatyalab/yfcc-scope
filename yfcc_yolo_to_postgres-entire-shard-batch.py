from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import psycopg2, psycopg2.extras
import os
import time
import webdataset as wds
import json
from webdataset.shardlists import expand_urls

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

def extract_image_file_id(sample):
    """
    Extract image_file_id in format: {image_id}_{image_crc}
    Try JSON metadata first, then fall back to __key__
    """
    # Try to get from JSON metadata if available
    metadata_bytes = sample["json"]
    # print(metadata)
    metadata = json.loads(metadata_bytes)
    image_id = str(metadata['image_id'])
    image_crc = str(metadata['image_crc'])
    res = f"{image_id}_{image_crc}"
    # res = "hello"
    # print(res)
    return res

# def make_image_url_from_metadata(metadata):
#     """
#     Constructs a Flickr-style URL for yfcc100m.cmusatyalab.org
#     Example:
#     https://yfcc100m.cmusatyalab.org/set_2/data_2/images/20/71278857_624cf540cb.jpg
#     """
    
#     # This matches your desired format
#     return f"https://yfcc100m.cmusatyalab.org/set_2/data_2/images/......jpg"
def make_image_url_from_metadata(cur, image_file_id: str):
    """
    Look up metadata table using image_file_id and construct:

    https://yfcc100m.cmusatyalab.org/set_{set_id}/data_{data_id}/images/{bucket_id}/{image_id}_{image_crc}.jpg
    """

    sql = """
        SELECT image_id, set_id, data_id, bucket_id, image_crc
        FROM public.metadata
        WHERE image_file_id = %s
        LIMIT 1;
    """

    cur.execute(sql, (image_file_id,))
    row = cur.fetchone()

    if not row:
        return None  # or "" if you prefer

    image_id, set_id, data_id, bucket_id, image_crc = row

    BASE_URL = "https://yfcc100m.cmusatyalab.org"
    print(".")

    return (
        f"{BASE_URL}/set_{set_id}/data_{data_id}/images/"
        f"{bucket_id}/{image_id}_{image_crc}.jpg"
    )

def fetch_metadata_map(cur, image_file_ids):
    """
    image_file_id -> (image_id, set_id, data_id, bucket_id, image_crc)
    """
    if not image_file_ids:
        return {}
    sql = """
        SELECT image_file_id, image_id, set_id, data_id, bucket_id, image_crc
        FROM public.metadata
        WHERE image_file_id = ANY(%s);
    """
    cur.execute(sql, (list(image_file_ids),))
    rows = cur.fetchall()
    return {
        image_file_id: (image_id, set_id, data_id, bucket_id, image_crc)
        for (image_file_id, image_id, set_id, data_id, bucket_id, image_crc) in rows
    }

def make_image_url_from_row(row):
    image_id, set_id, data_id, bucket_id, image_crc = row
    BASE_URL = "https://yfcc100m.cmusatyalab.org"
    return f"{BASE_URL}/set_{set_id}/data_{data_id}/images/{bucket_id}/{image_id}_{image_crc}.jpg"

    
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

def upsert_image_into_yfcc_index(cur, image_file_id, path, counts):
    total_bboxes = sum(counts.values())
    row_vals = [int(counts.get(lbl, 0)) for lbl in YOLO_LABELS]
    cur.execute(
        UPSERT_LABELS_SQL,
        (
            image_file_id,
            path,
            total_bboxes,
            *row_vals,
        ),
    )

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
# YOLO inference (from bytes)
# -------------------------

def yolo_counts_and_bboxes_from_bytes(img_bytes: bytes, conf: float = 0.4):
    """
    Run YOLO inference on image bytes (from webdataset).
    """
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
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

cur.execute("""
CREATE TABLE IF NOT EXISTS yfcc_index (
  image_file_id   TEXT PRIMARY KEY,
  path       TEXT NOT NULL,
  ts         TIMESTAMPTZ DEFAULT now(),
  total_bboxes INT DEFAULT 0
);
""")
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

# # Clear both tables (add this line here!)
# cur.execute("TRUNCATE TABLE bb_table, yfcc_index CASCADE;")
# print("✅ Tables cleared")
# 

conn.commit()
print("schema ready")
print("ver feb9")

# -------------------------
# Main loop using webdataset
# -------------------------

BASE_TEMPLATE = "https://storage.cmusatyalab.org/yfcc100m/yfcc100m-{shard:06d}.tar"
START_SHARD = 0
MAX_SHARDS = 403  # 000000..000402 inclusive

processed = 0
start_time = time.time()

print(f"Starting processing from shard {START_SHARD:06d}")

for shard in range(START_SHARD, MAX_SHARDS):
    shard_url = BASE_TEMPLATE.format(shard=shard)
    print(f"\n--- Processing shard {shard:06d}: {shard_url} ---")

    # New dataset PER shard (this is the key change)
    dataset = wds.WebDataset(
        shard_url,
        shardshuffle=False,
        handler=wds.warn_and_continue,
    )

    bbox_buffer = []
    batch_samples = []   # now this means "samples in this shard"
    shard_start = time.time()


    print("starting to collect whole shard")

    # Collect the whole shard (closest to your original structure)
    for sample in dataset:
        batch_samples.append(sample)
    print(f"📦 Shard contains {len(batch_samples)} raw samples")


    # -------------------------
    # Build dict for this shard
    # -------------------------
    id_to_sample = {}
    for s in batch_samples:
        try:
            image_file_id = extract_image_file_id(s)
            id_to_sample[image_file_id] = s
        except Exception:
            continue
    print("done batching samples")

    # -------------------------
    # One metadata query for this shard
    # -------------------------
    meta_dict = fetch_metadata_map(cur, id_to_sample.keys())
    print("done fetching metadata dict")
    # -------------------------
    # Process the shard (same as your original)
    # -------------------------
    print("starting to process shard"
    for image_file_id, s in id_to_sample.items():
        key = s.get("__key__", "?")
        try:
            meta_row = meta_dict.get(image_file_id)
            if not meta_row:
                continue

            path = make_image_url_from_row(meta_row)

            # IMPORTANT: use s, not sample
            jpg_bytes = s["jpg"]

            counts, bboxes = yolo_counts_and_bboxes_from_bytes(jpg_bytes)

            upsert_image_into_yfcc_index(cur, image_file_id, path, counts)

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

            processed += 1
            if processed % 100 == 0:
                print(f"done processing {processed} images")
            

        except Exception as e:
            print(f"⚠️ Error processing {key}: {e}")

    # -------------------------
    # Commit ONCE per shard
    # -------------------------
    bulk_insert_bboxes(cur, bbox_buffer)
    conn.commit()

    shard_elapsed = time.time() - shard_start
    total_elapsed = time.time() - start_time
    imgs_per_sec = processed / total_elapsed if total_elapsed > 0 else 0

    print(
    f"✅ Shard {shard:06d} committed | "
    f"raw={len(batch_samples)} | "
    f"processed={shard_processed} | "
    f"shard time {format_seconds(shard_elapsed)} | "
    f"total {processed} images | "
    f"{imgs_per_sec:.2f} img/s"
)

elapsed = time.time() - start_time
print(f"✅ Finished! Processed {processed} images in {format_seconds(elapsed)}")