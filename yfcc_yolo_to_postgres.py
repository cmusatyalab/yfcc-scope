import os, glob
from PIL import Image
from IPython.display import display
import torch, torchvision as tv

BASE_URL = os.environ.get("BASE_URL", "http://localhost/images")
IMG_DIR = os.environ.get("IMG_DIR", "/path/to/yfcc/images")
imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

from ultralytics import YOLO
import re
from PIL import Image
from IPython.display import display
import psycopg2, psycopg2.extras
import json, os, hashlib

yolo = YOLO("yolo11s.pt") 

def space_to_underscore(label: str) -> str:
    return label.lower().replace(" ", "_")
    
def yolo_names():
    names = yolo.model.names
    return [space_to_underscore(names[i]) for i in range(len(names))]

YOLO_LABELS = yolo_names()
# print(YOLO_LABELS)

def image_id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def path_to_link(p: str) -> str:
    rel = os.path.relpath(p, IMG_DIR)  # relative to dataset root
    return f"{BASE_URL}/{rel.replace(os.sep, '/')}"  # URL-style slashes

def flickr_link(p: str) -> str:
    photo_id = image_id_from_path(p)
    return f"https://www.flickr.com/photo.gne?id={photo_id}"

def open_conn():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "yfcc"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", 5432)),
    )

def make_upsert_sql():
    base = ['image_id', 'path', 'total_bboxes']
    all_cols = base + YOLO_LABELS
    insert_cols = ", ".join(f'"{c}"' for c in all_cols)
    placeholders = ", ".join(["%s"] * len(all_cols))

    set_parts = [f'"{c}"=EXCLUDED."{c}"' for c in base[1:]]  # path,total_bboxes
    set_parts += [f'"{c}"=EXCLUDED."{c}"' for c in YOLO_LABELS]
    set_parts.append('ts=now()')

    return f"""
    INSERT INTO yfcc_index ({insert_cols})
    VALUES ({placeholders})
    ON CONFLICT (image_id) DO UPDATE
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
            image_id_from_path(path),
            path,
            total_bboxes,
            *row_vals,
        ),
    )


def upsert_all_bboxes_for_image_into_bb_table(cur, image_id, bboxes):
        
    upsert_sql = """
    INSERT INTO bb_table (
        image_id, bounding_box_number, label, confidence_score,
        center_x, center_y, width, height
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    for i, box in enumerate(bboxes, start=1):
        center_x, center_y, width, height = box["bbox_xywh"]
        cur.execute(
            upsert_sql,
            (
                image_id,
                i,                          # bounding_box_number
                box["label"],
                box["score"],
                center_x,
                center_y,
                width,
                height,
            ),
        )
def yolo_counts_and_bboxes(path):
    # Set conf=0 so we keep every detection, no pre-filtering
    res = yolo.predict(path, conf=0.4, verbose=False)[0]

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


conn = open_conn()
cur = conn.cursor()
# print("trying to truncate bbtable")
# cur.execute("TRUNCATE TABLE bb_table;")
# print("trying to truncate yfcc_index")
# cur.execute("TRUNCATE TABLE yfcc_index CASCADE;")
# print("done truncating")
cur.execute("""
CREATE TABLE IF NOT EXISTS yfcc_index (
  image_id   TEXT PRIMARY KEY,
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
  image_id TEXT REFERENCES yfcc_index(image_id) ON DELETE CASCADE,
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


for path in imgs[:7000]:
    counts, bboxes = yolo_counts_and_bboxes(path)
    # display(Image.open(path))
    # print(path, counts)
    # if (len(counts) == 0):
        # print("problem!! " + path)
    
    upsert_image_into_yfcc_index(cur, path, counts)
    upsert_all_bboxes_for_image_into_bb_table(cur, image_id_from_path(path), bboxes)
    print(".")
    
conn.commit()
cur.close(); conn.close()

print("✅ Upserted rows into yfcc_index")