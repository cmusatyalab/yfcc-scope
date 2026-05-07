# Starlette app library code for YFCC viewer (image with boxes) and API.
# Runs on port 8081 by default (yfcc-viewer apps use this port to connect to the API)

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from PIL import Image, ImageDraw
from io import BytesIO
from string import Template

import psycopg2
import hashlib
import colorsys
import html
import os
import time
import json
import logging
import re
import requests


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("yfcc")


# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
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
    cleaned = [
        (n, label, cx, cy, w, h, conf)
        for _, n, label, cx, cy, w, h, conf in rows
        if label is not None
    ]
    return (path, cleaned)


# -----------------------------------------------------------------------------
# Labels + Colors
# -----------------------------------------------------------------------------
LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic_light","fire_hydrant","stop_sign","parking_meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports_ball",
    "kite","baseball_bat","baseball_glove","skateboard","surfboard","tennis_racket",
    "bottle","wine_glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot_dog","pizza","donut","cake","chair",
    "couch","potted_plant","bed","dining_table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell_phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy_bear","hair_drier","toothbrush"
]


def color_for_label(label: str):
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 360
    s, v = 0.75, 1.0
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# -----------------------------------------------------------------------------
# Confidence binning
# -----------------------------------------------------------------------------
def conf_to_bin(c: float) -> int:
    c = max(0.0, min(1.0, round(float(c), 2)))
    return int(round(c * 100))


# -----------------------------------------------------------------------------
# Histogram helpers
# -----------------------------------------------------------------------------
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
        cur.execute("""
            INSERT INTO yfcc_label_conf_hist (label, conf_bin, box_count, updated_at)
            SELECT
                b.label,
                GREATEST(0, LEAST(100, FLOOR(b.confidence_score * 100.0 + 0.000001)))::SMALLINT AS conf_bin,
                COUNT(*) AS box_count,
                %s AS updated_at
            FROM bb_table b
            WHERE b.confidence_score IS NOT NULL
            GROUP BY b.label, conf_bin;
        """, (now,))
        cur.execute("""
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
        """, (now,))
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
        cur.execute("""
            SELECT label, COALESCE(SUM(box_count),0)
            FROM yfcc_label_conf_hist
            WHERE conf_bin >= %s
            GROUP BY label;
        """, (tb,))
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
        cur.execute("""
            SELECT COALESCE(SUM(image_count),0)
            FROM yfcc_images_maxbin_hist
            WHERE max_bin >= %s;
        """, (tb,))
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


# -----------------------------------------------------------------------------
# Vector table
# -----------------------------------------------------------------------------
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
            cur.execute("""
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
            """, (now,))
            cur.execute("COMMIT;")
    except Exception as e:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK;")
        conn.close()
        log.exception("Vector table rebuild failed: %s", e)
        raise
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# HTML template
# -----------------------------------------------------------------------------
HTML_TEMPLATE = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>YFCC Boxes Viewer</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; }
    form { display: grid; gap: 12px; }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 6px 12px; max-height: 320px; overflow: auto; padding: 8px; border: 1px solid #ddd; border-radius: 8px; }
    .lbl { white-space: nowrap; }
    .img-wrap { margin-top: 10px; }
    button { padding: 8px 12px; border-radius: 8px; border: 1px solid #ccc; cursor: pointer; }
    input[type="text"] { padding: 6px 8px; border-radius: 8px; border: 1px solid #ccc; min-width: 220px; }
    .muted { color: #666; font-size: 13px; }
    .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  </style>
</head>
<body>
  <form id="ctrl_form" method="GET" action="/">
    <div class="row toolbar">
      <input name="image_file_id" value="$qsafe" placeholder="1234567890" />
      <button type="submit">Show</button>
      <span class="muted">Tip: turn "Select all" off to pick specific labels.</span>
    </div>

    <div class="row toolbar" id="top_toolbar">
      <label style="font-weight:600;">
        <input id="select_all" type="checkbox" $all_checked_attr>
        Select all
      </label>
      <input type="hidden" id="select_all_hidden" name="select_all" value="$select_all_val">

      <button type="button" id="sort_original">Original order</button>
      <button type="button" id="sort_alpha">A–Z</button>
      <button type="button" id="sort_freq">By frequency</button>

      <div class="row" style="gap:8px; align-items:center; margin-left:8px;">
        <label for="min_conf" class="muted" style="font-weight:600;">Min confidence</label>
        <input type="range" id="min_conf" name="min_conf" min="0.4" max="1" step="0.01" value="$min_conf">
        <span id="min_conf_val" class="muted">$min_conf_fmt</span>
      </div>
    </div>

    <div class="grid" id="labels_grid">
      $checkboxes_html
    </div>

    <div class="muted" id="freq_updated_note" style="margin-top:6px;"></div>
  </form>

  <div class="img-wrap">
    $img_tag
  </div>

<script>
  const form = document.getElementById('ctrl_form');
  const selectAll = document.getElementById('select_all');
  const grid = document.getElementById('labels_grid');
  const hiddenAll = document.getElementById('select_all_hidden');
  const minConf = document.getElementById('min_conf');
  const minConfVal = document.getElementById('min_conf_val');

  function updateSelectAllState() {
    const boxes = Array.from(grid.querySelectorAll('input[type="checkbox"][name="label"]'));
    const checked = boxes.filter(b => b.checked).length;
    const isAll = (checked === boxes.length);
    selectAll.checked = isAll;
    hiddenAll.value = isAll ? '1' : '0';
  }

  form.addEventListener('submit', (e) => {
    const boxes = grid.querySelectorAll('input[type="checkbox"][name="label"]');
    if (selectAll.checked) {
      boxes.forEach(b => b.disabled = true);
      setTimeout(() => boxes.forEach(b => b.disabled = false), 0);
    }
  });

  selectAll.addEventListener('change', () => {
    const boxes = grid.querySelectorAll('input[type="checkbox"][name="label"]');
    if (selectAll.checked) {
      boxes.forEach(b => b.checked = true);
      hiddenAll.value = '1';
    } else {
      boxes.forEach(b => b.checked = false);
      hiddenAll.value = '0';
    }
    form.requestSubmit();
  });

  grid.addEventListener('change', (e) => {
    if (e.target && e.target.name === 'label') {
      updateSelectAllState();
      form.requestSubmit();
    }
  });

  function updateMinConfVal(){
    if (minConfVal) minConfVal.textContent = Number(minConf.value).toFixed(2);
  }
  minConf?.addEventListener('input', updateMinConfVal);
  minConf?.addEventListener('change', () => { form.requestSubmit(); });
  updateMinConfVal();

  updateSelectAllState();
</script>

<script src="/freqs.js"></script>
</body>
</html>
""")


def render_checkboxes(selected: set, all_checked: bool):
    items = []
    for lab in LABELS:
        checked = "checked" if (all_checked or lab in selected) else ""
        items.append(f'<label class="lbl"><input type="checkbox" name="label" value="{lab}" {checked}> {lab}</label>')
    return "\n".join(items)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
async def home(request: Request):
    qp = request.query_params
    image_file_id = qp.get("image_file_id", "")
    selected = set(qp.getlist("label"))
    all_checked = (qp.get("select_all", "1") == "1")

    try:
        min_conf = float(qp.get("min_conf", "0.4"))
    except ValueError:
        min_conf = 0.4
    min_conf = max(0.0, min(1.0, round(min_conf, 2)))

    img_params = [
        f"image_file_id={image_file_id}",
        f"select_all={'1' if all_checked else '0'}",
        f"min_conf={min_conf:.2f}",
    ]
    if not all_checked:
        img_params += [f"label={lab}" for lab in selected]
    img_src = "/image?" + "&".join(img_params) if image_file_id else ""
    img_tag = f"<img style='max-width:95%; border:1px solid #eee;' src='{img_src}'/>" if img_src else ""

    html_str = HTML_TEMPLATE.safe_substitute(
        qsafe=html.escape(image_file_id),
        all_checked_attr=("checked" if all_checked else ""),
        select_all_val=("1" if all_checked else "0"),
        min_conf=f"{min_conf:.2f}",
        min_conf_fmt=f"{min_conf:.2f}",
        checkboxes_html=render_checkboxes(selected, all_checked),
        img_tag=img_tag,
    )
    return HTMLResponse(html_str)


async def image(request: Request):
    qp = request.query_params
    image_file_id = qp.get("image_file_id", "")
    all_checked = (qp.get("select_all", "1") == "1")
    selected = set(qp.getlist("label"))

    try:
        min_conf = float(qp.get("min_conf", "0.4"))
    except ValueError:
        min_conf = 0.4
    min_conf = max(0.0, min(1.0, round(min_conf, 2)))

    path, rows = fetch(image_file_id)
    if not path:
        msg = f"not found (image_file_id={image_file_id})"
        log.warning(msg)
        return HTMLResponse(msg, status_code=404)

    try:
        resp = requests.get(path, timeout=10, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        msg = f"failed to fetch/open image: {path} err={e}"
        log.error(msg)
        return HTMLResponse(msg, status_code=502)

    W, H = img.size
    d = ImageDraw.Draw(img)

    for n, label, cx, cy, w, h, conf in rows:
        if (conf is None) or (conf < min_conf):
            continue
        if not all_checked and (not selected or label not in selected):
            continue
        x0 = (cx - w / 2) * W
        y0 = (cy - h / 2) * H
        x1 = (cx + w / 2) * W
        y1 = (cy + h / 2) * H
        color = color_for_label(label)
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = f"{n}: {label} ({conf:.2f})"
        d.text((x0 + 2, y0 + 1), tag, fill=(255, 255, 255))

    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# -----------------------------------------------------------------------------
# NEW: Run arbitrary SQL query, return rows in vector_rows format
# -----------------------------------------------------------------------------
BLOCKED_KEYWORDS = [
    "DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE", "ALTER",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
]


def validate_sql(raw_sql: str):
    sql = (raw_sql or "").strip()
    if not sql:
        raise ValueError("No SQL provided")

    sql = sql.rstrip().rstrip(";").rstrip()
    first_word = sql.lstrip().split()[0].upper() if sql else ""
    if first_word != "SELECT":
        raise ValueError("Only SELECT queries are permitted")

    upper_sql = sql.upper()

    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper_sql):
            raise ValueError(f"Query contains forbidden keyword: {kw}")

    # Keep user SQL simple since the backend already wraps it in a CTE.
    if re.search(r"\bWITH\b", upper_sql):
        raise ValueError(
            "Queries using WITH/CTEs are not allowed. "
            "Return a single SELECT statement."
        )

    # Block expensive correlated-subquery ranking patterns
    if re.search(
        r"ORDER\s+BY\s*\(\s*SELECT\s+(AVG|MAX|MIN|SUM|COUNT)\s*\(",
        upper_sql,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            "Correlated subqueries in ORDER BY are not allowed. "
            "Use JOIN + GROUP BY + ORDER BY MAX(...) or AVG(...)."
        )

    if re.search(
        r"SELECT\s+(AVG|MAX|MIN|SUM|COUNT)\s*\([^)]*\)\s+FROM\s+BB_TABLE\s+WHERE\s+BB_TABLE\.IMAGE_FILE_ID\s*=",
        upper_sql,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            "Correlated subqueries against bb_table are not allowed. "
            "Use JOIN + GROUP BY instead."
        )

    return sql


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
            log.info("about to set statement timeout")
            cur.execute("SET statement_timeout TO 30000;")
            log.info("about to execute wrapped query")
            cur.execute(wrapped_sql)
            log.info("wrapped query executed")
            rows = cur.fetchall()
            log.info("fetched %d rows", len(rows))
            return rows
    finally:
        conn.close()


async def run_query(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        raw_sql = validate_sql(body.get("sql"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info("run_query received SQL: %s", raw_sql)

    try:
        rows = await run_in_threadpool(execute_wrapped_query, raw_sql)
    except Exception as e:
        log.error("run_query failed: %s\nSQL: %s", e, raw_sql)
        return JSONResponse({"error": f"Query failed: {e}"}, status_code=400)

    out_rows = []
    for image_file_id, path, total_bboxes, counts_json in rows:
        counts = counts_json or {}
        item = {
            "image_file_id": image_file_id,
            "path": path,
            "thumb_url": path,
            "total_bboxes": int(total_bboxes or 0),
        }
        for lab in LABELS:
            item[lab] = int(counts.get(lab, 0) or 0)
        out_rows.append(item)

    log.info("run_query returned %d rows", len(out_rows))
    return JSONResponse({"rows": out_rows, "count": len(out_rows)})


# -----------------------------------------------------------------------------
# Existing API routes
# -----------------------------------------------------------------------------
FREQS_CLIENT_JS = r"""
(function(){
  async function getJSON(u){ const r = await fetch(u, {cache:'no-store'}); if(!r.ok) throw new Error('fetch '+u); return r.json(); }
  async function post(u){ const r = await fetch(u, {method:'POST'}); if(!r.ok) throw new Error('post '+u); return r.text(); }

  let freqMap = {};
  let updatedAt = 0;
  let totalImagesYFCC = 0;
  let imagesWithBoxes = 0;
  let originalOrder = null;

  function labelKey(lbl){
    const input = lbl.querySelector('input[name="label"]');
    return input ? input.value : lbl.textContent.trim();
  }

  function decorateFractions(grid){
    grid.querySelectorAll('label.lbl').forEach(lbl => {
      const key = labelKey(lbl);
      const ent = freqMap[key] || {fraction:0};
      const pretty = `(${ent.fraction.toFixed(4)})`;
      let span = lbl.querySelector('span.__freq');
      if (!span){
        span = document.createElement('span');
        span.className = 'muted __freq';
        span.style.marginLeft = '4px';
        lbl.appendChild(span);
      }
      span.textContent = pretty;
    });
  }

  function updateFooter(){
    const note = document.getElementById('freq_updated_note');
    if (!note) return;
    if (updatedAt){
      const ts = new Date(updatedAt * 1000);
      note.textContent = 'Frequencies last updated: ' + ts.toLocaleString() +
                         ' — Total images in index: ' + totalImagesYFCC.toLocaleString() +
                         ' — Images with boxes (at threshold): ' + imagesWithBoxes.toLocaleString();
    } else {
      note.textContent = '';
    }
  }

  function sortGridBy(grid, cmp){
    const items = Array.from(grid.querySelectorAll('label.lbl'));
    items.sort(cmp);
    items.forEach(el => grid.appendChild(el));
  }

  async function main(){
    const grid = document.getElementById('labels_grid');
    const selectAllRow = document.getElementById('top_toolbar');
    const minConf = document.getElementById('min_conf');
    if(!grid || !minConf) return;
    if(!originalOrder) originalOrder = Array.from(grid.querySelectorAll('label.lbl'));

    const btnOrig = document.getElementById('sort_original');
    const btnAlpha = document.getElementById('sort_alpha');
    const btnFreq = document.getElementById('sort_freq');

    btnOrig?.addEventListener('click', ()=> originalOrder.forEach(el => grid.appendChild(el)));
    btnAlpha?.addEventListener('click', ()=> sortGridBy(grid,(a,b)=>labelKey(a).localeCompare(labelKey(b))));
    btnFreq?.addEventListener('click', ()=> sortGridBy(grid,(a,b)=>{
      const fa=freqMap[labelKey(a)]?.fraction||0, fb=freqMap[labelKey(b)]?.fraction||0;
      if(fb!==fa) return fb-fa; return labelKey(a).localeCompare(labelKey(b));
    }));

    if (selectAllRow && !document.getElementById('recalc_btn')) {
      const btn = document.createElement('button');
      btn.type = 'button'; btn.id='recalc_btn'; btn.textContent='Recalculate frequencies';
      btn.style.marginLeft='12px';
      btn.addEventListener('click', async()=>{
        btn.disabled=true; btn.textContent='Recalculating...';
        try{
          await post('/recalc');
          location.reload();
        }
        catch(e){
          alert('Recalc failed: '+e.message); btn.disabled=false; btn.textContent='Recalculate frequencies';
        }
      });
      selectAllRow.appendChild(btn);
    }

    let data;
    try{
      const c = Number(minConf.value).toFixed(2);
      data = await getJSON('/freqs?min_conf='+encodeURIComponent(c));
    }catch(e){ return; }
    freqMap=data.labels||{};
    updatedAt=data.updated_at||0;
    totalImagesYFCC=data.total_images_yfcc||0;
    imagesWithBoxes=data.images_with_boxes||0;

    decorateFractions(grid);
    updateFooter();
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',main); else main();
})();
"""


async def freqs_client_js(request: Request):
    return HTMLResponse(FREQS_CLIENT_JS, media_type="application/javascript")


async def freqs_api(request: Request):
    qp = request.query_params
    try:
        min_conf = float(qp.get("min_conf", "0.4"))
    except ValueError:
        min_conf = 0.4
    min_conf = max(0.0, min(1.0, round(min_conf, 2)))

    counts, total_boxes, updated_at = read_label_counts_at_threshold(min_conf)
    images_with_boxes = read_images_with_boxes_at_threshold(min_conf)
    total_images_yfcc = read_total_images_yfcc()

    labels_payload = {}
    for lab in LABELS:
        cnt = counts.get(lab, 0)
        frac = (float(cnt) / float(total_boxes)) if total_boxes else 0.0
        labels_payload[lab] = {"fraction": frac}

    return JSONResponse({
        "updated_at": updated_at,
        "total_images_yfcc": total_images_yfcc,
        "images_with_boxes": images_with_boxes,
        "min_conf": f"{min_conf:.2f}",
        "labels": labels_payload,
    })


async def recalc_freqs(request: Request):
    try:
        await run_in_threadpool(rebuild_histograms)
    except Exception as e:
        return PlainTextResponse(f"recalc failed: {e}", status_code=500)
    return PlainTextResponse("ok (hist rebuilt)")


MAX_LIMIT = 500


def parse_conf_ranges_0_100(conf_str: str):
    if not conf_str:
        return None
    ranges = []
    for part in conf_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo_bin = int(lo_s)
            hi_bin = int(hi_s)
        else:
            lo_bin = hi_bin = int(part)
        lo_bin = max(0, min(100, lo_bin))
        hi_bin = max(0, min(100, hi_bin))
        if lo_bin > hi_bin:
            lo_bin, hi_bin = hi_bin, lo_bin
        lo = lo_bin / 100.0
        hi = min((hi_bin + 1) / 100.0, 1.0)
        ranges.append((lo, hi))
    return ranges if ranges else None


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


async def images_api(request: Request):
    qp = request.query_params
    labels = [l.strip() for l in qp.getlist("label") if l.strip()]
    try:
        limit = int(qp.get("limit", "50"))
    except ValueError:
        return JSONResponse({"error": "limit must be an int"}, status_code=400)
    try:
        offset = int(qp.get("offset", "0"))
    except ValueError:
        return JSONResponse({"error": "offset must be an int"}, status_code=400)

    limit = max(1, min(MAX_LIMIT, limit))
    offset = max(0, offset)
    conf_ranges = parse_conf_ranges_0_100((qp.get("conf") or "").strip())

    try:
        images = await run_in_threadpool(fetch_images_for_labels, labels, limit, offset, conf_ranges)
    except Exception as e:
        return JSONResponse({"error": f"images query failed: {e}"}, status_code=500)

    return JSONResponse({"labels": labels, "limit": limit, "offset": offset, "images": images})


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


async def conf_hist(request: Request):
    labels = request.query_params.getlist("label")
    if not labels:
        return JSONResponse({"error": "need ?label=cat&label=dog"}, status_code=400)
    try:
        bins = await run_in_threadpool(conf_hist_sync, labels)
    except Exception as e:
        return JSONResponse({"error": f"conf_hist failed: {e}"}, status_code=500)
    return JSONResponse({"labels": labels, "bins": bins})


def vector_rows_sync(limit, offset, min_total):
    conn = open_conn()
    try:
        with conn.cursor() as cur:
            ensure_vector_table(conn)
            cur.execute("""
                SELECT v.image_file_id, y.path, v.total_bboxes, v.counts
                FROM yfcc_image_label_counts v
                JOIN yfcc_index y ON y.image_file_id = v.image_file_id
                WHERE v.total_bboxes >= %s
                ORDER BY y.ts DESC
                LIMIT %s OFFSET %s;
            """, (min_total, limit, offset))
            rows = cur.fetchall()
            cur.execute("SELECT COALESCE(MAX(updated_at),0) FROM yfcc_image_label_counts;")
            updated_at = cur.fetchone()[0] or 0
        return rows, updated_at
    finally:
        conn.close()


async def vector_rows_api(request: Request):
    qp = request.query_params
    try:
        limit = int(qp.get("limit", "5000"))
    except ValueError:
        return JSONResponse({"error": "limit must be an int"}, status_code=400)
    try:
        offset = int(qp.get("offset", "0"))
    except ValueError:
        return JSONResponse({"error": "offset must be an int"}, status_code=400)

    limit = max(1, min(20000, limit))
    offset = max(0, offset)

    try:
        min_total = int(qp.get("min_total_bboxes", "1"))
    except ValueError:
        min_total = 1
    min_total = max(0, min(10_000, min_total))

    try:
        rows, updated_at = await run_in_threadpool(vector_rows_sync, limit, offset, min_total)
    except Exception as e:
        return JSONResponse({"error": f"vector_rows failed: {e}"}, status_code=500)

    out_rows = []
    for image_file_id, path, total_bboxes, counts_json in rows:
        counts = counts_json or {}
        item = {
            "image_file_id": image_file_id,
            "path": path,
            "thumb_url": path,
            "total_bboxes": int(total_bboxes or 0),
        }
        for lab in LABELS:
            item[lab] = int(counts.get(lab, 0) or 0)
        out_rows.append(item)

    return JSONResponse({"updated_at": updated_at, "limit": limit, "offset": offset, "rows": out_rows})


async def recalc_vectors(request: Request):
    try:
        await run_in_threadpool(rebuild_vector_table)
    except Exception as e:
        return PlainTextResponse(f"recalc vectors failed: {e}", status_code=500)
    return PlainTextResponse("ok (vectors rebuilt)")



# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Starlette(routes=[
    Route("/", home),
    Route("/image", image),
    Route("/api/images", images_api),
    Route("/api/conf_hist", conf_hist, methods=["GET"]),
    Route("/api/vector_rows", vector_rows_api),
    Route("/api/run_query", run_query, methods=["POST"]),   # NEW
    Route("/freqs", freqs_api),
    Route("/recalc", recalc_freqs, methods=["POST"]),
    Route("/recalc_vectors", recalc_vectors, methods=["POST"]),
    Route("/freqs.js", freqs_client_js),
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://128.2.212.50:5174",
        "http://128.2.212.50:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
