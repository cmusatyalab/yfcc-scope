# starlette_app.py
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.requests import Request
from PIL import Image, ImageDraw
from io import BytesIO
import psycopg2, hashlib, colorsys, html
import os, time, json, logging, math
from string import Template

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

def fetch(image_id: str):
    """
    Return (path, rows) where rows are:
      (bounding_box_number, label, center_x, center_y, width, height, confidence_score)
    """
    try:
        conn = open_conn(); cur = conn.cursor()
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
              ON y.image_id = b.image_id
            WHERE y.image_id = %s
            ORDER BY COALESCE(b.bounding_box_number, 0)
            """,
            (image_id,)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception as e:
        log.error("DB fetch failed for image_id=%r: %s", image_id, e)
        return (None, [])

    if not rows:
        log.warning("image_id %r not found in yfcc_index", image_id)
        return (None, [])

    path = rows[0][0]
    cleaned = [ (n, label, cx, cy, w, h, conf)
               for _, n, label, cx, cy, w, h, conf in rows
               if label is not None ]
    return (path, cleaned)

# -----------------------------------------------------------------------------
# Labels + Colors
# -----------------------------------------------------------------------------
LABELS = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
          'traffic_light','fire_hydrant','stop_sign','parking_meter','bench','bird','cat',
          'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
          'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports_ball',
          'kite','baseball_bat','baseball_glove','skateboard','surfboard','tennis_racket',
          'bottle','wine_glass','cup','fork','knife','spoon','bowl','banana','apple',
          'sandwich','orange','broccoli','carrot','hot_dog','pizza','donut','cake','chair',
          'couch','potted_plant','bed','dining_table','toilet','tv','laptop','mouse','remote',
          'keyboard','cell_phone','microwave','oven','toaster','sink','refrigerator','book',
          'clock','vase','scissors','teddy_bear','hair_drier','toothbrush']

def color_for_label(label: str):
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 360
    s, v = 0.75, 1.0
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return (int(r*255), int(g*255), int(b*255))

# -----------------------------------------------------------------------------
# Confidence binning (0..100 for 0.00..1.00)
# -----------------------------------------------------------------------------
def conf_to_bin(c: float) -> int:
    # Matches UI: min_conf rounded to 2 decimals then multiplied by 100
    c = max(0.0, min(1.0, round(float(c), 2)))
    return int(round(c * 100))

# -----------------------------------------------------------------------------
# Histogram helpers (DB-backed; no CSV)
# -----------------------------------------------------------------------------
# Schema:
#   yfcc_label_conf_hist(label TEXT, conf_bin SMALLINT, box_count BIGINT, updated_at INT, PRIMARY KEY(label, conf_bin))
#   yfcc_images_maxbin_hist(max_bin SMALLINT PRIMARY KEY, image_count BIGINT, updated_at INT)
#
# Build-time:
#   - yfcc_label_conf_hist: per (label, conf_bin) counts, where conf_bin=floor(conf*100)
#   - yfcc_images_maxbin_hist: per max_bin counts of images whose max confidence bin equals that bin

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
    """
    Fully rebuild yfcc_label_conf_hist and yfcc_images_maxbin_hist from bb_table.
    """
    conn = open_conn()
    ensure_hist_tables(conn)
    cur = conn.cursor()
    now = int(time.time())

    try:
        cur.execute("BEGIN;")

        # Clear existing hist
        cur.execute("TRUNCATE yfcc_label_conf_hist;")
        cur.execute("TRUNCATE yfcc_images_maxbin_hist;")

        # Fill label/bin histogram
        # conf_bin = floor(conf * 100) clamped to [0,100]
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

        # Compute per-image max bin, then aggregate how many images have each max_bin
        cur.execute("""
            WITH maxbin AS (
                SELECT
                    image_id,
                    GREATEST(0, LEAST(100, FLOOR(MAX(confidence_score) * 100.0 + 0.000001)))::SMALLINT AS max_bin
                FROM bb_table
                WHERE confidence_score IS NOT NULL
                GROUP BY image_id
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
    """
    Returns dict[label] = count of boxes with conf >= min_conf (by summing histogram bins).
    Also returns total_boxes and last_updated.
    """
    tb = conf_to_bin(min_conf)
    conn = open_conn(); cur = conn.cursor()
    try:
        # Sum counts for bins >= threshold
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
        cur.close(); conn.close()

    counts = {lab: 0 for lab in LABELS}
    for lab, cnt in label_rows:
        if lab in counts:
            counts[lab] = int(cnt)
    total_boxes = sum(counts.values())
    return counts, total_boxes, updated_at

def read_images_with_boxes_at_threshold(min_conf: float) -> int:
    """
    Returns number of images that have at least one box with conf >= min_conf,
    computed from the images_maxbin histogram (cumulative from threshold up).
    """
    tb = conf_to_bin(min_conf)
    conn = open_conn(); cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COALESCE(SUM(image_count),0)
            FROM yfcc_images_maxbin_hist
            WHERE max_bin >= %s;
        """, (tb,))
        n = cur.fetchone()[0] or 0
    finally:
        cur.close(); conn.close()
    return int(n)

def read_total_images_yfcc() -> int:
    conn = open_conn(); cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM yfcc_index;")
        n = cur.fetchone()[0] or 0
    finally:
        cur.close(); conn.close()
    return int(n)

# -----------------------------------------------------------------------------
# HTML (use Template to avoid f-string brace collisions)
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
      <input name="image_id" value="$qsafe" placeholder="1234567890" />
      <button type="submit">Show</button>
      <span class="muted">Tip: turn “Select all” off to pick specific labels.</span>
    </div>

    <div class="row toolbar" id="top_toolbar">
      <label style="font-weight:600;">
        <input id="select_all" type="checkbox" $all_checked_attr>
        Select all
      </label>
      <input type="hidden" id="select_all_hidden" name="select_all" value="$select_all_val">

      <!-- Sort controls -->
      <button type="button" id="sort_original">Original order</button>
      <button type="button" id="sort_alpha">A–Z</button>
      <button type="button" id="sort_freq">By frequency</button>

      <!-- Min confidence slider -->
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

# -----------------------------------------------------------------------------
# HTML helpers
# -----------------------------------------------------------------------------
def render_checkboxes(selected: set[str], all_checked: bool):
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
    image_id = qp.get("image_id", "")
    selected = set(qp.getlist("label"))
    all_checked = (qp.get("select_all", "1") == "1")

    try:
        min_conf = float(qp.get("min_conf", "0.4"))
    except ValueError:
        min_conf = 0.4
    min_conf = max(0.0, min(1.0, round(min_conf, 2)))

    img_params = [f"image_id={image_id}",
                  f"select_all={'1' if all_checked else '0'}",
                  f"min_conf={min_conf:.2f}"]
    if not all_checked:
        img_params += [f"label={lab}" for lab in selected]
    img_src = "/image?" + "&".join(img_params) if image_id else ""
    img_tag = f"<img style='max-width:95%; border:1px solid #eee;' src='{img_src}'/>" if img_src else ""

    html_str = HTML_TEMPLATE.safe_substitute(
        qsafe=html.escape(image_id),
        all_checked_attr=("checked" if all_checked else ""),
        select_all_val=('1' if all_checked else '0'),
        min_conf=f"{min_conf:.2f}",
        min_conf_fmt=f"{min_conf:.2f}",
        checkboxes_html=render_checkboxes(selected, all_checked),
        img_tag=img_tag
    )
    return HTMLResponse(html_str)

async def image(request: Request):
    qp = request.query_params
    image_id = qp.get("image_id", "")
    all_checked = (qp.get("select_all", "1") == "1")
    selected = set(qp.getlist("label"))

    try:
        min_conf = float(qp.get("min_conf", "0.4"))
    except ValueError:
        min_conf = 0.4
    min_conf = max(0.0, min(1.0, round(min_conf, 2)))

    path, rows = fetch(image_id)
    if not path:
        msg = f"not found (image_id={image_id})"
        log.warning(msg)
        return HTMLResponse(msg, status_code=404)

    if not os.path.exists(path):
        msg = f"image file missing on disk: {path} (image_id={image_id})"
        log.error(msg)
        return HTMLResponse(msg, status_code=404)

    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        msg = f"failed to open image: {path} (image_id={image_id}) err={e}"
        log.error(msg)
        return HTMLResponse(msg, status_code=500)

    W, H = img.size
    d = ImageDraw.Draw(img)

    drawn = 0
    for n, label, cx, cy, w, h, conf in rows:
        if (conf is None) or (conf < min_conf):
            continue
        if not all_checked:
            if not selected or (label not in selected):
                continue

        x0 = (cx - w/2) * W; y0 = (cy - h/2) * H
        x1 = (cx + w/2) * W; y1 = (cy + h/2) * H
        color = color_for_label(label)
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = f"{n}: {label} ({conf:.2f})"
        d.text((x0+2, y0+1), tag, fill=(255,255,255))
        drawn += 1

    if drawn == 0:
        log.info("No boxes drawn for image_id=%r at min_conf=%.2f (labels filter size=%d, all=%s)",
                 image_id, min_conf, len(selected), all_checked)

    buf = BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# -----------------------------------------------------------------------------
# Client JS (uses histogram-backed /freqs; Recalc rebuilds hist)
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

    // Buttons
    const btnOrig = document.getElementById('sort_original');
    const btnAlpha = document.getElementById('sort_alpha');
    const btnFreq = document.getElementById('sort_freq');

    btnOrig?.addEventListener('click', ()=> originalOrder.forEach(el => grid.appendChild(el)));
    btnAlpha?.addEventListener('click', ()=> sortGridBy(grid,(a,b)=>labelKey(a).localeCompare(labelKey(b))));
    btnFreq?.addEventListener('click', ()=> sortGridBy(grid,(a,b)=>{
      const fa=freqMap[labelKey(a)]?.fraction||0, fb=freqMap[labelKey(b)]?.fraction||0;
      if(fb!==fa) return fb-fa; return labelKey(a).localeCompare(labelKey(b));
    }));

    // Recalc button: rebuilds DB histograms; then reload
    if (selectAllRow && !document.getElementById('recalc_btn')) {
      const btn = document.createElement('button');
      btn.type = 'button'; btn.id='recalc_btn'; btn.textContent='Recalculate frequencies';
      btn.style.marginLeft='12px';
      btn.addEventListener('click', async()=>{
        btn.disabled=true; btn.textContent='Recalculating...';
        try{
          await post('/recalc'); // full rebuild; threshold-independent
          location.reload();
        }
        catch(e){
          alert('Recalc failed: '+e.message); btn.disabled=false; btn.textContent='Recalculate frequencies';
        }
      });
      selectAllRow.appendChild(btn);
    }

    // Load freqs at current threshold
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

# -----------------------------------------------------------------------------
# API Routes (histogram-backed)
# -----------------------------------------------------------------------------
async def freqs_api(request: Request):
    qp = request.query_params
    try:
        min_conf = float(qp.get("min_conf", "0.4"))
    except ValueError:
        min_conf = 0.4
    min_conf = max(0.0, min(1.0, round(min_conf, 2)))

    # From histogram tables:
    counts, total_boxes, updated_at = read_label_counts_at_threshold(min_conf)
    images_with_boxes = read_images_with_boxes_at_threshold(min_conf)
    total_images_yfcc = read_total_images_yfcc()

    # Fractions: per label count / total_boxes (guard 0)
    labels_payload = {}
    for lab in LABELS:
        cnt = counts.get(lab, 0)
        frac = (float(cnt) / float(total_boxes)) if total_boxes else 0.0
        labels_payload[lab] = {"fraction": frac}

    payload = {
        "updated_at": updated_at,
        "total_images_yfcc": total_images_yfcc,
        "images_with_boxes": images_with_boxes,
        "min_conf": f"{min_conf:.2f}",
        "labels": labels_payload
    }
    return JSONResponse(payload)

async def recalc_freqs(request: Request):
    # Full histogram rebuild (threshold-independent)
    try:
        rebuild_histograms()
    except Exception as e:
        return PlainTextResponse(f"recalc failed: {e}", status_code=500)
    return PlainTextResponse("ok (hist rebuilt)")

# -----------------------------------------------------------------------------
# Starlette app
# -----------------------------------------------------------------------------
app = Starlette(routes=[
    Route("/", home),
    Route("/image", image),
    Route("/freqs", freqs_api),
    Route("/recalc", recalc_freqs, methods=["POST"]),
    Route("/freqs.js", freqs_client_js),
])