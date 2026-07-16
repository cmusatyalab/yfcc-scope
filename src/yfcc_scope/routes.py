# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import datetime
import json
import urllib.parse
from importlib.resources import files
from io import BytesIO

import numpy as np
import open_clip
import requests
import torch
from PIL import Image, ImageDraw
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from starlette.templating import Jinja2Templates
from stream_zip import ZIP_32, stream_zip

from .constants import LABELS
from .db import (
    conf_hist_sync,
    execute_query,
    execute_wrapped_query,
    fetch,
    fetch_images_for_labels,
    fetch_paths_by_ids,
    read_images_with_boxes_at_threshold,
    read_label_counts_at_threshold,
    read_total_images_yfcc,
    rebuild_histograms,
    rebuild_vector_table,
    search_clip_ids,
    search_clip_images,
    vector_rows_sync,
)
from .log import log
from .settings import MAX_LIMIT, SCOPE_BASE
from .utils import (
    build_vector_row,
    color_for_label,
    parse_conf_ranges_0_100,
    sanitize_scope_name,
    validate_sql,
)

TEMPLATES_DIR = files("yfcc_scope") / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Load CLIP model and tokenizer once at startup
_model, _preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_model.eval()
_tokenizer = open_clip.get_tokenizer("ViT-B-32")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = _model.to(_device)


async def redirect_to_viewer(request: Request):
    rel_path = request.url.path.lstrip("/")
    target = f"/viewer/{rel_path}" if rel_path else "/viewer"
    return RedirectResponse(url=target, status_code=302)


def _parse_min_conf(qp, default=0.4):
    try:
        min_conf = float(qp.get("min_conf", f"{default}"))
    except ValueError:
        min_conf = default
    return max(0.0, min(1.0, round(min_conf, 2)))


def _parse_limit_offset(qp, default_limit, max_limit):
    try:
        limit = int(qp.get("limit", f"{default_limit}"))
    except ValueError as exc:
        raise ValueError("limit must be an int") from exc
    try:
        offset = int(qp.get("offset", "0"))
    except ValueError as exc:
        raise ValueError("offset must be an int") from exc

    limit = max(1, min(max_limit, limit))
    offset = max(0, offset)
    return limit, offset


async def home(request: Request):
    qp = request.query_params
    image_file_id = qp.get("image_file_id", "")
    selected = set(qp.getlist("label"))
    all_checked = qp.get("select_all", "1") == "1"
    min_conf = _parse_min_conf(qp, default=0.4)

    img_params = [
        f"image_file_id={image_file_id}",
        f"select_all={'1' if all_checked else '0'}",
        f"min_conf={min_conf:.2f}",
    ]
    if not all_checked:
        img_params += [f"label={lab}" for lab in selected]
    img_src = "/image?" + "&".join(img_params) if image_file_id else ""

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "image_file_id": image_file_id,
            "all_checked": all_checked,
            "select_all_val": "1" if all_checked else "0",
            "min_conf": f"{min_conf:.2f}",
            "min_conf_fmt": f"{min_conf:.2f}",
            "labels": LABELS,
            "selected_labels": selected,
            "img_src": img_src,
        },
    )


async def image(request: Request):
    qp = request.query_params
    image_file_id = qp.get("image_file_id", "")
    all_checked = qp.get("select_all", "1") == "1"
    selected = set(qp.getlist("label"))
    min_conf = _parse_min_conf(qp, default=0.4)

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


def _compute_text_features(texts):
    text_input = _tokenizer(texts).to(_device)
    with torch.no_grad(), torch.autocast(_device):
        text_feat = _model.encode_text(text_input)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.float().cpu().numpy().astype(np.float16, copy=False)


def _compute_image_features(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_tensor = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad(), torch.autocast(_device):
        img_feat = _model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    return img_feat.float().cpu().numpy().astype(np.float16, copy=False)


def _compute_clip_query(is_text, source, limit, search_fn):
    if is_text:
        feat = _compute_text_features([source])[0]
    else:
        feat = _compute_image_features(source)[0]

    embedding_list = feat.astype(np.float32).tolist()
    rows = search_fn(feat, limit)
    return feat, embedding_list, rows


def _do_clip_query(is_text, source, limit):
    _, embedding_list, rows = _compute_clip_query(
        is_text, source, limit, search_clip_images
    )
    out_rows = [
        build_vector_row(image_file_id, path, 0, None) for image_file_id, path in rows
    ]

    return {"embedding": embedding_list, "rows": out_rows}


async def clip_text_query(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    try:
        limit = int(body.get("limit", 20))
    except (ValueError, TypeError):
        limit = 20
    limit = max(1, min(1000, limit))

    log.info("clip_text_query received text: %s, limit: %d", text, limit)

    results = await run_in_threadpool(_do_clip_query, True, text, limit)
    return JSONResponse(results)


async def clip_image_query(request: Request):
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "Invalid form data"}, status_code=400)

    image_file = form.get("image")
    if not image_file:
        return JSONResponse({"error": "image file is required"}, status_code=400)

    try:
        limit = int(form.get("limit", "20"))
    except (ValueError, TypeError):
        limit = 20
    limit = max(1, min(1000, limit))

    image_bytes = await image_file.read()
    log.info(
        "clip_image_query received image: %s (%d bytes), limit: %d",
        getattr(image_file, "filename", "unknown"),
        len(image_bytes),
        limit,
    )

    results = await run_in_threadpool(_do_clip_query, False, image_bytes, limit)
    return JSONResponse(results)


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
        return JSONResponse({"error": f"Query failed: {e}"}, status_code=500)

    out_rows = [
        build_vector_row(image_file_id, path, total_bboxes, counts_json)
        for image_file_id, path, total_bboxes, counts_json in rows
    ]

    log.info("run_query returned %d rows", len(out_rows))
    return JSONResponse({"rows": out_rows, "count": len(out_rows)})


async def run_query_count(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        raw_sql = validate_sql(body.get("sql"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info("run_query_count received SQL: \n%s", raw_sql)

    try:
        rows = await run_in_threadpool(execute_query, raw_sql)
    except Exception as e:
        log.error("run_query_count failed: %s\nSQL: %s", e, raw_sql)
        return JSONResponse({"error": f"Query failed: {e}"}, status_code=500)

    return JSONResponse({"rows": [{"count": r[0]} for r in rows]})


def _do_clip_scope_query(is_text, source, limit):
    _, _, rows = _compute_clip_query(is_text, source, limit, search_clip_ids)
    image_ids = [str(row[0].split("_", 1)[0]) for row in rows if row[0]]
    return image_ids


def _scope_body_from_ids(image_ids):
    if not image_ids:
        return ""
    return "\n".join(image_ids) + "\n"


def _publish_scope(scope_name, image_ids):
    body = _scope_body_from_ids(image_ids)
    if not body:
        return JSONResponse({"error": "No IDs found"}, status_code=404)

    import os

    resp = requests.post(
        f"{SCOPE_BASE}/{scope_name}.scope",
        headers={"X-API-Key": os.environ.get("SCOPE_API_KEY", "")},
        data=body,
    )

    message = resp.text.strip() or resp.reason
    payload = {"message": message}
    if resp.status_code >= 400:
        payload["error"] = message
    return JSONResponse(content=payload, status_code=resp.status_code)


async def create_scope_coco(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        query = body.get("query")
        scope_name = sanitize_scope_name(query)
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        raw_sql = validate_sql(body.get("sql"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info(
        "create_scope received query: %s, scope name: %s, SQL: \n%s",
        query,
        scope_name,
        raw_sql,
    )

    rows = await run_in_threadpool(execute_query, raw_sql)
    obj_keys = [str(row[0]).split("_", 1)[0] for row in rows if row[0]]
    return _publish_scope(scope_name, obj_keys)


async def create_scope_clip_text(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        size = int(body.get("size", 200000))
        query = body.get("query")
        scope_name = sanitize_scope_name(query)
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    log.info(
        "create_scope_clip_text received query: %s, size: %d",
        query,
        size,
    )

    image_ids = await run_in_threadpool(_do_clip_scope_query, True, query, size)
    log.info("create_scope_clip_image found %d image IDs", len(image_ids))
    return _publish_scope(scope_name, image_ids)


async def create_scope_clip_image(request: Request):
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "Invalid form data"}, status_code=400)

    image_file = form.get("image")
    if not image_file:
        return JSONResponse({"error": "image file is required"}, status_code=400)

    name = (form.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    try:
        size = int(form.get("size", 200000))
    except (ValueError, TypeError):
        size = 200000
    size = max(1, min(200000, size))

    scope_name = sanitize_scope_name(name)
    image_bytes = await image_file.read()
    log.info(
        "create_scope_clip_image received image: %s (%d bytes), size: %d",
        getattr(image_file, "filename", "unknown"),
        len(image_bytes),
        size,
    )

    image_ids = await run_in_threadpool(_do_clip_scope_query, False, image_bytes, size)
    log.info("create_scope_clip_image found %d image IDs", len(image_ids))
    return _publish_scope(scope_name, image_ids)


async def download_zip(request: Request):
    # Parse image id list from request body
    try:
        body = await request.json()
        image_ids = body.get("ids", [])
    except Exception:
        # Fallback for form-data
        try:
            raw_body = await request.body()
            parsed = urllib.parse.parse_qs(raw_body.decode("utf-8"))
            ids_str = parsed.get("ids", ["[]"])[0]
            image_ids = json.loads(ids_str)
        except Exception as e:
            log.error(f"Failed to parse form body: {e}")
            return JSONResponse({"error": "Invalid request"}, status_code=400)

    if not image_ids:
        return JSONResponse({"error": "No IDs provided"}, status_code=400)

    log.info(f"download_zip: requested {len(image_ids)} images")

    # Get the paths of all requested image IDs in a single DB query
    id_to_path = fetch_paths_by_ids(image_ids)

    def zip_files():
        dt_now = datetime.datetime.now()

        session = requests.Session()

        # Fetch a single image into memory
        def fetch_image(img_id):
            path = id_to_path.get(img_id)
            if not path:
                return img_id, None, None, None
            try:
                resp = session.get(path, timeout=10)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "image/jpeg")
                if "png" in content_type:
                    ext = "png"
                elif "gif" in content_type:
                    ext = "gif"
                else:
                    ext = "jpg"

                return img_id, ext, resp.content, None
            except Exception as e:
                return img_id, None, None, e

        import concurrent.futures

        # Download chunk of images in parallel, yielding results as they arrive
        # for streaming zip generation
        chunk_size = 5
        for i in range(0, len(image_ids), chunk_size):
            chunk_ids = image_ids[i : i + chunk_size]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=chunk_size
            ) as executor:
                # Submit fetch tasks for the current chunk of images to thread pool
                futures = [executor.submit(fetch_image, img_id) for img_id in chunk_ids]

                # Extract results and yield them for zip streaming
                for future in futures:
                    img_id, ext, content, err = future.result()
                    if err is not None:
                        log.error(f"download_zip: error fetching {img_id}: {err}")
                        continue
                    if ext is None:
                        log.warning(f"download_zip: {img_id} not found in DB")
                        continue

                    filename = f"{img_id}.{ext}"
                    yield filename, dt_now, 0o600, ZIP_32, [content]

    headers = {
        "Content-Disposition": 'attachment; filename="images.zip"',
        "Content-Type": "application/zip",
    }

    return StreamingResponse(
        stream_zip(zip_files()), headers=headers, media_type="application/zip"
    )


async def freqs_api(request: Request):
    qp = request.query_params
    min_conf = _parse_min_conf(qp, default=0.4)

    counts, total_boxes, updated_at = read_label_counts_at_threshold(min_conf)
    images_with_boxes = read_images_with_boxes_at_threshold(min_conf)
    total_images_yfcc = read_total_images_yfcc()

    labels_payload = {}
    for lab in LABELS:
        cnt = counts.get(lab, 0)
        frac = (float(cnt) / float(total_boxes)) if total_boxes else 0.0
        labels_payload[lab] = {"fraction": frac}

    return JSONResponse(
        {
            "updated_at": updated_at,
            "total_images_yfcc": total_images_yfcc,
            "images_with_boxes": images_with_boxes,
            "min_conf": f"{min_conf:.2f}",
            "labels": labels_payload,
        }
    )


async def recalc_freqs(request: Request):
    try:
        await run_in_threadpool(rebuild_histograms)
    except Exception as e:
        return PlainTextResponse(f"recalc failed: {e}", status_code=500)
    return PlainTextResponse("ok (hist rebuilt)")


async def images_api(request: Request):
    qp = request.query_params
    labels = [label.strip() for label in qp.getlist("label") if label.strip()]
    try:
        limit, offset = _parse_limit_offset(qp, default_limit=50, max_limit=MAX_LIMIT)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    conf_ranges = parse_conf_ranges_0_100((qp.get("conf") or "").strip())

    try:
        images = await run_in_threadpool(
            fetch_images_for_labels, labels, limit, offset, conf_ranges
        )
    except Exception as e:
        return JSONResponse({"error": f"images query failed: {e}"}, status_code=500)

    return JSONResponse(
        {"labels": labels, "limit": limit, "offset": offset, "images": images}
    )


async def conf_hist(request: Request):
    labels = request.query_params.getlist("label")
    if not labels:
        return JSONResponse({"error": "need ?label=cat&label=dog"}, status_code=400)
    try:
        bins = await run_in_threadpool(conf_hist_sync, labels)
    except Exception as e:
        return JSONResponse({"error": f"conf_hist failed: {e}"}, status_code=500)
    return JSONResponse({"labels": labels, "bins": bins})


async def vector_rows_api(request: Request):
    qp = request.query_params
    try:
        limit, offset = _parse_limit_offset(qp, default_limit=5000, max_limit=20000)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        min_total = int(qp.get("min_total_bboxes", "1"))
    except ValueError:
        min_total = 1
    min_total = max(0, min(10_000, min_total))

    try:
        rows, updated_at = await run_in_threadpool(
            vector_rows_sync, limit, offset, min_total
        )
    except Exception as e:
        return JSONResponse({"error": f"vector_rows failed: {e}"}, status_code=500)

    out_rows = [
        build_vector_row(image_file_id, path, total_bboxes, counts_json)
        for image_file_id, path, total_bboxes, counts_json in rows
    ]

    return JSONResponse(
        {"updated_at": updated_at, "limit": limit, "offset": offset, "rows": out_rows}
    )


async def recalc_vectors(request: Request):
    try:
        await run_in_threadpool(rebuild_vector_table)
    except Exception as e:
        return PlainTextResponse(f"recalc vectors failed: {e}", status_code=500)
    return PlainTextResponse("ok (vectors rebuilt)")
