# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import datetime
import json
import urllib.parse

import numpy as np
import requests
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from stream_zip import ZIP_32, stream_zip

from .db import (
    execute_query,
    execute_wrapped_query,
    fetch_paths_by_ids,
    rebuild_vector_table,
    search_clip_ids,
    search_clip_images,
)
from .embedding import clip_image_features, clip_text_features
from .log import log
from .scope.app import image_url
from .settings import SCOPE_BASE
from .utils import (
    build_vector_row,
    sanitize_scope_name,
    validate_sql,
)


def _compute_clip_query(is_text, source, limit, search_fn):
    if is_text:
        feat = clip_text_features([source])[0]
    else:
        feat = clip_image_features(source)[0]

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


async def recalc_vectors(request: Request):
    try:
        await run_in_threadpool(rebuild_vector_table)
    except Exception as e:
        return PlainTextResponse(f"recalc vectors failed: {e}", status_code=500)
    return PlainTextResponse("ok (vectors rebuilt)")
