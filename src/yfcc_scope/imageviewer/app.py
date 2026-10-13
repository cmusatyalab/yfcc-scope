# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import datetime
import os
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import Annotated, TypeVar

import numpy as np
import requests
from annotated_types import Len
from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import TypeAdapter
from stream_zip import ZIP_32, Method, async_stream_zip

from ..db import (
    execute_query,
    execute_wrapped_query,
    search_clip_ids,
    search_clip_images,
)
from ..embedding import ClipModel
from ..log import log
from ..scope.app import image_url
from ..scope.slicer import get_image_by_id
from ..settings import SCOPE_BASE
from ..utils import (
    build_vector_row,
    sanitize_scope_name,
    validate_sql,
)

router = APIRouter(tags=["image viewer"])


def _compute_clip_query(is_text, source, limit, search_fn):
    if is_text:
        feat = ClipModel().text_features([source])[0]
    else:
        feat = ClipModel().image_features(source)[0]

    embedding_list = feat.astype(np.float32).tolist()
    rows = search_fn(feat, limit)
    return feat, embedding_list, rows


def _do_clip_query(request: Request, is_text, source, limit):
    _, embedding_list, rows = _compute_clip_query(
        is_text, source, limit, search_clip_images
    )
    out_rows = [
        build_vector_row(image_file_id, image_url(request, image_file_id), 0, None)
        for image_file_id, _ in rows
    ]
    return {"embedding": embedding_list, "rows": out_rows}


@router.post("/clip_text_query")
def clip_text_query(
    request: Request,
    text: Annotated[str, Body()],
    limit: Annotated[int, Body(ge=1, le=1000)] = 20,
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"error": "text is required"})

    log.info("clip_text_query received text: %s, limit: %d", text, limit)
    return _do_clip_query(request, True, text, limit)


@router.post("/clip_image_query")
def clip_image_query(
    request: Request,
    image: Annotated[UploadFile, File()],
    limit: Annotated[int, Form(ge=1, le=1000)] = 20,
):
    image_bytes = image.file.read()
    log.info(
        "clip_image_query received image: %s (%d bytes), limit: %d",
        image.filename,
        len(image_bytes),
        limit,
    )
    return _do_clip_query(request, False, image_bytes, limit)


@router.post("/run_query")
def run_query(request: Request, sql: Annotated[str, Body()]):
    try:
        raw_sql = validate_sql(sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)}) from None

    log.info("run_query received SQL: %s", raw_sql)

    try:
        rows = execute_wrapped_query(raw_sql)
    except Exception as e:
        log.error("run_query failed: %s\nSQL: %s", e, raw_sql)
        raise HTTPException(
            status_code=500, detail={"error": f"Query failed: {e}"}
        ) from None

    out_rows = [
        build_vector_row(
            image_file_id, image_url(request, image_file_id), total_bboxes, counts_json
        )
        for image_file_id, path, total_bboxes, counts_json in rows
    ]
    log.info("run_query returned %d rows", len(out_rows))
    return {"rows": out_rows, "count": len(out_rows)}


@router.post("/run_query_count")
def run_query_count(sql: Annotated[str, Body()]):
    try:
        raw_sql = validate_sql(sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)}) from None

    log.info("run_query_count received SQL: \n%s", raw_sql)

    try:
        rows = execute_query(raw_sql)
    except Exception as e:
        log.error("run_query_count failed: %s\nSQL: %s", e, raw_sql)
        raise HTTPException(
            status_code=500, detail={"error": f"Query failed: {e}"}
        ) from None

    return {"rows": [{"count": r[0]} for r in rows]}


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
        raise HTTPException(status_code=404, detail={"error": "No IDs found"})

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


@router.post("/create_scope_coco")
async def create_scope_coco(query: Annotated[str, Body()], sql: Annotated[str, Body()]):
    scope_name = sanitize_scope_name(query)
    try:
        raw_sql = validate_sql(sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)}) from None

    log.info(
        "create_scope received query: %s, scope name: %s, SQL: \n%s",
        query,
        scope_name,
        raw_sql,
    )

    rows = execute_query(raw_sql)
    obj_keys = [str(row[0]).split("_", 1)[0] for row in rows if row[0]]
    return _publish_scope(scope_name, obj_keys)


@router.post("/create_scope_clip_text")
def create_scope_clip_text(
    query: Annotated[str, Body()],
    size: Annotated[int, Body(ge=1, le=200000)] = 200000,
):
    scope_name = sanitize_scope_name(query)

    log.info(
        "create_scope_clip_text received query: %s, size: %d",
        query,
        size,
    )

    image_ids = _do_clip_scope_query(True, query, size)
    log.info("create_scope_clip_image found %d image IDs", len(image_ids))
    return _publish_scope(scope_name, image_ids)


@router.post("/create_scope_clip_image")
def create_scope_clip_image(
    image_file: Annotated[UploadFile, File()],
    name: Annotated[str, Form()],
    size: Annotated[int, Body(ge=1, le=200000)] = 200000,
):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail={"error": "name is required"})

    scope_name = sanitize_scope_name(name)
    image_bytes = image_file.file.read()

    log.info(
        "create_scope_clip_image received image: %s (%d bytes), size: %d",
        getattr(image_file, "filename", "unknown"),
        len(image_bytes),
        size,
    )

    image_ids = _do_clip_scope_query(False, image_bytes, size)
    log.info("create_scope_clip_image found %d image IDs", len(image_ids))
    return _publish_scope(scope_name, image_ids)


V = TypeVar("V")


async def alist(items: Iterable[V]) -> AsyncIterator[V]:
    """wrapper to convert an Iterable to an AsyncIterator"""
    for item in items:
        yield item


StreamZipMember = tuple[str, datetime.datetime, int, Method, AsyncIterable[bytes]]


async def zip_member_files(image_ids: list[str]) -> AsyncIterator[StreamZipMember]:
    dt_now = datetime.datetime.now()
    for image_id in image_ids:
        img_id = image_id.split("_", 1)[0]
        content = await get_image_by_id(img_id)
        filename = f"{img_id}.jpg"
        yield filename, dt_now, 0o600, ZIP_32, alist([content])


DownloadZipFormIds = TypeAdapter(Annotated[list[str], Len(min_length=1)])


class ZipResponse(StreamingResponse):
    media_type = "application/zip"


@router.post("/download_zip", response_class=ZipResponse)
async def download_zip(
    ids: Annotated[str, Form()], response: Response
) -> AsyncIterable[bytes]:
    image_ids = DownloadZipFormIds.validate_json(ids)
    log.info(f"download_zip: requested {len(image_ids)} images")

    response.headers["Content-Disposition"] = 'attachment; filename="images.zip"'
    response.headers["Content-Type"] = "application/zip"
    async for chunk in async_stream_zip(zip_member_files(image_ids)):
        yield chunk
