# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import logging
from importlib.resources import files
from io import BytesIO
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw

from ..constants import LABELS
from ..scope.slicer import get_image_by_id
from ..utils import color_for_label
from .db import (
    fetch_bboxes,
    read_images_with_boxes_at_threshold,
    read_label_counts_at_threshold,
    read_total_images_yfcc,
    rebuild_histograms,
)

log = logging.getLogger(__name__)

TEMPLATES_DIR = files("yfcc_scope") / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["bboxviewer"])
router.mount(
    "/static",
    StaticFiles(packages=[("yfcc_scope", "static")]),
    name="static",
)


@router.get("/", response_class=HTMLResponse, name="bboxviewer")
async def boxviewer(
    request: Request,
    image_file_id: str = "",
    label: list[str] | None = None,
    select_all: bool = True,
    min_conf: Annotated[float, Query(ge=0.4, le=1.0)] = 0.4,
) -> HTMLResponse:
    selected = set(label or [])

    img_src = (
        request.url_for("bbox_image").replace(query=request.url.query)
        if image_file_id
        else None
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "image_file_id": image_file_id,
            "all_checked": select_all,
            "select_all_val": "1" if select_all else "0",
            "min_conf": f"{min_conf:.2f}",
            "min_conf_fmt": f"{min_conf:.2f}",
            "labels": LABELS,
            "selected_labels": selected,
            "img_src": img_src,
        },
    )


class PngResponse(Response):
    media_type = "image/png"


@router.get("/image", response_class=PngResponse, name="bbox_image")
async def bbox_image(
    image_file_id: str = "",
    select_all: bool = True,
    label: list[str] | None = None,
    min_conf: Annotated[float, Query(ge=0.4, le=1.0)] = 0.4,
) -> PngResponse:
    selected = set(label or [])

    image_file_id = image_file_id.split("_", 1)[0]

    bboxes = await fetch_bboxes(image_file_id)
    if not bboxes:
        msg = f"bounding boxes for {image_file_id} not found"
        log.warning(msg)
        raise HTTPException(status_code=404, detail=msg)

    try:
        image_bytes = await get_image_by_id(image_file_id)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img.load()
    except Exception as e:
        msg = f"failed to fetch/open image: {image_file_id} err={e}"
        log.error(msg)
        raise HTTPException(status_code=502, detail=msg) from e

    W, H = img.size
    d = ImageDraw.Draw(img)

    for n, label, cx, cy, w, h, conf in bboxes:
        if conf is None or conf < min_conf:
            continue
        if not select_all and label not in selected:
            continue
        x0 = (cx - w / 2) * W
        x1 = (cx + w / 2) * W
        y0 = (cy - h / 2) * H
        y1 = (cy + h / 2) * H
        color = color_for_label(label)
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = f"{n}: {label} ({conf:.2f})"
        d.text((x0 + 2, y0 + 1), tag, fill=(255, 255, 255))

    with BytesIO() as buf:
        img.save(buf, "PNG")
        return PngResponse(content=buf.getvalue())


@router.get("/freqs", name="freqs")
async def freqs(
    min_conf: Annotated[float, Query(ge=0.4, le=1.0)] = 0.4,
) -> dict[str, Any]:
    counts, total_boxes, updated_at = await read_label_counts_at_threshold(min_conf)
    images_with_boxes = await read_images_with_boxes_at_threshold(min_conf)
    total_images_yfcc = await read_total_images_yfcc()

    labels_payload = {}
    for lab in LABELS:
        cnt = counts.get(lab, 0)
        frac = (float(cnt) / float(total_boxes)) if total_boxes else 0.0
        labels_payload[lab] = {"fraction": frac}

    return {
        "updated_at": updated_at,
        "total_images_yfcc": total_images_yfcc,
        "images_with_boxes": images_with_boxes,
        "min_conf": f"{min_conf:.2f}",
        "labels": labels_payload,
    }


@router.post("/recalc_freqs", response_class=PlainTextResponse)
async def recalc_freqs() -> str:
    try:
        await rebuild_histograms()
        return "ok (histogram rebuilt)"
    except Exception as err:
        msg = "Recalc histogram failed"
        raise HTTPException(status_code=500, detail=msg) from err
