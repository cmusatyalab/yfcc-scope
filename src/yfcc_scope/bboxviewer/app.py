# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import logging
from importlib.resources import files
from io import BytesIO

from PIL import Image, ImageDraw
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

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


def clip(value: float, minv: float = 0.0, maxv: float = 1.0) -> float:
    return max(minv, min(maxv, value))


async def boxviewer(request: Request) -> HTMLResponse:
    qp = request.query_params
    image_file_id = qp.get("image_file_id", "")
    selected = set(qp.getlist("label"))
    all_checked = qp.get("select_all", "1") == "1"
    min_conf = clip(float(qp.get("min_conf", "0.4")), minv=0.4)

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
            "all_checked": all_checked,
            "select_all_val": "1" if all_checked else "0",
            "min_conf": f"{min_conf:.2f}",
            "min_conf_fmt": f"{min_conf:.2f}",
            "labels": LABELS,
            "selected_labels": selected,
            "img_src": img_src,
        },
    )


async def image(request: Request) -> Response:
    qp = request.query_params
    image_file_id = qp.get("image_file_id", "")
    all_checked = qp.get("select_all", "1") == "1"
    selected = set(qp.getlist("label"))
    min_conf = clip(float(qp.get("min_conf", "0.4")), minv=0.4)

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
        if not all_checked and label not in selected:
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
        return Response(buf.getvalue(), media_type="image/png")


async def freqs_api(request: Request) -> JSONResponse:
    qp = request.query_params
    min_conf = clip(float(qp.get("min_conf", "0.4")), minv=0.4)

    counts, total_boxes, updated_at = await read_label_counts_at_threshold(min_conf)
    images_with_boxes = await read_images_with_boxes_at_threshold(min_conf)
    total_images_yfcc = await read_total_images_yfcc()

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


async def recalc_freqs(request: Request) -> PlainTextResponse:
    await rebuild_histograms()
    return PlainTextResponse("ok (histogram rebuilt)")
    try:
        pass
    except Exception as err:
        msg = "Recalc histogram failed"
        raise HTTPException(status_code=500, detail=msg) from err


routes = [
    Route("/", boxviewer, name="bboxviewer"),
    Route("/image", image, name="bbox_image"),
    Mount("/static", StaticFiles(packages=[("yfcc_scope", "static")]), name="static"),
]
api_routes = [
    Route("/freqs", freqs_api, name="freqs"),
    Route("/recalc_freqs", recalc_freqs, methods=["POST"], name="recalc_freqs"),
]


app = Starlette(routes=routes)
app.mount("/api", api_routes)
