# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request

from ..db import conf_hist_sync, fetch_images_for_labels
from ..settings import MAX_LIMIT
from ..utils import parse_conf_ranges_0_100

router = APIRouter(tags=["dashboard"])


@router.get("/images")
def images_api(
    request: Request,
    label: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    conf: str = "",
):
    labels = [lbl for lbl in (lbl.strip() for lbl in label or []) if lbl]
    conf_ranges = parse_conf_ranges_0_100(conf.strip())

    images = fetch_images_for_labels(request, labels, limit, offset, conf_ranges)

    return {"labels": labels, "limit": limit, "offset": offset, "images": images}


@router.get("/conf_hist")
def conf_hist(
    label: Annotated[list[str], Query()],
):
    bins = conf_hist_sync(label)
    return {"labels": label, "bins": bins}
