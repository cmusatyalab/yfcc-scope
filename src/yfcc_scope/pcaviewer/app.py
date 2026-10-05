# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from typing import Annotated

from fastapi import APIRouter, Query, Request

from ..db import vector_rows_sync
from ..scope.app import image_url
from ..utils import build_vector_row

router = APIRouter(tags=["3d pcaviewer"])


@router.get("/vector_rows")
def vector_rows_api(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=20000)] = 5000,
    offset: Annotated[int, Query(ge=0)] = 0,
    min_total_bboxes: Annotated[int, Query(ge=0, le=10000)] = 1,
):
    rows, updated_at = vector_rows_sync(limit, offset, min_total_bboxes)
    out_rows = [
        build_vector_row(
            image_file_id, image_url(request, image_file_id), total_bboxes, counts_json
        )
        for image_file_id, _path, total_bboxes, counts_json in rows
    ]
    return {
        "updated_at": updated_at,
        "limit": limit,
        "offset": offset,
        "rows": out_rows,
    }
