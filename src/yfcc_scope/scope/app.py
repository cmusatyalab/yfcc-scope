# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import contextlib
import secrets
from collections.abc import AsyncGenerator, AsyncIterator
from importlib.resources import files
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import APIKeyHeader
from rich import print
from starlette.datastructures import Secret

from .. import settings
from ..types import ClusterID, TarStreamingResponse, WIDSDescriptor
from ..utils import generate_webdataset_shard
from .db import delete_scope as db_delete_scope
from .db import export_scope, get_items_in_scope, import_scope
from .slicer import generate_wids_descriptor, get_image_by_id
from .util import chunks_to_lines

apikey_scheme = APIKeyHeader(name="X-API-Key")


async def require_apikey(api_key: Annotated[Secret, Depends(apikey_scheme)]) -> str:
    is_correct_key = secrets.compare_digest(str(api_key), str(settings.SCOPE_API_KEY))
    if not is_correct_key:
        raise apikey_scheme.make_not_authenticated_error()
    return "API"


@contextlib.asynccontextmanager
async def scope_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    if str(settings.SCOPE_API_KEY) == str(settings.SESSION_KEY):
        print(f"[red]NOTE:[/red]\t  Temporary API key is: {settings.SCOPE_API_KEY}")
    yield
    # cleanup


scope_router = APIRouter(
    lifespan=scope_lifespan,
    tags=["scope"],
)
image_router = APIRouter()


class HTMLFileResponse(FileResponse):
    media_type = "text/html"


@scope_router.get("/", response_class=HTMLFileResponse)
async def animation() -> str:
    animation = (
        files("yfcc_scope").joinpath("scope").joinpath("webdataset-animation.html")
    )
    return str(animation)


@scope_router.get("/{scope}.json", tags=["webdataset index"])
async def get_webdataset_index(request: Request, scope: str) -> WIDSDescriptor:
    """Return a WIDS descriptor for a defined scope."""
    base_url = str(request.base_url)
    return await generate_wids_descriptor(scope, base_url)


@scope_router.get(
    "/{scope}-{shard}.tar",
    response_class=TarStreamingResponse,
    tags=["webdataset shard"],
)
async def get_webdataset_shard(
    scope: str, shard: Annotated[ClusterID, Path(ge=0)]
) -> AsyncIterator[bytes]:
    """Return a WebDataset shard for a defined scope."""
    items = get_items_in_scope(scope, shard)
    async for chunk in generate_webdataset_shard(items):
        yield chunk


@scope_router.get("/{scope}.scope")
async def get_scope(scope: str) -> StreamingResponse:
    """Retrieve all object identifiers in the defined scope."""
    items = (f"{key}\n" async for key in export_scope(scope))
    return StreamingResponse(items, media_type="text/plain")


@scope_router.post("/{scope}.scope", response_class=PlainTextResponse)
async def create_scope(
    request: Request, scope: str, apikey: Annotated[str, Depends(require_apikey)]
) -> str:
    """Create a new scope."""
    try:
        items = chunks_to_lines(request.stream())
        nitems = await import_scope(scope, items)
        return f'Created "{scope}" with {nitems} items'
    except FileExistsError as e:
        # scope already exists
        raise HTTPException(status_code=409, detail=e.args[0]) from None
    except (BufferError, UnicodeDecodeError):
        # unable to parse the request.body stream to object ids
        raise HTTPException(
            status_code=400, detail="Unable to process scope list"
        ) from None


@scope_router.delete("/{scope}.scope", status_code=204)
async def delete_scope(
    scope: str, apikey: Annotated[str, Depends(require_apikey)]
) -> None:
    """Delete a scope."""
    try:
        await db_delete_scope(scope)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f'Scope "{scope}" not found'
        ) from None


class JpegResponse(Response):
    media_type = "image/jpeg"


@image_router.get("/{image_id}.jpg", response_class=JpegResponse, name="get_image")
async def get_image(image_id: str) -> JpegResponse:
    """Extract the image from the webdataset and return as jpeg."""
    try:
        image_id = image_id.split("_", 1)[0]
        image = await get_image_by_id(image_id)
        return JpegResponse(image, headers={"Cache-Control": "public, max-age=3600"})
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f'Image "{image_id}" not found'
        ) from None


def image_url(request: Request, image_id: str) -> str:
    image_id = image_id.split("_", 1)[0]
    return str(request.url_for("get_image", image_id=image_id))
