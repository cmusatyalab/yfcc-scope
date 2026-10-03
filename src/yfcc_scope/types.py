# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from enum import Enum
from typing import Annotated, NewType

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


ClusterID = NewType("ClusterID", int)


class EmbeddingType(str, Enum):
    clip = "clip-2"
    dinov3 = "dinov3-2"


class Embeddings(BaseModel):
    embeddings: Annotated[
        list[list[float]], Field(description="List of embedding vectors")
    ]


class TarStreamingResponse(StreamingResponse):
    media_type = "application/x-tar"


class WebDatasetShard(BaseModel):
    url: str
    nsamples: Annotated[int, Field(ge=0)]


class WIDSDescriptor(BaseModel):
    wids_version: int = 1
    name: str
    description: str = ""
    base: str
    shardlist: list[WebDatasetShard]
