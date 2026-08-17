# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import TypeVar

import anyio

V = TypeVar("V")

# Theoretical max path length in a tar archive with PAX extensions
# we will probably never see keys this long (or longer).
MAX_KEY_LENGTH = 32767


# I don't want the python3.12 dependency just yet, here is a
# 'itertools.batched' implementation.
# Probably not as efficient, but it should work
async def batched(iterable: AsyncIterable[V], size: int) -> AsyncIterator[list[V]]:
    batch = []
    async for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


async def async_reader(path: Path | str) -> AsyncIterator[str]:
    """async reader that returns the lines in the file."""
    async with await anyio.open_file(path) as f:
        async for line in f:
            yield line.strip()


async def chunks_to_lines(stream: AsyncIterable[bytes]) -> AsyncIterator[str]:
    """Helper to convert a stream of binary chunks to lines"""
    body = b""
    async for chunk in stream:
        body += chunk
        while b"\n" in body:
            bline, body = body.split(b"\n", 1)
            line = bline.decode().strip()
            if line:
                yield line
        # DoS protection for when there are no newlines in the input
        if len(body) > MAX_KEY_LENGTH:
            msg = "Input line too long"
            raise BufferError(msg)
    line = body.decode().strip()
    if line:
        yield line
