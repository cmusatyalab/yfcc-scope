# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio

import typer
from rich import print
from tqdm.asyncio import tqdm

from .db import get_items_in_scope
from .slicer import generate_shard, generate_wids_descriptor

app = typer.Typer(help="Test wids and shard generators.", no_args_is_help=True)


async def create_wids(scope: str) -> None:
    base_url = "http://localhost:8000/"
    wids_descriptor = await generate_wids_descriptor(scope, base_url)
    print(wids_descriptor)


@app.command()
def wids(scope: str) -> None:
    """Generate a 'WIDS' json descriptor for the scope.

    This would normally be the response to URL_BASE/<scope>.json
    """
    asyncio.run(create_wids(scope))


async def create_shard(scope: str, shard: int) -> None:
    items = get_items_in_scope(scope, shard)
    items = tqdm(items)

    with open(f"{scope}-{shard:06d}.tar", "wb") as f:
        async for obj in generate_shard(items):
            f.write(obj)


@app.command()
def shard(scope: str, shard: int = 0) -> None:
    """Generate a shard tar archive for the scope.

    This would normally be the response to accessing
    URL_BASE/<scope>/webdataset-{shard:06d}.tar
    """
    asyncio.run(create_shard(scope, shard))


if __name__ == "__main__":
    app()
