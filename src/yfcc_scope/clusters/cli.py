# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio

import typer
from rich import print
from tqdm.asyncio import tqdm

from ..scope.slicer import generate_shard
from .clusters import ClusterID, get_items_in_cluster, scoped_wids_descriptor

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run a webserver instance to generate scoped datasets."""
    import uvicorn

    uvicorn.run("yfcc_scope.clusters.app:app", host=host, port=port, log_level="info")


async def create_wids(embedding: str, clusters: list[ClusterID] | None) -> None:
    base_url = "http://localhost:8000/"
    wids_descriptor = await scoped_wids_descriptor(embedding, base_url, clusters)
    print(wids_descriptor)


@app.command()
def wids(embedding: str, cluster: list[int] | None = None) -> None:
    """Generate a 'WIDS' json descriptor for the embedding.

    This would normally be the response to URL_BASE/<embedding>.json
    """
    asyncio.run(create_wids(embedding, cluster))


async def create_shard(embedding: str, cluster: ClusterID) -> None:
    items = get_items_in_cluster(embedding, cluster)
    items = tqdm(items)

    with open(f"{embedding}-{cluster:06d}.tar", "wb") as f:
        async for obj in generate_shard(items):
            f.write(obj)


@app.command()
def shard(embedding: str, cluster: int) -> None:
    """Generate a shard tar archive for the embedding.

    This would normally be the response to accessing
    URL_BASE/<embedding>-{cluster:06d}.tar
    """
    asyncio.run(create_shard(embedding, cluster))


if __name__ == "__main__":
    app()
