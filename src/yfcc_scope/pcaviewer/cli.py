# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from ..db import rebuild_vector_table

app = typer.Typer(
    help="3D PCA viewer db maintenance", no_args_is_help=True
)

console = Console()

@app.command()
def recalc_vectors():
    """Rebuild yfcc_image_label_counts table."""
    try:
        with console.status("Rebuilding..."):
            rebuild_vector_table()
        console.print("ok (vectors rebuilt)")
    except Exception as e:
        console.print_exception()

