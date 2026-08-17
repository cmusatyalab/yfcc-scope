# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .db import delete_scope, export_scope, import_scope, list_scopes
from .util import async_reader

app = typer.Typer(help="Add/remove scopes.", no_args_is_help=True)


@app.command()
def list() -> None:
    """Show a list of available scopes."""

    async def list_() -> None:
        table = Table()
        table.add_column("Scope", no_wrap=True)
        table.add_column("# Items")

        async for scope, count in list_scopes():
            table.add_row(scope, str(count))

        console = Console()
        console.print(table)

    asyncio.run(list_())


@app.command("import")
def import_(scopefile: Path, scope: str | None = None) -> None:
    """Import a new scope from a file."""
    if scope is None:
        scope = scopefile.stem

    if not scopefile.is_file():
        print(f"Scope {scopefile} does not exist")
        raise typer.Exit()

    try:
        nitems = asyncio.run(import_scope(scope, async_reader(scopefile)))
    except FileExistsError as err:
        print(err.args[0])
        raise typer.Exit() from err

    print(f'Scope from {scopefile} added as "{scope}" with {nitems} items.')


@app.command()
def export(scope: str) -> None:
    """Export a scope to stdout."""

    async def export_to_stdout(scope: str) -> None:
        async for key in export_scope(scope):
            print(key)

    asyncio.run(export_to_stdout(scope))


@app.command()
def delete(scope: str) -> None:
    """Remove a scope."""
    try:
        asyncio.run(delete_scope(scope))
    except KeyError as err:
        print(err.args[0])
        raise typer.Exit() from err


if __name__ == "__main__":
    app()
