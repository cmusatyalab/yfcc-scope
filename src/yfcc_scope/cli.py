# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import typer

from .scope import cli as scope_cli

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run a webserver instance to run the YFCC100M dataset viewers."""
    import uvicorn

    uvicorn.run("yfcc_scope.app:app", host=host, port=port, log_level="info")


app.add_typer(scope_cli.scope_app, name="scope")
app.add_typer(scope_cli.db_app, name="scope-db")
app.add_typer(scope_cli.test_app, name="scope-test")

if __name__ == "__main__":
    app()
