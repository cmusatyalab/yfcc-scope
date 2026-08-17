# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import typer

from .db import app as db_app
from .scope import app as scope_app
from .test import app as test_app

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run a webserver instance to generate scoped datasets."""
    import uvicorn

    uvicorn.run("hawk_scope.app:app", host=host, port=port, log_level="info")


app.add_typer(db_app, name="db")
app.add_typer(scope_app, name="scope")
app.add_typer(test_app, name="test")

if __name__ == "__main__":
    app()
