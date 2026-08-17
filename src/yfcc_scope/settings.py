# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import secrets

import typer
from starlette.config import Config
from starlette.datastructures import Secret

config = Config(".env")

try:
    DEBUG: bool = config("DEBUG", cast=bool, default=False)

    DB_NAME: str = config.get("DB_NAME", default="yfcc")
    DB_USER: str = config.get("DB_USER", default="postgres")
    DB_PASSWORD: Secret = config.get("DB_PASSWORD", cast=Secret)
    DB_HOST: str = config.get("DB_HOST", default="localhost")
    DB_PORT: int = config.get("DB_PORT", default=5432)

    MAX_LIMIT: int = config.get("MAX_LIMIT", default=500)

    SESSION_KEY = Secret(secrets.token_urlsafe())
    SCOPE_API_KEY: Secret | str = config(
        "SCOPE_API_KEY", cast=Secret, default=str(SESSION_KEY)
    )
    SCOPE_BASE: str = config.get("SCOPE_BASE", default="internal")

    SCOPE_BATCH_SIZE: int = config("SCOPE_BATCH_SIZE", cast=int, default=10_000)
    SCOPE_FETCH_WINDOW: int = config("SCOPE_FETCH_WINDOW", cast=int, default=64)
except KeyError as err:
    print(f"Missing .env configuration file ({err})")
    raise typer.Exit() from err
