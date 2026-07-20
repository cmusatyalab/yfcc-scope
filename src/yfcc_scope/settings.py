# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import typer
from starlette.config import Config
from starlette.datastructures import Secret

config = Config(".env")

try:
    DB_NAME: str = config.get("DB_NAME", default="yfcc")
    DB_USER: str = config.get("DB_USER", default="postgres")
    DB_PASSWORD: Secret = config.get("DB_PASSWORD", cast=Secret)
    DB_HOST: str = config.get("DB_HOST", default="localhost")
    DB_PORT: int = config.get("DB_PORT", default=5432)

    MAX_LIMIT: int = config.get("MAX_LIMIT", default=500)

    SCOPE_BASE: str = config.get("SCOPE_BASE")
    SCOPE_API_KEY: Secret = config.get("SCOPE_API_KEY", cast=Secret)
except KeyError as err:
    print(f"Missing .env configuration file ({err})")
    raise typer.Exit() from err
