# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

# Hatchling build hook for bundling Vite frontend with our Python backend.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

FRONTEND_DIR = Path("yfcc-viewer")
BACKEND_DIR = Path("src", "yfcc_scope")


def is_newer(src: Path, mtime: float) -> bool:
    for root, dirs, files in os.walk(src):
        for file in files:
            if Path(root, file).stat().st_mtime >= mtime:
                return True
        if "node_modules" in dirs:
            dirs.remove("node_modules")
    return False


class ViteBuildHook(BuildHookInterface):
    PLUGIN_NAME = "build-vite"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        package_json = root / FRONTEND_DIR / "package.json"
        dist_dir = root / BACKEND_DIR / "dist"
        index_html = dist_dir / "index.html"

        # `pip install -e .` editable install targets Python development.
        # don't bundle the frontend with the backend, but use `npm run dev`.
        # print("BUILDING", self.target_name, version)
        if self.target_name == "wheel" and version == "editable":
            self.app.display_info(
                "[build-vite] Development install. "
                f"Use `cd {FRONTEND_DIR} && npm run dev` to run frontend manually."
            )
            return

        if os.environ.get("SKIP_FRONTEND_BUILD") == "1":
            self.app.display_info("[build-vite] skipped via SKIP_FRONTEND_BUILD=1")
            return

        if not package_json.is_file():
            # self.app.display_info(
            #     f"[build-vite] no {package_json}, assuming prebuilt {dist_dir}"
            # )
            return

        if (
            os.environ.get("FORCE_FRONTEND_BUILD") != "1"
            and index_html.exists()
            and not is_newer(FRONTEND_DIR, index_html.stat().st_mtime)
        ):
            self.app.display_info(
                f"[build-vite] skipped, {BACKEND_DIR}/dist more recent than source in "
                f"{FRONTEND_DIR}. Force rebuild by setting FORCE_FRONTEND_BUILD=1"
            )
            return

        try:
            installer = shutil.which("npm")
            if installer is None:
                raise FileNotFoundError()

            subprocess.run(
                [str(installer), "install"],
                cwd=root / FRONTEND_DIR,
                check=True,
            )
            subprocess.run(
                [str(installer), "run", "build"],
                cwd=root / FRONTEND_DIR,
                check=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[build-vite] {exc}. "
                "Install `npm` or set SKIP_FRONTEND_BUILD=1 to bypass."
            ) from exc

        if not index_html.is_file():
            raise RuntimeError(
                f"[build-vite] build finished but {index_html} is missing. "
                f"Check {FRONTEND_DIR}/vite.config.ts build.outDir."
            )
        self.app.display_info(f"[build-vite] frontend installed in {dist_dir}")
