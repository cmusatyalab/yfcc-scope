# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import colorsys
import hashlib
import re

from .constants import BLOCKED_KEYWORDS, LABELS


def color_for_label(label: str):
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 360
    s, v = 0.75, 1.0
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def conf_to_bin(c: float) -> int:
    c = max(0.0, min(1.0, round(float(c), 2)))
    return int(round(c * 100))


def parse_conf_ranges_0_100(conf_str: str):
    if not conf_str:
        return None
    ranges = []
    for part in conf_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo_bin = int(lo_s)
            hi_bin = int(hi_s)
        else:
            lo_bin = hi_bin = int(part)
        lo_bin = max(0, min(100, lo_bin))
        hi_bin = max(0, min(100, hi_bin))
        if lo_bin > hi_bin:
            lo_bin, hi_bin = hi_bin, lo_bin
        lo = lo_bin / 100.0
        hi = min((hi_bin + 1) / 100.0, 1.0)
        ranges.append((lo, hi))
    return ranges if ranges else None


def validate_sql(raw_sql: str):
    sql = (raw_sql or "").strip()
    if not sql:
        raise ValueError("No SQL provided")

    sql = sql.rstrip().rstrip(";").rstrip()
    first_word = sql.lstrip().split()[0].upper() if sql else ""
    if first_word != "SELECT":
        raise ValueError("Only SELECT queries are permitted")

    upper_sql = sql.upper()

    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper_sql):
            raise ValueError(f"Query contains forbidden keyword: {kw}")

    # Keep user SQL simple since the backend already wraps it in a CTE.
    if re.search(r"\bWITH\b", upper_sql):
        raise ValueError(
            "Queries using WITH/CTEs are not allowed. Return a single SELECT statement."
        )

    # Block expensive correlated-subquery ranking patterns
    if re.search(
        r"ORDER\s+BY\s*\(\s*SELECT\s+(AVG|MAX|MIN|SUM|COUNT)\s*\(",
        upper_sql,
        flags=re.IGNORECASE,
    ):
        msg = (
            "Correlated subqueries in ORDER BY are not allowed. "
            "Use JOIN + GROUP BY + ORDER BY MAX(...) or AVG(...)."
        )
        raise ValueError(msg)

    if re.search(
        r"SELECT\s+(AVG|MAX|MIN|SUM|COUNT)\s*\([^)]*\)\s+FROM\s+BB_TABLE\s+WHERE\s+BB_TABLE\.IMAGE_FILE_ID\s*=",
        upper_sql,
        flags=re.IGNORECASE,
    ):
        msg = (
            "Correlated subqueries against bb_table are not allowed. "
            "Use JOIN + GROUP BY instead."
        )
        raise ValueError(msg)

    return sql


def sanitize_scope_name(query: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKC", query or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    return s[:64] or "scope"


def build_vector_row(image_file_id, path, total_bboxes, counts_json):
    counts = counts_json or {}
    item = {
        "image_file_id": image_file_id,
        "path": path,
        # currently using the same url for thumbnails, might want to change
        # this if we add separate thumbnail URLs in the future
        "thumb_url": path,
        "total_bboxes": int(total_bboxes or 0),
    }
    for lab in LABELS:
        item[lab] = int(counts.get(lab, 0) or 0)
    return item
