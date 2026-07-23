# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import logging
import time

from sqlalchemy import (
    BIGINT,
    INT,
    REAL,
    SMALLINT,
    TEXT,
    TIMESTAMP,
    URL,
    ForeignKey,
    cast,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..constants import LABELS
from ..settings import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from ..utils import conf_to_bin

log = logging.getLogger(__name__)

DB_URL = URL.create(
    "postgresql+asyncpg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)
engine = create_async_engine(DB_URL, echo=True)


class SQLTable(AsyncAttrs, DeclarativeBase):
    pass


# Histogram Tables
class YFCC_Index(SQLTable):
    __tablename__ = "yfcc_index"

    image_file_id: Mapped[str] = mapped_column(TEXT, primary_key=True)
    path: Mapped[str] = mapped_column(TEXT)
    ts: Mapped[int] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )
    total_bboxes: Mapped[int] = mapped_column(INT, default=0)


class BoundingBoxes(SQLTable):
    __tablename__ = "bb_table"

    image_file_id: Mapped[str] = mapped_column(
        ForeignKey("yfcc_index.image_file_id"), primary_key=True
    )
    bounding_box_number: Mapped[int] = mapped_column(INT, primary_key=True)
    label: Mapped[str] = mapped_column(TEXT)
    confidence_score: Mapped[float] = mapped_column(REAL)
    center_x: Mapped[float] = mapped_column(REAL)
    center_y: Mapped[float] = mapped_column(REAL)
    width: Mapped[float] = mapped_column(REAL)
    height: Mapped[float] = mapped_column(REAL)


# Histogram Tables
class LabelConfidence_Histograms(SQLTable):
    __tablename__ = "yfcc_label_conf_hist"

    label: Mapped[str] = mapped_column(TEXT, primary_key=True)
    conf_bin: Mapped[int] = mapped_column(SMALLINT, primary_key=True)
    box_count: Mapped[int] = mapped_column(BIGINT)
    updated_at: Mapped[int] = mapped_column(INT)


class ImagesMaxbin_Histograms(SQLTable):
    __tablename__ = "yfcc_images_maxbin_hist"

    max_bin: Mapped[int] = mapped_column(SMALLINT, primary_key=True)
    image_count: Mapped[int] = mapped_column(BIGINT)
    updated_at: Mapped[int] = mapped_column(INT)


async def fetch(image_file_id: str):
    path_stmt = select(YFCC_Index.path).where(YFCC_Index.image_file_id == image_file_id)
    bbox_stmt = (
        select(BoundingBoxes)
        .where(BoundingBoxes.image_file_id == image_file_id)
        .where(BoundingBoxes.label is not None)
        .order_by(BoundingBoxes.bounding_box_number)
    )

    path = None
    cleaned = []
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(path_stmt)
            path = result.scalar()

            bboxes = await session.execute(bbox_stmt)
            cleaned = [
                (
                    bbox.bounding_box_number,
                    bbox.label,
                    bbox.center_x,
                    bbox.center_y,
                    bbox.width,
                    bbox.height,
                    bbox.confidence_score,
                )
                for (bbox,) in bboxes
            ]
    except Exception as e:
        log.error("DB fetch failed for image_file_id=%r: %s", image_file_id, e)
        return (None, [])

    if path is None:
        log.warning("image_file_id %r not found in yfcc_index", image_file_id)

    return (path, cleaned)


async def read_label_counts_at_threshold(min_conf: float):
    tb = conf_to_bin(min_conf)
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(func.max(LabelConfidence_Histograms.updated_at))
        )
        updated_at = result.scalar() or 0

        result = await session.execute(
            select(
                LabelConfidence_Histograms.label,
                func.sum(LabelConfidence_Histograms.box_count),
            )
            .where(LabelConfidence_Histograms.conf_bin >= tb)
            .group_by(LabelConfidence_Histograms.label)
        )
        counts = {lab: 0 for lab in LABELS}
        for lab, cnt in result:
            if lab in counts:
                counts[lab] = int(cnt)
        total_boxes = sum(counts.values())
    return counts, total_boxes, updated_at


async def read_images_with_boxes_at_threshold(min_conf: float) -> int:
    tb = conf_to_bin(min_conf)
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(func.sum(ImagesMaxbin_Histograms.image_count)).where(
                ImagesMaxbin_Histograms.max_bin >= tb
            )
        )
        return int(result.scalar() or 0)


async def read_total_images_yfcc() -> int:
    async with AsyncSession(engine) as session:
        result = await session.execute(select(func.count()).select_from(YFCC_Index))
        return int(result.scalar() or 0)


async def rebuild_histograms() -> None:
    now = int(time.time())

    async with engine.begin() as conn:
        conn.run_sync(SQLTable.metadata.create_all)

    async with AsyncSession(engine) as session:
        await session.execute(text("TRUNCATE TABLE yfcc_label_conf_hist;"))
        await session.execute(text("TRUNCATE TABLE yfcc_image_maxbin_hist;"))

        def bin_func(arg):
            return cast(
                func.least(func.greatest(func.floor(arg * 100.0 + 0.00001), 0), 100),
                SMALLINT,
            )

        stmt = (
            select(
                BoundingBoxes.label,
                bin_func(BoundingBoxes.confidence_score).label("conf_bin"),
                func.count(),
                now,
            )
            .where(BoundingBoxes.confidence_score.isnot(None))
            .group_by(BoundingBoxes.label, "conf_bin")
        )
        await session.execute(
            insert(LabelConfidence_Histograms).from_select(
                [
                    LabelConfidence_Histograms.label,
                    LabelConfidence_Histograms.conf_bin,
                    LabelConfidence_Histograms.box_count,
                    LabelConfidence_Histograms.updated_at,
                ],
                stmt,
            )
        )

        subq = (
            select(
                BoundingBoxes.image_file_id,
                bin_func(func.max(BoundingBoxes.confidence_score)).label("max_bin"),
            )
            .where(BoundingBoxes.confidence_score.isnot(None))
            .group_by(BoundingBoxes.image_file_id)
            .subquery()
        )
        stmt = select(subq.c.max_bin, func.count(), now).group_by(subq.c.max_bin)
        await session.execute(
            insert(ImagesMaxbin_Histograms).from_select(
                [
                    ImagesMaxbin_Histograms.max_bin,
                    ImagesMaxbin_Histograms.image_count,
                    ImagesMaxbin_Histograms.updated_at,
                ],
                stmt,
            )
        )
        session.commit()


async def cleanup() -> None:
    """Release any remaining connections from the sqlalchemy pool"""
    await engine.dispose()
