# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio
import tarfile
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

import niquests
import typer
from braceexpand import braceexpand
from sqlalchemy import URL, BigInteger, ForeignKey, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from tqdm.asyncio import tqdm

from .. import settings
from .util import batched


class SQLTable(AsyncAttrs, DeclarativeBase):
    pass


class Scope(SQLTable):
    __tablename__ = "scope_name"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)


class ScopeList(SQLTable):
    __tablename__ = "scope_list"

    scope_id: Mapped[int] = mapped_column(ForeignKey("scope_name.id"), primary_key=True)
    object_id: Mapped[int] = mapped_column(
        ForeignKey("scope_object.id"), primary_key=True
    )


class Shard(SQLTable):
    __tablename__ = "scope_shard"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(unique=True)


class Object(SQLTable):
    __tablename__ = "scope_object"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(unique=True)
    shard_id: Mapped[int] = mapped_column(ForeignKey("scope_shard.id"))
    offset: Mapped[int] = mapped_column(BigInteger)
    end: Mapped[int] = mapped_column(BigInteger)


DATABASE_URL = URL.create(
    "postgresql+asyncpg",
    username=settings.DB_USER,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
)
engine = create_async_engine(DATABASE_URL, echo=settings.DEBUG)


async def create_scope_db() -> None:
    async with engine.begin() as con:
        await con.run_sync(SQLTable.metadata.create_all)


async def build_shard_index(shard: str, items: Iterable[tuple[str, int, int]]) -> None:
    async with AsyncSession(engine) as session:
        shard_obj = Shard(url=shard)
        session.add(shard_obj)

        await session.flush()
        assert shard_obj.id is not None

        # There is no real reason to batch here, I think
        # the typical shard contains only about 10k-25k object
        session.add_all(
            [
                Object(key=key, shard_id=shard_obj.id, offset=off, end=end)
                for key, off, end in items
            ]
        )
        await session.commit()


async def list_scopes() -> AsyncIterator[tuple[str, int]]:
    async with AsyncSession(engine) as session:
        stmt = (
            select(Scope.name, func.count(ScopeList.object_id))
            .join(ScopeList, isouter=True)
            .group_by(Scope.id)
            .order_by(Scope.name)
        )
        result = await session.execute(stmt)
        for name, n in result:
            yield name, n


async def import_scope(scope: str, items: AsyncIterable[str]) -> int:
    async with AsyncSession(engine) as session:
        try:
            scope_obj = Scope(name=scope)
            session.add(scope_obj)
            await session.flush()
        except IntegrityError as err:
            msg = f"Scope {scope} already exists"
            raise FileExistsError(msg) from err

        # Here we have to batch because the in_ operator only accepts up to N
        # variables, and a scope can easily refer to 200k objects or more.
        # we might need to use lower level insert operations if we want to
        # reduce memory overhead on significantly larger scope lists.
        async for batch in batched(items, 1000):
            stmt = select(Object).where(Object.key.in_(batch))
            results = await session.execute(stmt)
            session.add_all(
                [
                    ScopeList(scope_id=scope_obj.id, object_id=obj.id)
                    for (obj,) in results
                ]
            )
            await session.flush()
        await session.commit()
    return await count_items_in_scope(scope)


async def export_scope(scope: str) -> AsyncIterator[str]:
    async with AsyncSession(engine) as session:
        stmt = select(Object.key).join(ScopeList).join(Scope).where(Scope.name == scope)
        result = await session.execute(stmt)
        for (name,) in result:
            yield name


async def delete_scope(scope: str) -> None:
    async with AsyncSession(engine) as session:
        stmt = select(Scope).where(Scope.name == scope)
        result = await session.execute(stmt)
        scope_obj = result.one()[0]

        stmt = delete(ScopeList).where(ScopeList.scope_id == scope_obj.id)
        await session.execute(stmt)
        await session.delete(scope_obj)
        await session.commit()


async def count_items_in_scope(scope: str) -> int:
    async with AsyncSession(engine) as session:
        stmt = (
            select(func.count("*"))
            .select_from(ScopeList)
            .join(Scope)
            .where(Scope.name == scope)
        )
        result = await session.execute(stmt)
        return result.scalars().one()


async def get_items_in_scope(
    scope: str, shard: int
) -> AsyncIterator[tuple[str, int, int]]:
    limit = settings.SCOPE_BATCH_SIZE
    offset = shard * settings.SCOPE_BATCH_SIZE

    async with AsyncSession(engine) as session:
        stmt = (
            select(Shard.url, Object.offset, Object.end)
            .join(Shard)
            .join(ScopeList)
            .join(Scope)
            .where(Scope.name == scope)
            .order_by(Object.shard_id, Object.offset)
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        for url, offset, end in result:
            yield url, offset, end


async def get_item_by_id(object_key: str) -> tuple[str, int, int]:
    async with AsyncSession(engine) as session:
        stmt = (
            select(Shard.url, Object.offset, Object.end)
            .join(Shard)
            .where(Object.key == object_key)
        )
        result = await session.execute(stmt)
        row = result.fetchone()
        if row is None:
            msg = f"{object_key} not found"
            raise KeyError(msg)
        return row


def generate_shard_index(url: str) -> Iterator[tuple[str, int, int]]:
    with niquests.get(url, stream=True) as response:
        response.raise_for_status()

        obj_key: str | None = None
        obj_off = obj_end = 0
        with tarfile.open(fileobj=response.raw, mode="r|") as shard:
            for entry in shard:
                cur_key = entry.name.rsplit(".", 1)[0]
                if obj_key != cur_key:
                    if obj_key is not None:
                        yield obj_key, obj_off, obj_end
                    obj_key = cur_key
                    obj_off = entry.offset
                # move end to include additional entries with the same key
                obj_end = shard.offset - 1
                # print("#", entry.name, entry.offset, shard.offset)
            # yield the final object
            if obj_key is not None:
                yield obj_key, obj_off, obj_end


async def index_shards(shards: list[str]) -> None:
    async for shard in tqdm(shards, unit=" shards"):
        items = tqdm(generate_shard_index(shard), unit=" objects")
        await build_shard_index(shard, items)


app = typer.Typer(help="Manipulate the scope database.", no_args_is_help=True)


@app.command()
def create() -> None:
    """Create scope related tables in the database."""
    asyncio.run(create_scope_db())


@app.command()
def index(descriptor: str) -> None:
    """Build an index for all objects in a webdataset (shard)."""
    shards = list(braceexpand(descriptor))
    asyncio.run(index_shards(shards))


if __name__ == "__main__":
    app()
