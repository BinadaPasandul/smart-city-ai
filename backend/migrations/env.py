"""Alembic environment using the same asyncpg URL and ORM metadata as the app."""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.db.base import Base
from app.db.models import User  # noqa: F401 - imports table into Base.metadata


def database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is required for Alembic migrations")
    parsed = make_url(url)
    if parsed.drivername != "postgresql+asyncpg" or not parsed.database:
        raise RuntimeError("DATABASE_URL must use postgresql+asyncpg and name a database")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(), target_metadata=Base.metadata,
        literal_binds=True, dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_with_connection(connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run_migrations_with_connection)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
