"""Application-scoped async engine and request-scoped database sessions."""

from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


class DatabaseManager:
    """Own one engine and a reusable factory; no connection is opened at import."""

    def __init__(self, database_url: str) -> None:
        try:
            url = make_url(database_url)
        except Exception as exc:
            raise ValueError("DATABASE_URL must be a PostgreSQL asyncpg URL") from exc
        if url.drivername != "postgresql+asyncpg" or not url.database:
            raise ValueError("DATABASE_URL must use postgresql+asyncpg and name a database")
        self.engine = create_async_engine(url, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def aclose(self) -> None:
        await self.engine.dispose()


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a fresh session and roll back uncommitted work after failures."""
    manager: DatabaseManager | None = request.app.state.db_manager
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "database_unavailable", "message": "Account service is unavailable."},
        )
    async with manager.session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
