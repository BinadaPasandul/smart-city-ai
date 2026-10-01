"""Explicit PostgreSQL user operations; create owns its write transaction."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User


def normalize_email(email: str) -> str:
    """Trim and lowercase the complete address for both writes and lookups."""
    return email.strip().lower()


class DuplicateAccountError(Exception):
    """A unique email constraint rejected a registration."""


class UserRepository:
    """One session per request; the create method commits or rolls back."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, email: str, password_hash: str, display_name: str | None
    ) -> User:
        user = User(
            email=normalize_email(email), password_hash=password_hash,
            display_name=display_name,
        )
        self._session.add(user)
        try:
            await self._session.commit()
            await self._session.refresh(user)
        except IntegrityError as exc:
            await self._session.rollback()
            if getattr(exc.orig, "sqlstate", None) == "23505":
                raise DuplicateAccountError() from None
            raise
        return user

    async def get_by_email(self, email: str) -> User | None:
        statement = select(User).where(User.email == normalize_email(email))
        return await self._session.scalar(statement)

    async def get_by_id(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)
