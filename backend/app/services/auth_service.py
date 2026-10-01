"""Registration, credential checks, and access-token issuance."""

import asyncio
import logging

from app.db.models.user import User
from app.repositories.user_repository import UserRepository, normalize_email
from app.security.password import PasswordHasher
from app.security.token import JwtIssuer

logger = logging.getLogger(__name__)


class InvalidCredentialsError(Exception):
    """One public failure category for unknown, wrong-password, or inactive users."""


class AuthService:
    def __init__(
        self, repository: UserRepository, hasher: PasswordHasher, issuer: JwtIssuer
    ) -> None:
        self._repository = repository
        self._hasher = hasher
        self._issuer = issuer

    async def register(
        self, *, email: str, password: str, display_name: str | None
    ) -> User:
        password_hash = await asyncio.to_thread(self._hasher.hash, password)
        user = await self._repository.create(
            email=normalize_email(email), password_hash=password_hash,
            display_name=display_name,
        )
        logger.info("Registration succeeded user_id=%s", user.id)
        return user

    async def login(self, *, email: str, password: str) -> tuple[str, int]:
        user = await self._repository.get_by_email(normalize_email(email))
        if user is None:
            await asyncio.to_thread(self._hasher.verify_unknown_user, password)
            logger.warning("Login rejected category=invalid_credentials")
            raise InvalidCredentialsError()
        valid = await asyncio.to_thread(self._hasher.verify, password, user.password_hash)
        if not valid or not user.is_active:
            logger.warning("Login rejected category=invalid_credentials")
            raise InvalidCredentialsError()
        logger.info("Login succeeded user_id=%s", user.id)
        return self._issuer.issue(user.id), self._issuer.expires_in
