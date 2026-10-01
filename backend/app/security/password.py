"""Argon2id password hashing through the maintained pwdlib implementation."""

from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError


class PasswordHasher:
    def __init__(self) -> None:
        self._hasher = PasswordHash.recommended()
        self._dummy_hash = self._hasher.hash("dummy-password-for-timing-only")

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password: str, password_hash: str) -> bool:
        try:
            return self._hasher.verify(password, password_hash)
        except (UnknownHashError, ValueError):
            return False

    def verify_unknown_user(self, password: str) -> None:
        """Spend one normal verification on unknown accounts to reduce timing gaps."""
        self.verify(password, self._dummy_hash)
