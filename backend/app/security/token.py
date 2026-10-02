"""Issue minimal access tokens compatible with the existing JWT verifier."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt

from app.core.config import Settings


class JwtIssuer:
    def __init__(self, settings: Settings) -> None:
        if settings.jwt_secret is None or not settings.jwt_issuer or not settings.jwt_audience:
            raise ValueError("JWT issuance requires secret, issuer, and audience")
        self._secret = settings.jwt_secret
        self._algorithm = settings.jwt_algorithm
        self._issuer = settings.jwt_issuer
        self._audience = settings.jwt_audience
        self.expires_in = settings.jwt_access_token_expire_minutes * 60

    def issue(self, user_id: UUID) -> str:
        now = datetime.now(timezone.utc)
        return jwt.encode(
            {
                "sub": str(user_id),
                "iss": self._issuer,
                "aud": self._audience,
                "iat": now,
                "exp": now + timedelta(seconds=self.expires_in),
                "jti": str(uuid4()),
            },
            self._secret.get_secret_value(),
            algorithm=self._algorithm,
        )
