"""Single SQLAlchemy declarative base for application tables."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
