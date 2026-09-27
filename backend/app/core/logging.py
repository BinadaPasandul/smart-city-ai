"""Small standard-library logging setup for the application."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging once without replacing existing handlers."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level.upper())
        return

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
