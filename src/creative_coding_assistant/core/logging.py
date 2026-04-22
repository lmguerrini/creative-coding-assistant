"""Logging setup kept separate from UI code."""

from __future__ import annotations

import logging
import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure Loguru while keeping stdlib log records visible."""

    normalized_level = level.upper()
    logging.basicConfig(level=getattr(logging, normalized_level, logging.INFO))
    logger.remove()
    logger.add(
        sys.stderr,
        level=normalized_level,
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {name} {message}",
    )
