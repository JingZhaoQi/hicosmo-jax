"""
Logging utilities for HIcosmo.

Provides a consistent logging setup across the package with support for
verbosity levels and environment-based overrides.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_ENV_LEVEL = os.getenv("HICOSMO_LOGLEVEL", "").upper()


def _level_from_verbosity(verbosity: int) -> int:
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


def configure_logging(
    *,
    verbosity: int = 1,
    level: Optional[str] = None,
    fmt: Optional[str] = None,
) -> None:
    """
    Configure HIcosmo logging.

    Parameters
    ----------
    verbosity : int
        0 = WARNING, 1 = INFO, 2+ = DEBUG.
    level : str, optional
        Explicit log level name (overrides verbosity).
    fmt : str, optional
        Logging format. If None, a concise default is used.
    """
    if level is None:
        level = _ENV_LEVEL or logging.getLevelName(_level_from_verbosity(verbosity))

    log_level = logging.getLevelName(level)
    if isinstance(log_level, str):
        log_level = _level_from_verbosity(verbosity)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt or "%(levelname)s:%(name)s:%(message)s")
        )
        root.addHandler(handler)

    root.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with NullHandler by default."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
