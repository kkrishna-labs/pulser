"""Logging setup for Pulser.

Configure once via setup_logging(), then get_logger() returns
a child logger that propagates to the root (no duplicate handlers).
"""

from __future__ import annotations

import logging
import sys

from pulser.config import LOG_LEVEL

_configured = False


def setup_logging() -> None:
    """Configure root pulser logger once."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("pulser")
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger. Must call setup_logging() first (done in cli.py)."""
    return logging.getLogger(name)
