from __future__ import annotations

from enum import IntEnum

__all__ = ["LogType"]


class LogType(IntEnum):
    """Enumeration of log output destinations (for zvec compatibility)."""
    CONSOLE = 0
    FILE = 1
