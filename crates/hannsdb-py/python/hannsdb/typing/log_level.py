from __future__ import annotations

from ._enum import _StringEnum

__all__ = ["LogLevel"]


class LogLevel(_StringEnum):
    Debug = "debug"
    Info = "info"
    Warn = "warn"
    Error = "error"
    # SCREAMING_SNAKE_CASE aliases for zvec compatibility
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    WARNING = "warn"  # Python logging convention alias
