from __future__ import annotations

from enum import Enum


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        compact = normalized.replace("_", "")
        for member in cls:
            member_value = member.value
            if member_value == normalized:
                return member
            if member_value.replace("_", "") == compact:
                return member
        return None
