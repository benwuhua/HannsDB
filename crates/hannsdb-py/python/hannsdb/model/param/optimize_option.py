from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = ["OptimizeOption"]


@dataclass(frozen=True)
class OptimizeOption:
    def _get_native(self):
        return _native_module.OptimizeOption()
