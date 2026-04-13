from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = ["AddColumnOption"]


@dataclass(frozen=True)
class AddColumnOption:
    concurrency: int = 0

    def __post_init__(self) -> None:
        if type(self.concurrency) is not int:
            raise TypeError("concurrency must be an int")
        if self.concurrency < 0:
            raise ValueError("concurrency must be >= 0")

    def _get_native(self):
        return _native_module.AddColumnOption(self.concurrency)
