from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = ["CollectionOption"]


@dataclass(frozen=True)
class CollectionOption:
    read_only: bool = False
    enable_mmap: bool = True

    def __post_init__(self) -> None:
        if type(self.read_only) is not bool:
            raise TypeError("read_only must be a bool")
        if type(self.enable_mmap) is not bool:
            raise TypeError("enable_mmap must be a bool")

    def _get_native(self):
        return _native_module.CollectionOption(self.read_only, self.enable_mmap)
