from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = ["CollectionOption"]


@dataclass(frozen=True, slots=True)
class CollectionOption:
    read_only: bool = False
    enable_mmap: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_only", bool(self.read_only))
        object.__setattr__(self, "enable_mmap", bool(self.enable_mmap))

    def _get_native(self):
        return _native_module.CollectionOption(self.read_only, self.enable_mmap)
