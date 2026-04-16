from __future__ import annotations

from enum import IntEnum

__all__ = ["IndexType"]


class IndexType(IntEnum):
    """Enumeration of supported index types.

    Values match zvec's IndexType for compatibility.
    """
    UNDEFINED = 0
    HNSW = 1
    IVF = 3
    FLAT = 4
    INVERT = 10
    HNSW_SQ = 11
    HNSW_HVQ = 12
    IVF_USQ = 13
