from __future__ import annotations

from abc import ABC, abstractmethod

from ..model.doc import Doc


class ReRanker(ABC):
    def __init__(self, topn: int = 10) -> None:
        if not isinstance(topn, int):
            raise TypeError("topn must be an int")
        if topn < 0:
            raise ValueError("topn must be >= 0")
        self._topn = topn

    @property
    def topn(self) -> int:
        return self._topn

    @abstractmethod
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        raise NotImplementedError


def _clone_doc_with_score(doc: Doc, score: float) -> Doc:
    return Doc(
        id=doc.id,
        fields=dict(doc.fields),
        score=score,
    )


__all__ = ["ReRanker"]
