from __future__ import annotations

from collections import defaultdict

from ..model.doc import Doc
from .rerank_function import ReRanker, _clone_doc_with_score


class RrfReRanker(ReRanker):
    def __init__(self, topn: int = 10, rank_constant: int = 60) -> None:
        super().__init__(topn=topn)
        if not isinstance(rank_constant, int):
            raise TypeError("rank_constant must be an int")
        if rank_constant < 0:
            raise ValueError("rank_constant must be >= 0")
        self._rank_constant = rank_constant

    @property
    def rank_constant(self) -> int:
        return self._rank_constant

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self._rank_constant + rank + 1)

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Doc] = {}

        for query_result in query_results.values():
            for rank, doc in enumerate(query_result):
                scores[doc.id] += self._rrf_score(rank)
                documents.setdefault(doc.id, doc)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            _clone_doc_with_score(documents[doc_id], score)
            for doc_id, score in ranked[: self.topn]
        ]


__all__ = ["RrfReRanker"]
