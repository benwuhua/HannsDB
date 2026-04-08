from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real

from ..model.doc import Doc
from ..typing import MetricType
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


class WeightedReRanker(ReRanker):
    def __init__(
        self,
        topn: int = 10,
        metric: MetricType | str = MetricType.L2,
        metrics: dict[str, MetricType | str] | None = None,
        weights: dict[str, Real] | None = None,
    ) -> None:
        super().__init__(topn=topn)
        self._metric = self._normalize_metric(metric)
        self._metrics = self._normalize_metrics(metrics)
        self._weights = self._normalize_weights(weights)

    @staticmethod
    def _normalize_metric(metric: MetricType | str) -> MetricType:
        try:
            return MetricType(metric)
        except ValueError as error:
            raise ValueError("metric must be one of l2, ip, cosine") from error

    @staticmethod
    def _normalize_weights(weights: dict[str, Real] | None) -> dict[str, float]:
        if weights is None:
            return {}
        if not isinstance(weights, dict):
            raise TypeError("weights must be a dict[str, number] or None")

        normalized: dict[str, float] = {}
        for field_name, weight in weights.items():
            if not isinstance(field_name, str):
                raise TypeError("weights keys must be strings")
            if not isinstance(weight, Real) or isinstance(weight, bool):
                raise TypeError("weights values must be numbers")
            normalized[field_name] = float(weight)
        return normalized

    @staticmethod
    def _normalize_metrics(
        metrics: dict[str, MetricType | str] | None,
    ) -> dict[str, MetricType]:
        if metrics is None:
            return {}
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict[str, metric] or None")

        normalized: dict[str, MetricType] = {}
        for field_name, metric in metrics.items():
            if not isinstance(field_name, str):
                raise TypeError("metrics keys must be strings")
            normalized[field_name] = WeightedReRanker._normalize_metric(metric)
        return normalized

    @property
    def metric(self) -> MetricType:
        return self._metric

    @property
    def metrics(self) -> dict[str, MetricType]:
        return dict(self._metrics)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @staticmethod
    def _resolve_field_key(field_name: str, mapping: dict[str, object]) -> str | None:
        if field_name in mapping:
            return field_name
        base_field_name, separator, suffix = field_name.rpartition("#")
        if separator and suffix.isdigit() and base_field_name in mapping:
            return base_field_name
        return None

    @staticmethod
    def _normalize_score(score: float, metric: MetricType) -> float:
        if metric == MetricType.L2:
            return 1.0 - 2.0 * math.atan(score) / math.pi
        if metric == MetricType.Ip:
            return 0.5 - math.atan(score) / math.pi
        if metric == MetricType.Cosine:
            return 1.0 - score / 2.0
        raise ValueError("unsupported metric")

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Doc] = {}

        for field_name, query_result in query_results.items():
            metric_key = self._resolve_field_key(field_name, self._metrics)
            field_metric = self._metrics.get(metric_key, self._metric)
            weight_key = self._resolve_field_key(field_name, self._weights)
            field_weight = self._weights.get(weight_key, 1.0)
            for doc in query_result:
                normalized_score = self._normalize_score(doc.score, field_metric)
                scores[doc.id] += normalized_score * field_weight
                documents.setdefault(doc.id, doc)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            _clone_doc_with_score(documents[doc_id], score)
            for doc_id, score in ranked[: self.topn]
        ]


__all__ = ["RrfReRanker", "WeightedReRanker"]
