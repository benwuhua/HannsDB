from __future__ import annotations

from .. import _native as _native_module

__all__ = ["Doc"]


def _normalize_vector(value):
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _normalize_fields(fields):
    if fields is None:
        return {}
    return {str(name): value for name, value in dict(fields).items()}


def _normalize_vectors(vectors):
    if vectors is None:
        return {}
    normalized = {}
    for name, vector in dict(vectors).items():
        normalized[str(name)] = _normalize_vector(vector)
    return normalized


class Doc:
    __slots__ = ("_id", "_score", "_fields", "_vectors", "_field_name")

    def __init__(
        self,
        id,
        vector=None,
        field_name="dense",
        fields=None,
        score=None,
        vectors=None,
    ):
        self._id = str(id)
        self._score = score
        self._fields = _normalize_fields(fields)
        self._vectors = _normalize_vectors(vectors)
        self._field_name = str(field_name)

        if vector is not None:
            self._vectors[self._field_name] = _normalize_vector(vector)

        if not self._vectors:
            self._field_name = str(field_name)
        elif self._field_name not in self._vectors:
            self._field_name = next(iter(self._vectors))

    @classmethod
    def _from_native(cls, doc):
        vectors = _normalize_vectors(getattr(doc, "vectors", None))
        field_name = next(iter(vectors), "dense")
        return cls(
            id=getattr(doc, "id"),
            score=getattr(doc, "score", None),
            fields=getattr(doc, "fields", None),
            vectors=vectors,
            field_name=field_name,
        )

    @property
    def id(self):
        return self._id

    @property
    def score(self):
        return self._score

    @property
    def fields(self):
        return dict(self._fields)

    @property
    def vectors(self):
        return {name: list(vector) for name, vector in self._vectors.items()}

    @property
    def field_name(self):
        return self._field_name

    def has_field(self, name):
        return str(name) in self._fields

    def has_vector(self, name):
        return str(name) in self._vectors

    def field(self, name):
        key = str(name)
        if key not in self._fields:
            raise KeyError(key)
        return self._fields[key]

    def vector(self, name):
        key = str(name)
        if key not in self._vectors:
            raise KeyError(key)
        return list(self._vectors[key])

    def field_names(self):
        return list(self._fields)

    def vector_names(self):
        return list(self._vectors)

    def _replace(self, **changes):
        allowed = {"id", "vector", "field_name", "fields", "score", "vectors"}
        unknown = set(changes) - allowed
        if unknown:
            unknown_names = ", ".join(sorted(unknown))
            raise TypeError(f"Doc._replace() got unexpected field names: {unknown_names}")

        data = {
            "id": changes.get("id", self.id),
            "vector": None,
            "field_name": changes.get("field_name", self.field_name),
            "fields": changes.get("fields", self.fields),
            "score": changes.get("score", self.score),
            "vectors": changes.get("vectors", self.vectors),
        }
        if "vector" in changes:
            data["vector"] = changes["vector"]
        return type(self)(**data)

    def _get_native(self):
        if not self._vectors:
            return _native_module.Doc(
                id=self.id,
                fields=self.fields,
                score=self.score,
            )

        field_name = self._field_name if self._field_name in self._vectors else next(iter(self._vectors))
        return _native_module.Doc(
            id=self.id,
            vector=list(self._vectors[field_name]),
            field_name=field_name,
            fields=self.fields,
            score=self.score,
        )

    def __repr__(self):
        return (
            "Doc("
            f"id={self.id!r}, score={self.score!r}, "
            f"fields={self.fields!r}, vectors={self.vectors!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, Doc):
            return NotImplemented
        return (
            self.id == other.id
            and self.score == other.score
            and self.fields == other.fields
            and self.vectors == other.vectors
        )
