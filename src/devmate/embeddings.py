"""Embedding implementations used by DevMate."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Deterministic local embeddings for offline tests and smoke checks."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def _hash_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = list(digest)
        repeated = values * ((self.dimensions // len(values)) + 1)
        sliced = repeated[: self.dimensions]
        return [value / 255.0 for value in sliced]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_text(text)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._hash_text(text) for text in texts]
