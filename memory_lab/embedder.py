"""Turning text into vectors (embeddings) so memories can be found by meaning.

We use all-MiniLM-L6-v2: small (about 90 MB), runs on a laptop CPU, 384 numbers per text.
Vectors are normalised to length 1, so cosine similarity is just a dot product.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(model, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
