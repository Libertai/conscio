from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

# bge-m3 produces 1024-dim vectors. Stored as float32 little-endian BLOBs and
# reranked with brute-force cosine over FTS candidates — fine up to ~50k facts;
# revisit vector indexing (e.g. sqlite-vec) only past that threshold.
EMBED_DIM = 1024

_DEFAULT_MODEL = "bge-m3"


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float] | None: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None: ...


class LibertAIEmbedder:
    """Embedder backed by the LLM client's embeddings endpoint (bge-m3)."""

    def __init__(self, client, model: str = _DEFAULT_MODEL) -> None:
        self.client = client
        self.model = model

    async def embed(self, text: str) -> list[float] | None:
        return await self.client.embed(text, model=self.model)

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return await self.client.embed_batch(texts, model=self.model)


class StubEmbedder:
    """Deterministic test embedder: hash -> unit vector, no network.

    The same text always maps to the same unit-norm vector, so tests can rely
    on cosine(text, text) == 1.0 and stable rankings across runs.
    """

    model = "stub"

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed(self, text: str) -> list[float] | None:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return [self._vector(text) for text in texts]


def pack(vec: list[float]) -> bytes:
    """Pack a vector as a float32 little-endian BLOB for SQLite storage."""
    return np.asarray(vec, dtype="<f4").tobytes()


def unpack(blob: bytes) -> np.ndarray:
    """Unpack a float32 little-endian BLOB back into a numpy array."""
    return np.frombuffer(blob, dtype="<f4")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; 0.0 if either has zero norm."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cosine similarity of query `q` against each row of `M` (batched rerank).

    Rows (or a query) with zero norm score 0.0.
    """
    q = np.asarray(q, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    row_norms = np.linalg.norm(M, axis=1)
    denom = row_norms * q_norm
    out = np.zeros(M.shape[0], dtype=np.float32)
    nonzero = denom > 0
    if q_norm > 0 and nonzero.any():
        out[nonzero] = (M[nonzero] @ q) / denom[nonzero]
    return out
