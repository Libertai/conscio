from __future__ import annotations

import unittest

import numpy as np

from conscio.llm.client import LLMClient
from conscio.memory.embeddings import (
    EMBED_DIM,
    StubEmbedder,
    cosine,
    cosine_matrix,
    pack,
    unpack,
)


class _FakeEmbeddingData:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingsResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingData(v) for v in vectors]


class _FakeEmbeddingsAPI:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    async def create(self, *, model: str, input: list[str]):
        self.calls.append({"model": model, "input": input})
        if self.error is not None:
            raise self.error
        return _FakeEmbeddingsResponse([[1.0, 0.0, 0.0]] * len(input))


class _FakeSyncEmbeddingsAPI(_FakeEmbeddingsAPI):
    def create(self, *, model: str, input: list[str]):  # type: ignore[override]
        self.calls.append({"model": model, "input": input})
        if self.error is not None:
            raise self.error
        return _FakeEmbeddingsResponse([[1.0, 0.0, 0.0]] * len(input))


class _FakeOpenAI:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings


class PackUnpackTests(unittest.TestCase):
    def test_roundtrip_preserves_values(self) -> None:
        vec = [0.5, -1.25, 0.0, 3.75]
        blob = pack(vec)
        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), 4 * len(vec))
        out = unpack(blob)
        self.assertEqual(out.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(out, np.asarray(vec, dtype="<f4"))

    def test_roundtrip_full_dim_vector(self) -> None:
        rng = np.random.default_rng(7)
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32).tolist()
        out = unpack(pack(vec))
        self.assertEqual(out.shape, (EMBED_DIM,))
        np.testing.assert_allclose(out, vec, rtol=1e-6)

    def test_pack_is_little_endian_float32(self) -> None:
        self.assertEqual(pack([1.0]), b"\x00\x00\x80\x3f")


class CosineTests(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(a, b), 0.0, places=6)

    def test_opposite_vectors_score_minus_one(self) -> None:
        a = np.array([1.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(a, -a), -1.0, places=6)

    def test_zero_vector_scores_zero(self) -> None:
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertEqual(cosine(a, b), 0.0)

    def test_cosine_matrix_matches_pairwise_cosine(self) -> None:
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        M = np.array(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [3.0, -2.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        scores = cosine_matrix(q, M)
        self.assertEqual(scores.shape, (4,))
        for i in range(M.shape[0]):
            self.assertAlmostEqual(float(scores[i]), cosine(q, M[i]), places=6)

    def test_cosine_matrix_zero_query(self) -> None:
        q = np.zeros(3, dtype=np.float32)
        M = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(cosine_matrix(q, M), np.zeros(1, dtype=np.float32))


class StubEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_text_is_deterministic(self) -> None:
        embedder = StubEmbedder()
        first = await embedder.embed("the cat sat on the mat")
        second = await embedder.embed("the cat sat on the mat")
        self.assertEqual(first, second)

    async def test_different_texts_differ(self) -> None:
        embedder = StubEmbedder()
        a = await embedder.embed("alpha")
        b = await embedder.embed("beta")
        self.assertNotEqual(a, b)

    async def test_vectors_are_unit_norm_and_full_dim(self) -> None:
        embedder = StubEmbedder()
        vec = await embedder.embed("hello world")
        self.assertEqual(len(vec), EMBED_DIM)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(vec))), 1.0, places=5)

    async def test_batch_matches_single(self) -> None:
        embedder = StubEmbedder()
        batch = await embedder.embed_batch(["one", "two"])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0], await embedder.embed("one"))
        self.assertEqual(batch[1], await embedder.embed("two"))


class ClientEmbedTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_returns_none_on_endpoint_error(self) -> None:
        client = LLMClient(base_url="http://localhost:1", api_key="x")
        client._async = _FakeOpenAI(_FakeEmbeddingsAPI(error=RuntimeError("down")))
        self.assertIsNone(await client.embed("hello"))
        self.assertIsNone(await client.embed_batch(["hello", "world"]))

    async def test_embed_returns_vectors_on_success(self) -> None:
        client = LLMClient(base_url="http://localhost:1", api_key="x")
        api = _FakeEmbeddingsAPI()
        client._async = _FakeOpenAI(api)
        vec = await client.embed("hello")
        self.assertEqual(vec, [1.0, 0.0, 0.0])
        batch = await client.embed_batch(["a", "b"])
        self.assertEqual(batch, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(api.calls[0]["model"], "bge-m3")

    async def test_embed_uses_explicit_model(self) -> None:
        client = LLMClient(base_url="http://localhost:1", api_key="x")
        api = _FakeEmbeddingsAPI()
        client._async = _FakeOpenAI(api)
        await client.embed("hello", model="custom-embed")
        self.assertEqual(api.calls[0]["model"], "custom-embed")

    def test_sync_embed_returns_none_on_endpoint_error(self) -> None:
        client = LLMClient(base_url="http://localhost:1", api_key="x")
        client._sync = _FakeOpenAI(_FakeSyncEmbeddingsAPI(error=RuntimeError("down")))
        self.assertIsNone(client.embed_sync("hello"))
        self.assertIsNone(client.embed_batch_sync(["hello"]))

    def test_sync_embed_returns_vectors_on_success(self) -> None:
        client = LLMClient(base_url="http://localhost:1", api_key="x")
        client._sync = _FakeOpenAI(_FakeSyncEmbeddingsAPI())
        self.assertEqual(client.embed_sync("hello"), [1.0, 0.0, 0.0])


class LibertAIEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_wraps_client_with_model(self) -> None:
        from conscio.memory.embeddings import LibertAIEmbedder

        client = LLMClient(base_url="http://localhost:1", api_key="x")
        api = _FakeEmbeddingsAPI()
        client._async = _FakeOpenAI(api)
        embedder = LibertAIEmbedder(client)
        vec = await embedder.embed("hello")
        self.assertEqual(vec, [1.0, 0.0, 0.0])
        self.assertEqual(api.calls[0]["model"], "bge-m3")
        batch = await embedder.embed_batch(["a", "b"])
        self.assertEqual(len(batch), 2)

    async def test_degrades_to_none_when_endpoint_down(self) -> None:
        from conscio.memory.embeddings import LibertAIEmbedder

        client = LLMClient(base_url="http://localhost:1", api_key="x")
        client._async = _FakeOpenAI(_FakeEmbeddingsAPI(error=RuntimeError("down")))
        embedder = LibertAIEmbedder(client)
        self.assertIsNone(await embedder.embed("hello"))
        self.assertIsNone(await embedder.embed_batch(["a"]))


if __name__ == "__main__":
    unittest.main()
