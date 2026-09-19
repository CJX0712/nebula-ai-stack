import uuid

import pytest

from nebula.vectorstore import MemoryVectorStore, QdrantVectorStore, VectorRecord


def _rec(i: int, vec: list[float]) -> VectorRecord:
    return VectorRecord(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"t::{i}")), text=f"文本{i}",
                        source=f"src{i % 2}", chunk_index=i, vector=vec)


def test_memory_roundtrip():
    s = MemoryVectorStore()
    recs = [_rec(0, [1.0, 0.0, 0.0]), _rec(1, [0.0, 1.0, 0.0]), _rec(2, [0.9, 0.1, 0.0])]
    assert s.upsert(recs) == 3
    hits = s.search([1.0, 0.0, 0.0], k=2)
    assert hits[0].id == recs[0].id
    assert hits[0].score > hits[1].score
    s.clear()
    assert s.count() == 0


def test_qdrant_or_fallback_roundtrip(tmp_path):
    s = QdrantVectorStore(tmp_path / "qdrant")
    recs = [_rec(0, [1.0, 0.0, 0.0]), _rec(1, [0.0, 1.0, 0.0])]
    s.upsert(recs)
    hits = s.search([1.0, 0.0, 0.0], k=1)
    assert hits and hits[0].id == recs[0].id
    assert s.count() >= 1
    s.clear()
    assert s.count() == 0


def test_qdrant_backend_is_not_silently_degraded(tmp_path):
    """关键：不得"看起来用了向量库，其实每次都在走内存兜底"。

    早先 `search()` 用了已废弃的 `query_vector=` 参数，异常被吞掉后静默回退内存，
    所有断言仍然通过 —— 只有检查 `_error` 才能发现。
    """
    s = QdrantVectorStore(tmp_path / "qdrant2")
    if s.degraded:
        pytest.skip("Qdrant 不可用，本机走内存兜底")
    s.upsert([_rec(0, [0.5, 0.5, 0.0]), _rec(1, [0.0, 0.0, 1.0])])
    hits = s.search([0.5, 0.5, 0.0], k=2)
    assert hits
    assert s._error is None, f"真实后端静默降级：{s._error}"
    assert s.count() == 2
