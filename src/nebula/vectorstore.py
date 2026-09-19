"""向量存储层：默认 Qdrant 本地模式（嵌入式、零服务、纯 Python wheel，跨平台），
不可用时自动降级为内存实现。

选型说明：LanceDB 在 Windows 上无 wheel（只有 manylinux/macOS），已弃用。
QdrantClient(path=...) 本地模式无需服务端即可持久化，是最省内存的工业级选择。

不变量：upsert N 条 -> 用其中同一条向量检索 -> 该条必在首位（往返一致）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class VectorRecord:
    id: str  # 必须为合法 UUID 字符串（Qdrant 点 ID 约束）
    text: str
    source: str = ""
    chunk_index: int = 0
    vector: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    score: float
    text: str = ""
    source: str = ""
    chunk_index: int = 0


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> int: ...
    def search(self, vector: list[float], k: int = 5) -> list[VectorHit]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


class MemoryVectorStore:
    """内存余弦检索：Qdrant 不可用时的兜底，也是单测默认实现。"""

    name = "memory"

    def __init__(self) -> None:
        self._rows: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> int:
        for r in records:
            self._rows[r.id] = r
        return len(records)

    def search(self, vector: list[float], k: int = 5) -> list[VectorHit]:
        scored = [
            VectorHit(id=r.id, score=_cosine(vector, r.vector), text=r.text, source=r.source,
                      chunk_index=r.chunk_index)
            for r in self._rows.values()
            if r.vector
        ]
        scored.sort(key=lambda h: (-h.score, h.id))
        return scored[:k]

    def count(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()


class QdrantVectorStore:
    """Qdrant 本地模式持久化存储。"""

    name = "qdrant-local"

    def __init__(self, uri: Path, collection: str = "nebula_chunks") -> None:
        self.uri = Path(uri)
        self.collection = collection
        self._client = None
        self._error: str | None = None
        self._dim: int | None = None
        self._rows: dict[str, VectorRecord] = {}
        self._connect()

    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient

            self.uri.parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.uri))
            self._error = None
        except Exception as e:
            self._client = None
            self._error = f"Qdrant 不可用，降级为内存：{e}"

    @property
    def degraded(self) -> bool:
        return self._client is None

    def _ensure_collection(self, dim: int) -> None:
        if self._client is None:
            return
        if self._dim == dim and self._has_collection():
            return
        from qdrant_client.models import Distance, VectorParams

        if not self._has_collection():
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._dim = dim

    def _has_collection(self) -> bool:
        try:
            return bool(self._client.collection_exists(self.collection))
        except Exception:
            try:
                return self.collection in [c.name for c in self._client.get_collections().collections]
            except Exception:
                return False

    def upsert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        for r in records:
            self._rows[r.id] = r
        if self._client is None:
            return len(records)
        try:
            self._ensure_collection(len(records[0].vector))
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=r.id,
                    vector=[float(x) for x in r.vector],
                    payload={"text": r.text, "source": r.source, "chunk_index": r.chunk_index},
                )
                for r in records
            ]
            self._client.upsert(collection_name=self.collection, points=points)
            return len(points)
        except Exception as e:
            self._error = f"写入失败（已镜像到内存）：{e}"
            return len(records)

    def search(self, vector: list[float], k: int = 5) -> list[VectorHit]:
        if self._client is None:
            return self._memory_search(vector, k)
        try:
            q = [float(x) for x in vector]
            # 新版客户端推荐 query_points；旧版只有 search()
            if hasattr(self._client, "query_points"):
                res = self._client.query_points(
                    collection_name=self.collection, query=q, limit=k, with_payload=True
                ).points
            else:
                res = self._client.search(
                    collection_name=self.collection, query_vector=q, limit=k, with_payload=True
                )
            hits = [
                VectorHit(
                    id=str(p.id),
                    score=float(p.score),
                    text=(p.payload or {}).get("text", ""),
                    source=(p.payload or {}).get("source", ""),
                    chunk_index=int((p.payload or {}).get("chunk_index", 0) or 0),
                )
                for p in res
            ]
            if hits:
                self._error = None  # 真实后端工作正常，清除历史告警
                return hits
        except Exception as e:
            self._error = f"检索失败，回退内存：{type(e).__name__}: {e}"
        return self._memory_search(vector, k)

    def _memory_search(self, vector: list[float], k: int) -> list[VectorHit]:
        scored = [
            VectorHit(id=r.id, score=_cosine(vector, r.vector), text=r.text, source=r.source,
                      chunk_index=r.chunk_index)
            for r in self._rows.values()
            if r.vector
        ]
        scored.sort(key=lambda h: (-h.score, h.id))
        return scored[:k]

    def count(self) -> int:
        if self._client is not None and self._has_collection():
            try:
                return int(self._client.count(collection_name=self.collection, exact=True).count)
            except Exception:
                pass
        return len(self._rows)

    def clear(self) -> None:
        """清空索引：**逐点删除**而不是删目录。

        删目录会触发文件系统级删除（在沙箱/受管环境下可能被回收站策略拦截），
        而逐点删除只动数据不动文件，成功率高且可预期。
        """
        self._rows.clear()
        if self._client is None or not self._has_collection():
            return
        try:
            while True:
                points, offset = self._client.scroll(
                    collection_name=self.collection, limit=512, with_payload=False, with_vectors=False
                )
                if not points:
                    break
                self._client.delete(
                    collection_name=self.collection, points_selector=[p.id for p in points]
                )
                if offset is None:
                    break
        except Exception:
            # 兜底：整体重建 collection（失败也不抛，保持链路可用）
            try:
                self._client.delete_collection(self.collection)
            except Exception:
                pass
        self._dim = None


def build_store(uri: Path, collection: str = "nebula_chunks") -> QdrantVectorStore:
    return QdrantVectorStore(uri, collection)
