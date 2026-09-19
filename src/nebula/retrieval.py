"""混合检索：BM25(稀疏) + Dense(稠密) -> RRF 一阶段融合 -> 重排二次融合。

这是全栈最关键的一处设计。重排**不接管**最终排序，只作为第三路信号以 w<1 参与
二次 RRF，理由见 fusion.rrf 的数学不变量（对抗性重排测试会钉住这一点）。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .chunking import split_text
from .fusion import rrf
from .sparse import BM25Index


@dataclass
class Hit:
    id: str
    text: str
    source: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    trace: dict = field(default_factory=dict)

    @property
    def top_id(self) -> str | None:
        return self.hits[0].id if self.hits else None


class HybridRetriever:
    """依赖全部注入：embedder / store / reranker 都可替换，便于单测与降级。"""

    def __init__(
        self,
        embedder,
        store,
        reranker=None,
        *,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        rerank_weight: float = 0.5,
        rerank_top_n: int = 20,
        chunk_size: int = 420,
        chunk_overlap: int = 60,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rerank_weight = rerank_weight
        self.rerank_top_n = rerank_top_n
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25 = BM25Index()
        self._texts: dict[str, str] = {}
        self._sources: dict[str, str] = {}

    # ---------- 写入 ----------
    def add_document(self, text: str, source: str = "") -> int:
        chunks = split_text(text, self.chunk_size, self.chunk_overlap, source=source)
        return self.add_chunks([(c.text, source, c.index) for c in chunks])

    def add_documents(self, docs: list[tuple[str, str]]) -> int:
        total = 0
        for text, source in docs:
            total += self.add_document(text, source)
        return total

    def add_chunks(self, chunks: list[tuple[str, str, int]]) -> int:
        if not chunks:
            return 0
        from .vectorstore import VectorRecord

        texts = [c[0] for c in chunks]
        vectors = self.embedder.embed(texts)
        records: list[VectorRecord] = []
        for (c_text, source, idx), vec in zip(chunks, vectors):
            # Qdrant 点 ID 必须是合法 UUID 字符串或整数
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{idx}::{c_text[:64]}"))
            self._texts[doc_id] = c_text
            self._sources[doc_id] = source
            self.bm25.add(doc_id, c_text)
            records.append(
                VectorRecord(id=doc_id, text=c_text, source=source, chunk_index=idx, vector=vec)
            )
        self.store.upsert(records)
        return len(records)

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 5, use_rerank: bool | None = None) -> RetrievalResult:
        if self.store.count() == 0:
            return RetrievalResult(query=query, hits=[], trace={"reason": "empty_index"})

        q_vec = self.embedder.embed([query])[0]
        dense_hits = self.store.search(q_vec, k=max(top_k * 4, 20))
        dense_order = [h.id for h in dense_hits]
        dense_scores = {h.id: h.score for h in dense_hits}

        sparse_order = self.bm25.ranking(query, limit=max(top_k * 4, 20))
        sparse_scores = self.bm25.scores(query)

        stage1 = rrf(
            [sparse_order, dense_order],
            k=self.rrf_k,
            weights=[self.bm25_weight, self.dense_weight],
            names=["bm25", "dense"],
        )

        use_rr = use_rerank if use_rerank is not None else bool(getattr(self.reranker, "enabled", False))
        trace: dict = {
            "bm25_top": sparse_order[:5],
            "dense_top": dense_order[:5],
            "fused_top": stage1.order[:5],
            "rerank_used": False,
        }

        final_order = stage1.order
        rerank_scores: dict[str, float] = {}
        if use_rr and self.reranker is not None and len(stage1.order) > 1:
            shortlist = stage1.order[: max(self.rerank_top_n, top_k)]
            scores = self.reranker.score(query, [self._texts.get(i, "") for i in shortlist])
            rerank_scores = dict(zip(shortlist, scores))
            reranked = sorted(shortlist, key=lambda i: (-rerank_scores[i], i))
            stage2 = rrf(
                [stage1.order, reranked],
                k=self.rrf_k,
                weights=[1.0, self.rerank_weight],
                names=["fused", "rerank"],
            )
            final_order = stage2.order
            trace["rerank_used"] = True
            trace["rerank_top"] = reranked[:5]
            trace["final_top"] = stage2.order[:5]
            contributions = stage2.contributions
            scores_map = stage2.scores
        else:
            contributions = stage1.contributions
            scores_map = stage1.scores
            trace["final_top"] = final_order[:5]

        rank_sparse = {d: i + 1 for i, d in enumerate(sparse_order)}
        rank_dense = {d: i + 1 for i, d in enumerate(dense_order)}
        hits: list[Hit] = []
        for doc_id in final_order[:top_k]:
            hits.append(
                Hit(
                    id=doc_id,
                    text=self._texts.get(doc_id, ""),
                    source=self._sources.get(doc_id, ""),
                    score=scores_map.get(doc_id, 0.0),
                    dense_rank=rank_dense.get(doc_id),
                    sparse_rank=rank_sparse.get(doc_id),
                    rerank_score=rerank_scores.get(doc_id),
                    contributions=contributions.get(doc_id, {}),
                )
            )
        trace["dense_scores_max"] = round(max(dense_scores.values(), default=0.0), 4)
        trace["sparse_max"] = round(max(sparse_scores.values(), default=0.0), 4)
        return RetrievalResult(query=query, hits=hits, trace=trace)

    def clear(self) -> None:
        self.bm25.clear()
        self._texts.clear()
        self._sources.clear()
        self.store.clear()
