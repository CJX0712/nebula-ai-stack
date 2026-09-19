import pytest

from nebula.embedding import HashingEmbedder
from nebula.retrieval import HybridRetriever
from nebula.vectorstore import MemoryVectorStore

CORPUS = [
    ("CPU 上运行量化小模型时，推理瓶颈是内存带宽而不是算力。线程数应当锁在 2 到 4 之间，"
     "超过物理核数之后吞吐会明显下降。", "docs/threads.md"),
    ("交叉编码器重排不应直接接管最终排序。若把重排顺序当作最终顺序，一个在查询语言上"
     "力不从心的重排器会压掉正确答案。正确做法是把重排作为第三路信号参与二次融合。", "docs/rerank.md"),
    ("Windows 上 8000 到 8123 端口常被 WinNAT 与 Hyper-V 保留，绑定会失败并报 WinError 10013，"
     "默认端口应改为 8765。", "docs/windows.md"),
    ("检索增强生成要求答案附带引用编号，并在资料不足时明确拒答，避免编造。", "docs/rag.md"),
]


def _retriever(reranker=None, **kw):
    r = HybridRetriever(
        embedder=HashingEmbedder(dim=512),
        store=MemoryVectorStore(),
        reranker=reranker,
        chunk_size=300,
        chunk_overlap=40,
        **kw,
    )
    r.add_documents(CORPUS)
    return r


class _AdversarialReranker:
    """最差的排最前（完整倒序）——用来证明重排没有一票否决权。"""

    name = "adversarial"
    enabled = True

    def score(self, query, documents):
        return [float(i) for i in range(len(documents))]


class _GoodReranker:
    name = "keyword-boost"
    enabled = True

    def score(self, query, documents):
        return [float(d.count("重排")) for d in documents]


def test_ingest_and_dense_retrieval_top1():
    r = _retriever()
    assert r.store.count() >= 4
    res = r.search("CPU 线程数应该设多少", top_k=3, use_rerank=False)
    assert res.hits, "检索为空"
    assert res.hits[0].source == "docs/threads.md"


def test_trace_is_complete():
    r = _retriever()
    res = r.search("Windows 端口绑定失败", top_k=3, use_rerank=False)
    for key in ("bm25_top", "dense_top", "fused_top", "final_top", "rerank_used"):
        assert key in res.trace
    assert res.trace["rerank_used"] is False


def test_adversarial_rerank_keeps_champion():
    """核心防回归：注入完全倒序的重排器后，一阶段冠军必须仍是最终第一。

    w<1 时该性质由 RRF 的算术恒等式保证，不会偶发失败。
    """
    r = _retriever(reranker=_AdversarialReranker(), rerank_weight=0.5)
    q = "重排应该怎么使用"
    plain = r.search(q, top_k=3, use_rerank=False)
    hostile = r.search(q, top_k=3, use_rerank=True)
    assert plain.hits and hostile.hits
    assert hostile.trace["rerank_used"] is True
    assert hostile.hits[0].id == plain.hits[0].id, "重排接管了排序 —— 不变量被破坏"


def test_rerank_can_refine_without_overriding_champion():
    """好的重排器能够改善次序，但同样不能把一阶段冠军直接踢掉（w<1 的代价/收益平衡）。"""
    r = _retriever(reranker=_GoodReranker(), rerank_weight=0.5)
    res = r.search("重排权重与融合", top_k=4, use_rerank=True)
    assert res.hits[0].source == "docs/rerank.md"
    assert res.hits[0].rerank_score is not None


def test_rerank_disabled_path_still_works():
    r = _retriever(reranker=_AdversarialReranker())
    res = r.search("RAG 引用与拒答", top_k=2, use_rerank=False)
    assert res.hits and res.hits[0].rerank_score is None


def test_empty_index_is_safe():
    r = HybridRetriever(embedder=HashingEmbedder(), store=MemoryVectorStore())
    res = r.search("任意问题", top_k=3)
    assert res.hits == []
    assert res.trace.get("reason") == "empty_index"


def test_null_reranker_degrades_neutrally():
    from nebula.reranker import NullReranker

    rr = NullReranker()
    assert rr.enabled is False
    assert rr.score("q", ["a", "b"]) == [0.0, 0.0]


def test_rerank_shortlist_never_loses_candidates():
    """短名单长度必须 >= top_k，否则重排路径会静默丢候选。"""
    r = _retriever(reranker=_GoodReranker(), rerank_weight=0.5, rerank_top_n=1)
    res = r.search("端口 线程 重排", top_k=3, use_rerank=True)
    assert len(res.hits) == 3
