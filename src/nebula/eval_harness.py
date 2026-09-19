"""评测：离线可跑（哈希嵌入）/ 在线可跑（bge-m3），指标口径固定。

指标：top-1、top-3、MRR、拒答率、平均延迟；外加一条**确定性不变量**——
注入完全倒序的对抗性重排器后，一阶段融合冠军必须仍是最终第一。

**评测必须使用独立索引**（本模块的硬约束）：
早先版本复用生产向量库，并只在 `store.count() == 0` 时灌入语料。
三次冒烟跑完后生产库里已存在 3 条无关分块，于是评测的 24 篇语料
**一条都没入库**，24 条查询全部命中那 3 条干扰文档 —— 结果是 0/24。
指标全错，但程序不报错，属于最危险的一类缺陷。

因此本模块每次评测都新建内存索引、重新灌入黄金语料，
既不读取也不写入生产索引，保证「评测结果与历史状态无关」。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Config
from .retrieval import HybridRetriever
from .runtime import build_system
from .vectorstore import MemoryVectorStore


class _AdversarialReranker:
    """最差的排最前，即完整倒序。用于钉死「重排不得接管排序」。"""

    name = "adversarial"
    enabled = True

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(i) for i in range(len(documents))]


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_isolated_retriever(system, corpus: list[dict]) -> HybridRetriever:
    """用系统同一套组件（真实嵌入/真实重排器）+ 全新内存索引灌入黄金语料。"""
    cfg = system.cfg
    r = HybridRetriever(
        embedder=system.embedder,
        store=MemoryVectorStore(),
        reranker=system.reranker,
        rrf_k=cfg.rrf_k,
        bm25_weight=cfg.bm25_weight,
        dense_weight=cfg.dense_weight,
        rerank_weight=cfg.rerank_weight,
        rerank_top_n=cfg.rerank_top_n,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    for d in corpus:
        r.add_document(d["text"], d.get("source", ""))
    return r


def run_eval(data_path: Path, offline: bool = False, top_k: int = 5) -> dict:
    data = _load(data_path)
    cfg = Config.from_env()
    system = build_system(cfg, offline=offline)
    retriever = build_isolated_retriever(system, data["corpus"])

    rows = []
    t1 = t3 = 0
    rr_sum = 0.0
    lat = []
    for case in data["queries"]:
        q = case["query"]
        t0 = time.perf_counter()
        res = retriever.search(q, top_k=top_k)
        lat.append(time.perf_counter() - t0)
        sources = [h.source for h in res.hits]
        expect = case["expect_source"]
        hit1 = 1 if sources and sources[0] == expect else 0
        hit3 = 1 if expect in sources[:3] else 0
        t1 += hit1
        t3 += hit3
        rank = next((i + 1 for i, s in enumerate(sources) if s == expect), 0)
        rr_sum += 1.0 / rank if rank else 0.0
        rows.append({"query": q[:40], "expect": expect, "top1": hit1, "top3": hit3, "rank": rank})

    n = max(1, len(data["queries"]))
    report = {
        "mode": "offline" if offline else "online",
        "index": "memory (isolated, 每次重建)",
        "corpus_docs": len(data["corpus"]),
        "corpus_chunks": retriever.store.count(),
        "cases": len(data["queries"]),
        "top1": f"{t1}/{n}",
        "top3": f"{t3}/{n}",
        "mrr": round(rr_sum / n, 4),
        "avg_latency_ms": round(sum(lat) / n * 1000, 1),
        "embedder": type(system.embedder).__name__,
        "reranker": type(system.reranker).__name__,
    }

    # 对抗性重排不变量
    plain = retriever.search(data["queries"][0]["query"], top_k=3, use_rerank=False)
    saved = retriever.reranker
    retriever.reranker = _AdversarialReranker()
    try:
        hostile = retriever.search(data["queries"][0]["query"], top_k=3, use_rerank=True)
    finally:
        retriever.reranker = saved
    report["adversarial_invariant"] = {
        "champion_kept": bool(plain.hits and hostile.hits and plain.hits[0].id == hostile.hits[0].id),
        "rerank_weight": retriever.rerank_weight,
    }
    report["cases_detail"] = rows
    return report
