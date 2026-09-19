"""依赖装配（唯一一处把各模块拼起来的地方）。

依赖方向严格单向：
    gateway -> rag / agent -> retrieval -> {embedding, vectorstore, reranker} -> inference
任何模块都不反向依赖 gateway；任何模块都可单独构造与测试。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import ReactAgent, ToolRegistry, calculator_tool, file_read_tool, make_rag_tool, now_tool
from .config import Config
from .embedding import HashingEmbedder, build_embedder
from .inference import NullLLM, OllamaLLM
from .rag import RagEngine
from .reranker import NullReranker, build_reranker
from .retrieval import HybridRetriever
from .vectorstore import MemoryVectorStore, build_store


@dataclass
class System:
    cfg: Config
    llm: object
    embedder: object
    reranker: object
    store: object
    retriever: HybridRetriever
    rag: RagEngine
    tools: ToolRegistry
    agent: ReactAgent
    offline: bool = False
    notes: list[str] = field(default_factory=list)

    def health(self) -> dict:
        return {
            "llm": getattr(self.llm, "name", type(self.llm).__name__),
            "llm_model": getattr(self.llm, "model", ""),
            "llm_threads": getattr(self.llm, "threads", None),
            "embedder": getattr(self.embedder, "name", type(self.embedder).__name__),
            "embed_mode": getattr(self.embedder, "last_mode", "n/a"),
            "reranker": getattr(self.reranker, "name", type(self.reranker).__name__),
            "reranker_enabled": bool(getattr(self.reranker, "enabled", False)),
            "store": getattr(self.store, "name", type(self.store).__name__),
            "store_degraded": bool(getattr(self.store, "degraded", False)),
            "chunks": self.store.count(),
            "tools": self.tools.names(),
            "offline": self.offline,
        }


def build_system(cfg: Config, offline: bool = False) -> System:
    notes: list[str] = []
    if offline:
        llm = NullLLM(cfg.llm_threads)
        embedder = HashingEmbedder()
        reranker = NullReranker()
        store = MemoryVectorStore()
        notes.append("离线模式：未接入真实模型，链路与不变量仍可完整验证")
    else:
        llm = OllamaLLM(cfg.ollama_base, cfg.llm_model, cfg.llm_threads, cfg.llm_ctx, cfg.llm_timeout_s)
        embedder = build_embedder(cfg, offline=False)
        reranker = build_reranker(cfg.rerank_model_dir, cfg.rerank_enabled)
        if isinstance(reranker, NullReranker):
            notes.append(f"重排器未启用（模型目录：{cfg.rerank_model_dir}）")
        store = build_store(cfg.vector_dir)
        if getattr(store, "degraded", False):
            notes.append("向量库降级为内存实现")

    retriever = HybridRetriever(
        embedder=embedder,
        store=store,
        reranker=reranker,
        rrf_k=cfg.rrf_k,
        bm25_weight=cfg.bm25_weight,
        dense_weight=cfg.dense_weight,
        rerank_weight=cfg.rerank_weight,
        rerank_top_n=cfg.rerank_top_n,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    rag = RagEngine(retriever, llm, top_k=cfg.default_top_k, max_tokens=cfg.llm_max_tokens)

    tools = ToolRegistry()
    tools.register(make_rag_tool(rag))
    tools.register(calculator_tool())
    tools.register(now_tool())
    tools.register(file_read_tool(cfg.data_dir))
    agent = ReactAgent(llm=llm, tools=tools, max_steps=cfg.agent_max_steps,
                       max_tokens=max(256, cfg.llm_max_tokens // 2))

    return System(
        cfg=cfg, llm=llm, embedder=embedder, reranker=reranker, store=store,
        retriever=retriever, rag=rag, tools=tools, agent=agent, offline=offline, notes=notes,
    )
