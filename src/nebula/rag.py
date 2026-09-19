"""RAG 层：检索 -> 拼上下文 -> 生成带引用作答。

硬化点：
- 无依据必须拒答（不是编造），并把 mode 显式暴露出来，便于上层与评测区分
  「真的生成」和「降级摘录」。
- 引用编号与检索命中一一对应，不重排、不丢项。
"""
from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "你是严谨的中文知识助手。只依据【参考资料】回答，不得编造。\n"
    "要求：\n"
    "1. 答案中点明依据来源，格式为 [编号]。\n"
    "2. 资料不足以回答时，直接回复「资料中未提及该信息」，不要猜测。\n"
    "3. 回答简洁、分点，不重复资料原文。"
)


@dataclass
class RagAnswer:
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    mode: str = "rag"  # rag | abstain | degraded
    trace: dict = field(default_factory=dict)
    tok_s: float = 0.0
    model: str = ""


def _build_context(hits) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    cites: list[dict] = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] 来源：{h.source}\n{h.text}")
        cites.append(
            {
                "index": i,
                "id": h.id,
                "source": h.source,
                "score": round(h.score, 6),
                "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4),
                "dense_rank": h.dense_rank,
                "sparse_rank": h.sparse_rank,
                "snippet": h.text[:120],
            }
        )
    return "\n\n".join(blocks), cites


class RagEngine:
    def __init__(self, retriever, llm, top_k: int = 5, min_hits: int = 1, max_tokens: int = 1024) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_hits = min_hits
        self.max_tokens = max_tokens

    async def _generate(self, messages: list[dict]) -> tuple[object, str]:
        """调用 LLM，并对「思维链吃满预算、正文为空」做一次触发式重试。

        Qwen3 类模型无论怎么要求都会先推理，思维链与正文共用同一份 num_predict 预算。
        预算不足时会出现 content="" 而 thinking 非空 —— 这**不是**服务不可用，
        因此不能直接判为降级，必须放大预算重试一次。
        """
        out = await self.llm.achat(messages, temperature=0.2, num_predict=self.max_tokens)
        if not out.content and out.thinking:
            bigger = messages + [
                {"role": "user", "content": "请直接给出最终答案正文，不要重复推理过程，控制在 200 字内。"}
            ]
            out = await self.llm.achat(
                bigger, temperature=0.2, num_predict=int(self.max_tokens * 1.75)
            )
            return out, "thinking_truncated_retry"
        return out, "ok"

    async def aanswer(self, query: str, top_k: int | None = None, use_rerank: bool | None = None) -> RagAnswer:
        k = top_k or self.top_k
        res = self.retriever.search(query, top_k=k, use_rerank=use_rerank)
        if len(res.hits) < self.min_hits:
            return RagAnswer(
                query=query,
                answer="资料中未提及该信息（知识库为空或未命中）。",
                citations=[],
                mode="abstain",
                trace=res.trace,
            )
        context, cites = _build_context(res.hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{query}"},
        ]
        out, gen_note = await self._generate(messages)
        if not out.content:
            # 降级：给出资料摘录而非编造；并如实标注失败原因，避免"看起来像答了"
            reason = f"（原因：{out.mode}）" if str(out.mode).startswith("degraded") else ""
            excerpt = "\n".join(f"[{c['index']}] {c['snippet']}" for c in cites[:3])
            return RagAnswer(
                query=query,
                answer=f"生成模型不可用{reason}，以下为检索到的原文摘录：\n{excerpt}",
                citations=cites,
                mode="degraded",
                trace={**res.trace, "gen": gen_note, "llm_mode": out.mode},
            )
        return RagAnswer(
            query=query,
            answer=out.content.strip(),
            citations=cites,
            mode="rag",
            trace={**res.trace, "gen": gen_note, "thinking_len": len(out.thinking)},
            tok_s=round(out.tok_s, 2),
            model=out.model,
        )

    def answer(self, query: str, top_k: int | None = None, use_rerank: bool | None = None) -> RagAnswer:
        import asyncio

        try:
            return asyncio.run(self.aanswer(query, top_k, use_rerank))
        except RuntimeError:
            # 已在事件循环中（如 FastAPI 同步端点） → 用独立线程跑
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                return ex.submit(
                    lambda: asyncio.run(self.aanswer(query, top_k, use_rerank))
                ).result()
