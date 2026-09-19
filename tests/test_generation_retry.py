"""生成层护栏：思维链吃满输出预算时必须重试，而不是误判为"模型不可用"。

真实故障：Qwen3 类模型无论怎么要求都会先推理，思维链与正文共用同一份 num_predict
预算。预算被思维链吃满时 content="" 而 thinking 非空 —— 若把这种情况当作服务不可用，
RAG 会错误地返回 degraded（摘录）而不是真正的答案。
"""
from nebula.agent import ReactAgent, ToolRegistry, calculator_tool
from nebula.embedding import HashingEmbedder
from nebula.inference import LLMResult
from nebula.rag import RagEngine
from nebula.retrieval import HybridRetriever
from nebula.vectorstore import MemoryVectorStore


class _TruncatingLLM:
    """第一次调用只吐思维链（预算被吃满），第二次才吐正文。"""

    def __init__(self, second_content: str) -> None:
        self.second = second_content
        self.calls: list[int] = []

    async def achat(self, messages, **kw):
        self.calls.append(int(kw.get("num_predict", 0)))
        if len(self.calls) == 1:
            return LLMResult(content="", thinking="我需要先确认……" * 40, model="fake",
                             eval_tokens=kw.get("num_predict", 0), eval_seconds=1.0)
        return LLMResult(content=self.second, thinking="", model="fake", eval_tokens=20,
                         eval_seconds=2.0)


def _engine(llm):
    r = HybridRetriever(embedder=HashingEmbedder(dim=256), store=MemoryVectorStore())
    r.add_document("线程数应锁在 2 到 4 之间，因为瓶颈是内存带宽而不是算力。", "a.md")
    return RagEngine(r, llm, top_k=2, max_tokens=64)


def test_rag_retries_when_thinking_eats_budget():
    import asyncio

    llm = _TruncatingLLM("线程数应锁在 2 到 4 之间。[1]")
    res = asyncio.run(_engine(llm).aanswer("线程数应该设多少"))
    assert len(llm.calls) == 2, "必须在正文为空时重试一次"
    assert llm.calls[1] > llm.calls[0], "重试必须放大预算"
    assert res.mode == "rag"
    assert res.trace["gen"] == "thinking_truncated_retry"
    assert "2 到 4" in res.answer


def test_agent_retries_when_thinking_eats_budget():
    tools = ToolRegistry()
    tools.register(calculator_tool())
    llm = _TruncatingLLM('{"action":"final","answer":"42"}')
    res = ReactAgent(llm=llm, tools=tools, max_steps=3, max_tokens=32).run("终极问题")
    assert len(llm.calls) >= 2
    assert res.answer == "42"


def test_http_timeout_scales_with_token_budget():
    """回归护栏：预算给足了但 HTTP 先超时，会表现为"正文永远为空"这种最隐蔽的故障。

    CPU 上 4B 模型实测 2–5 tok/s，1024 个输出 token 需要 200–500s。
    超时必须随预算放大，否则调大 num_predict 反而更糟。
    """
    from nebula.inference import OllamaLLM

    llm = OllamaLLM(threads=4, timeout=300.0)
    assert llm._timeout_for(None) == 300.0
    assert llm._timeout_for(64) == 300.0, "小预算不应低于基线"
    assert llm._timeout_for(1024) >= 1024 / OllamaLLM.MIN_TOK_S
    assert llm._timeout_for(10_000_000) == OllamaLLM.MAX_TIMEOUT_S, "必须有上限"


def test_rag_abstains_without_hits():
    r = HybridRetriever(embedder=HashingEmbedder(), store=MemoryVectorStore())
    res = RagEngine(r, _TruncatingLLM("x")).answer("任何问题")
    assert res.mode == "abstain"
    assert res.citations == []
