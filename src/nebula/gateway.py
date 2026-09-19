"""统一网关：对外只暴露 HTTP，内部按模块分工。

- /v1/chat/completions 与 /v1/embeddings 保持 OpenAI 兼容（任何 OpenAI SDK 可直接接入）
- /v1/rag/* 与 /v1/agent/run 为原生能力（带引用、带 trace）
- /healthz 暴露每个模块的真实状态，便于运维与自检

注意：所有请求模型定义在**模块级**。若定义在 create_app 内部，配合
`from __future__ import annotations` 会让 FastAPI 无法解析前向引用，端点会静默返回 422。
"""
from __future__ import annotations

import json as _json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__


# ---------------- 请求模型（模块级，必需） ----------------
class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    model: str = ""
    messages: list[Msg]
    temperature: float = 0.2
    max_tokens: int = 512
    stream: bool = False
    seed: int | None = None


class EmbedIn(BaseModel):
    model: str = ""
    input: list[str]


class DocIn(BaseModel):
    text: str
    source: str = ""


class IngestIn(BaseModel):
    docs: list[DocIn] = Field(default_factory=list)
    reset: bool = False


class AnswerIn(BaseModel):
    query: str
    top_k: int = 0
    use_rerank: bool | None = None


class SearchIn(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool | None = None


class AgentIn(BaseModel):
    goal: str
    max_steps: int = 0


def create_app(system) -> FastAPI:
    app = FastAPI(title="Nebula AI Stack", version=__version__, description="模块化端到端 AI 系统")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.state.system = system

    # ---------------- 健康检查 ----------------
    @app.get("/healthz")
    def healthz():
        s = app.state.system
        return {
            "status": "ok",
            "version": __version__,
            "modules": s.health(),
            "config": s.cfg.summary(),
            "notes": s.notes,
        }

    @app.get("/v1/config")
    def get_config():
        return app.state.system.cfg.summary()

    # ---------------- RAG ----------------
    @app.post("/v1/rag/ingest")
    async def ingest(body: IngestIn):
        s = app.state.system
        if body.reset:
            s.retriever.clear()
        n = s.retriever.add_documents([(d.text, d.source) for d in body.docs])
        return {"chunks": n, "documents": s.store.count()}

    @app.post("/v1/rag/answer")
    async def answer(body: AnswerIn):
        s = app.state.system
        res = await s.rag.aanswer(body.query, top_k=body.top_k or None, use_rerank=body.use_rerank)
        return {
            "query": res.query,
            "answer": res.answer,
            "mode": res.mode,
            "citations": res.citations,
            "trace": res.trace,
            "tok_s": res.tok_s,
            "model": res.model,
        }

    @app.post("/v1/rag/search")
    def search(body: SearchIn):
        s = app.state.system
        res = s.retriever.search(body.query, top_k=body.top_k, use_rerank=body.use_rerank)
        return {
            "query": res.query,
            "hits": [
                {
                    "id": h.id,
                    "text": h.text,
                    "source": h.source,
                    "score": round(h.score, 6),
                    "dense_rank": h.dense_rank,
                    "sparse_rank": h.sparse_rank,
                    "rerank_score": h.rerank_score,
                    "contributions": h.contributions,
                }
                for h in res.hits
            ],
            "trace": res.trace,
        }

    @app.delete("/v1/rag/index")
    def clear_index():
        app.state.system.retriever.clear()
        return {"cleared": True}

    # ---------------- Agent ----------------
    @app.post("/v1/agent/run")
    async def agent_run(body: AgentIn):
        s = app.state.system
        if body.max_steps:
            s.agent.max_steps = body.max_steps
        res = await s.agent.arun(body.goal)
        return {
            "goal": res.goal,
            "answer": res.answer,
            "status": res.status,
            "iterations": res.iterations,
            "steps": [st.__dict__ for st in res.steps],
            "tools": s.tools.names(),
        }

    @app.get("/v1/agent/tools")
    def agent_tools():
        s = app.state.system
        return {
            "tools": [
                {"name": n, "signature": s.tools.get(n).signature()} for n in s.tools.names()
            ]
        }

    # ---------------- OpenAI 兼容 ----------------
    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatIn):
        s = app.state.system
        msgs = [{"role": m.role, "content": m.content} for m in body.messages]

        if body.stream:

            async def gen() -> AsyncIterator[bytes]:
                cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                async for piece in s.llm.astream(
                    msgs, temperature=body.temperature, num_predict=body.max_tokens
                ):
                    payload = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": s.cfg.llm_model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        out = await s.llm.achat(
            msgs, temperature=body.temperature, num_predict=body.max_tokens, seed=body.seed
        )
        prompt_tok = sum(len(m.content) for m in body.messages) // 2
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": s.cfg.llm_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": out.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tok,
                "completion_tokens": out.eval_tokens,
                "total_tokens": prompt_tok + out.eval_tokens,
            },
            "nebula": {"thinking_len": len(out.thinking), "tok_s": round(out.tok_s, 2)},
        }

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbedIn):
        s = app.state.system
        vecs = await s.embedder.aembed(body.input)
        return {
            "object": "list",
            "model": s.cfg.embed_model,
            "data": [
                {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)
            ],
            "usage": {"prompt_tokens": sum(len(t) for t in body.input) // 2, "total_tokens": 0},
        }

    @app.get("/v1/models")
    def models():
        s = app.state.system
        return {
            "object": "list",
            "data": [
                {"id": s.cfg.llm_model, "object": "model", "owned_by": "nebula"},
                {"id": s.cfg.embed_model, "object": "model", "owned_by": "nebula"},
            ],
        }

    return app
