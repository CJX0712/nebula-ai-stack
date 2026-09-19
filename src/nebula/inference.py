"""L0 推理层：Ollama HTTP 适配（OpenAI 兼容生态下最省内存的 CPU 方案）。

关键工程约束（来自实测）：
1. num_thread 必须锁 2..4。默认 cpu_count-1 会让小量化模型慢 2-4 倍（内存带宽瓶颈）。
2. **绝不发送 think:false**。它不省 token，只会把思维链倒进 content 字段。
   推理自然走 message.thinking，前端折叠展示。
3. 服务不可用时降级为“可重试”，绝不一次性锁死（服务可能只是还没起来）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .config import auto_threads


@dataclass
class LLMResult:
    content: str
    model: str = ""
    thinking: str = ""
    eval_tokens: int = 0
    eval_seconds: float = 0.0
    mode: str = "llm"  # llm | degraded

    @property
    def tok_s(self) -> float:
        return self.eval_tokens / self.eval_seconds if self.eval_seconds > 0 else 0.0

    def usage(self) -> dict:
        return {"completion_tokens": self.eval_tokens, "tok_s": round(self.tok_s, 2)}


class NullLLM:
    """离线/降级实现：不编造内容，显式声明不可用。"""

    name = "null"
    model = "none"

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or auto_threads()

    async def achat(self, messages, **kw) -> LLMResult:
        return LLMResult(content="", model="none", mode="degraded")

    async def astream(self, messages, **kw) -> AsyncIterator[str]:
        yield ""

    async def ahealth(self) -> bool:
        return False


class OllamaLLM:
    name = "ollama"

    def __init__(
        self,
        base: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b",
        threads: int | None = None,
        ctx: int = 4096,
        timeout: float = 180.0,
    ) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.threads = auto_threads() if threads is None else max(1, min(4, threads))
        self.ctx = ctx
        self.timeout = timeout

    # ---------- 内部 ----------
    #: 保守下界：CPU 上量化 4B 模型实测解码速度不低于约 2 tok/s。
    #: 超时按预算线性放大，避免"预算给足了但 HTTP 先超时"这种最隐蔽的空正文故障。
    MIN_TOK_S = 2.0
    MAX_TIMEOUT_S = 1800.0

    def _timeout_for(self, num_predict: int | None) -> float:
        if not num_predict or num_predict <= 0:
            return self.timeout
        need = float(num_predict) / self.MIN_TOK_S
        return max(self.timeout, min(self.MAX_TIMEOUT_S, need))

    def _payload(self, messages, stream: bool, **kw) -> dict:
        opts = {
            "num_thread": self.threads,
            "num_ctx": kw.get("num_ctx") or self.ctx,
        }
        if kw.get("num_predict"):
            opts["num_predict"] = int(kw["num_predict"])
        if kw.get("temperature") is not None:
            opts["temperature"] = float(kw["temperature"])
        if kw.get("seed") is not None:
            opts["seed"] = int(kw["seed"])
        # 注意：此处刻意不支持 think 字段
        return {"model": self.model, "messages": messages, "stream": stream, "options": opts}

    @staticmethod
    def _parse(data: dict, model: str) -> LLMResult:
        msg = data.get("message") or {}
        ec = int(data.get("eval_count") or 0)
        ed = float(data.get("eval_duration") or 0) / 1e9 if data.get("eval_duration") else 0.0
        return LLMResult(
            content=msg.get("content", "") or "",
            thinking=msg.get("thinking", "") or "",
            model=model,
            eval_tokens=ec,
            eval_seconds=ed,
        )

    # ---------- 同步 ----------
    def chat(self, messages, **kw) -> LLMResult:
        try:
            with httpx.Client(timeout=self._timeout_for(kw.get("num_predict"))) as c:
                r = c.post(f"{self.base}/api/chat", json=self._payload(messages, False, **kw))
                r.raise_for_status()
                return self._parse(r.json(), self.model)
        except Exception as e:
            return LLMResult(content="", model=self.model, mode=f"degraded: {type(e).__name__}: {e}")

    # ---------- 异步 ----------
    async def achat(self, messages, **kw) -> LLMResult:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_for(kw.get("num_predict"))) as c:
                r = await c.post(f"{self.base}/api/chat", json=self._payload(messages, False, **kw))
                r.raise_for_status()
                return self._parse(r.json(), self.model)
        except Exception as e:
            return LLMResult(content="", model=self.model, mode=f"degraded: {type(e).__name__}: {e}")

    async def astream(self, messages, **kw) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_for(kw.get("num_predict"))) as c:
                async with c.stream(
                    "POST", f"{self.base}/api/chat", json=self._payload(messages, True, **kw)
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = (obj.get("message") or {}).get("content") or ""
                        if piece:
                            yield piece
                        if obj.get("done"):
                            break
        except Exception:
            return

    async def ahealth(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.base}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=8.0) as c:
                r = c.get(f"{self.base}/api/tags")
                r.raise_for_status()
                return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []


def bench_once(llm: OllamaLLM, n: int = 1, prompt: str | None = None) -> dict:
    """固定提示词、固定 seed，跑 n 轮取最好成绩。用 eval_count/eval_duration 算 tok/s。"""
    p = prompt or (
        "请分五点说明为什么在 CPU 上运行量化小模型时，线程数不是越多越好，"
        "每点不少于三十个字，并给出可验证的测量方法。"
    )
    best = {"tok_s": 0.0, "tokens": 0}
    # warm-up：排除首次缺页与模型加载
    llm.chat([{"role": "user", "content": "你好"}], num_predict=8)
    for _ in range(max(1, n)):
        t0 = time.perf_counter()
        r = llm.chat([{"role": "user", "content": p}], num_predict=192, seed=42, temperature=0.0)
        dt = time.perf_counter() - t0
        tps = r.tok_s or (r.eval_tokens / dt if dt > 0 else 0.0)
        if tps > best["tok_s"]:
            best = {"tok_s": tps, "tokens": r.eval_tokens, "wall_s": round(dt, 2)}
    return best
