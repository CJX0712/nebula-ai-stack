"""L0 嵌入层：bge-m3（多语言，中文优先）走 Ollama；离线用确定性哈希嵌入兜底。

哈希嵌入不是玩具：它保证「干净环境无模型也能跑通全链路与全部测试」，
从而让 tests 与 CI 完全不依赖外部服务。生产路径永远是 bge-m3。
"""
from __future__ import annotations

import hashlib
import math

import httpx

from .sparse import tokenize


class HashingEmbedder:
    """确定性特征哈希（CJK 单字 + 双字，latin 词），L2 归一化。零依赖、可复现。"""

    name = "hashing-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokenize(text)
        grams = list(toks)
        for a, b in zip(toks, toks[1:]):
            grams.append(a + b)
        for g in grams:
            h = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm > 0 else v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t or "") for t in texts]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class OllamaEmbedder:
    """bge-m3 via Ollama /api/embed；失败时退回哈希嵌入，保证链路不中断。"""

    name = "ollama-bge-m3"

    def __init__(
        self,
        base: str = "http://127.0.0.1:11434",
        model: str = "bge-m3:latest",
        timeout: float = 60.0,
        fallback: bool = True,
    ) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._fallback = HashingEmbedder() if fallback else None
        self._dim: int | None = 1024  # bge-m3
        self.last_mode: str = "unknown"

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(f"{self.base}/api/embed", json={"model": self.model, "input": texts})
                r.raise_for_status()
                vecs = r.json().get("embeddings") or []
                if vecs and len(vecs) == len(texts):
                    self._dim = len(vecs[0])
                    self.last_mode = "ollama"
                    return [[float(x) for x in v] for v in vecs]
        except Exception:
            pass
        self.last_mode = "fallback-hashing"
        if self._fallback is None:
            raise RuntimeError("嵌入服务不可用且未启用兜底")
        vecs = self._fallback.embed(texts)
        self._dim = len(vecs[0]) if vecs else self._dim
        return vecs

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{self.base}/api/embed", json={"model": self.model, "input": texts})
                r.raise_for_status()
                vecs = r.json().get("embeddings") or []
                if vecs and len(vecs) == len(texts):
                    self._dim = len(vecs[0])
                    self.last_mode = "ollama"
                    return [[float(x) for x in v] for v in vecs]
        except Exception:
            pass
        self.last_mode = "fallback-hashing"
        if self._fallback is None:
            raise RuntimeError("嵌入服务不可用且未启用兜底")
        return self._fallback.embed(texts)

    async def ahealth(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.post(f"{self.base}/api/embed", json={"model": self.model, "input": ["ping"]})
                return r.status_code == 200
        except Exception:
            return False


def build_embedder(cfg, offline: bool = False):
    if offline:
        return HashingEmbedder()
    return OllamaEmbedder(cfg.ollama_base, cfg.embed_model, timeout=cfg.request_timeout_s)
