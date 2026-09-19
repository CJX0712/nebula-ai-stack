"""全局配置：全部走环境变量覆盖，默认值对 CPU 小机友好。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def auto_threads(logical: int | None = None) -> int:
    """小量化模型是内存带宽瓶颈，不是算力瓶颈。

    实测（8C/16T，Q4 量化）：4 线程 32.5 tok/s，16 线程 8.2 tok/s。
    默认 os.cpu_count()-1=15 恰好落在最差一格。线程必须锁在 2..4。
    """
    n = logical if logical is not None else (os.cpu_count() or 4)
    return max(2, min(4, n // 4))


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


@dataclass
class Config:
    # ---- L0 底座 ----
    ollama_base: str = field(default_factory=lambda: _env("NEBULA_OLLAMA_BASE", "http://127.0.0.1:11434"))
    llm_model: str = field(default_factory=lambda: _env("NEBULA_LLM_MODEL", "qwen3:4b"))
    embed_model: str = field(default_factory=lambda: _env("NEBULA_EMBED_MODEL", "bge-m3:latest"))
    llm_threads: int = field(default_factory=lambda: _env_int("NEBULA_LLM_THREADS", auto_threads()))
    llm_ctx: int = field(default_factory=lambda: _env_int("NEBULA_LLM_CTX", 4096))
    # Qwen3 这类模型"一定会先想一遍"，思维链与正文共用同一份输出预算。
    # 预算给小了会出现 thinking 很长、content 为空的情况，因此留足并允许触发式重试。
    llm_max_tokens: int = field(default_factory=lambda: _env_int("NEBULA_LLM_MAX_TOKENS", 1024))
    # CPU 上 4B 模型解码约 2–5 tok/s。超时**必须**随输出预算放大，
    # 否则 num_predict=1024 需要 200s+ 会直接撞上超时，表现为"正文永远为空"。
    # 实际超时由 OllamaLLM._timeout_for() 动态计算：max(此值, num_predict / 2 tok/s)。
    llm_timeout_s: float = field(default_factory=lambda: _env_float("NEBULA_LLM_TIMEOUT", 300.0))
    request_timeout_s: float = field(default_factory=lambda: _env_float("NEBULA_HTTP_TIMEOUT", 30.0))

    # ---- 重排 ----
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("NEBULA_RERANK_ENABLED", True))
    rerank_weight: float = field(default_factory=lambda: _env_float("NEBULA_RERANK_WEIGHT", 0.5))
    rerank_top_n: int = field(default_factory=lambda: _env_int("NEBULA_RERANK_TOP_N", 20))
    rerank_model_dir: Path = field(
        default_factory=lambda: Path(_env("NEBULA_RERANK_DIR", str(PROJECT_ROOT / "models" / "bge-reranker-base")))
    )

    # ---- 检索 ----
    rrf_k: int = field(default_factory=lambda: _env_int("NEBULA_RRF_K", 60))
    bm25_weight: float = field(default_factory=lambda: _env_float("NEBULA_BM25_WEIGHT", 1.0))
    dense_weight: float = field(default_factory=lambda: _env_float("NEBULA_DENSE_WEIGHT", 1.0))
    default_top_k: int = field(default_factory=lambda: _env_int("NEBULA_TOP_K", 5))

    # ---- 分块 ----
    chunk_size: int = field(default_factory=lambda: _env_int("NEBULA_CHUNK_SIZE", 420))
    chunk_overlap: int = field(default_factory=lambda: _env_int("NEBULA_CHUNK_OVERLAP", 60))

    # ---- 存储 / 服务 ----
    vector_dir: Path = field(
        default_factory=lambda: Path(_env("NEBULA_VECTOR_DIR", str(PROJECT_ROOT / "data" / "qdrant")))
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("NEBULA_DATA_DIR", str(PROJECT_ROOT / "data"))))
    # 8765：Windows WinNAT/Hyper-V 常保留 8000-8123，绑定会 WinError 10013
    api_host: str = field(default_factory=lambda: _env("NEBULA_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("NEBULA_API_PORT", 8765))

    # ---- Agent ----
    agent_max_steps: int = field(default_factory=lambda: _env_int("NEBULA_AGENT_MAX_STEPS", 6))

    @classmethod
    def from_env(cls) -> "Config":
        return cls()

    def summary(self) -> dict:
        return {
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
            "llm_threads": self.llm_threads,
            "rerank_enabled": self.rerank_enabled,
            "rerank_weight": self.rerank_weight,
            "rrf_k": self.rrf_k,
            "llm_max_tokens": self.llm_max_tokens,
            "vector_dir": str(self.vector_dir),
            "api_port": self.api_port,
        }
