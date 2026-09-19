"""Nebula AI Stack - 模块化端到端 AI 系统。

设计原则：
- 单一职责：每个子模块只做一件事，可独立 import、独立测试、独立替换。
- 依赖单向：gateway -> rag/agent -> retrieval -> {embedding, vectorstore, reranker} -> inference。
- 可降级：任何外部能力（LLM / 重排器）不可用时降级为中性行为，不抛异常。
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"
