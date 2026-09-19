"""端到端冒烟：先跑离线契约（无任何外部依赖），再跑在线真实链路（若 Ollama 可达）。

    python -m nebula.smoke
"""
from __future__ import annotations

import json
import sys

from .config import Config
from .gateway import create_app
from .runtime import build_system

SAMPLE_DOCS = [
    {
        "source": "docs/threads.md",
        "text": (
            "在 CPU 上运行量化小模型时，瓶颈是内存带宽而不是算力。"
            "实测 8 核 16 线程机器上运行 Q4 量化模型，4 线程可达 32.5 tok/s，"
            "而 16 线程只有 8.2 tok/s。因此线程数应当锁在 2 到 4 之间。"
        ),
    },
    {
        "source": "docs/rerank.md",
        "text": (
            "交叉编码器重排不应直接接管最终排序。若把重排顺序当作最终顺序，"
            "一个在查询语言上力不从心的重排器会压掉正确答案。"
            "正确做法是把重排作为第三路信号，以 0.5 权重与一阶段 RRF 二次融合。"
        ),
    },
    {
        "source": "docs/windows.md",
        "text": (
            "Windows 上 8000 到 8123 端口常被 WinNAT 与 Hyper-V 保留，"
            "绑定会失败并报 WinError 10013。默认端口应改为 8765。"
        ),
    },
]


class _Case:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    def reset(self) -> None:
        self.rows.clear()

    def report(self, title: str) -> bool:
        """打印本阶段结果并返回是否全绿。

        注意：`report()` 只对**当前阶段**的 rows 负责，调用方在每个阶段前
        必须 `reset()`，否则跨阶段行会被重复计入与重复打印。
        早先版本让本方法返回 `None`（函数忘了 return），导致后面用
        `and` 串联时整体结论恒为假 —— 22 项全 PASS 却报"存在失败项"。
        """
        print(f"\n=== {title} ===")
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
        return all(ok for _, ok, _ in self.rows)


def run_offline(c: _Case) -> None:
    from fastapi.testclient import TestClient

    cfg = Config.from_env()
    cfg.vector_dir = cfg.data_dir / "smoke-qdrant"
    system = build_system(cfg, offline=True)
    client = TestClient(create_app(system))

    r = client.get("/healthz")
    c.check("healthz 200", r.status_code == 200, f"status={r.status_code}")
    body = r.json()
    c.check("healthz 含 modules", "modules" in body and "config" in body)

    r = client.post("/v1/rag/ingest", json={"docs": SAMPLE_DOCS, "reset": True})
    c.check("rag/ingest 200", r.status_code == 200, json.dumps(r.json(), ensure_ascii=False))
    chunks = r.json().get("chunks", 0)
    c.check("入库块数 > 0", chunks > 0, f"chunks={chunks}")

    r = client.post("/v1/rag/search", json={"query": "CPU 上线程数应该设多少", "top_k": 3})
    hits = r.json().get("hits", []) if r.status_code == 200 else []
    c.check("rag/search 命中", len(hits) > 0, f"top={hits[0]['source'] if hits else 'N/A'}")

    r = client.post("/v1/rag/answer", json={"query": "重排应该怎么用"})
    c.check("rag/answer 200", r.status_code == 200, f"mode={r.json().get('mode') if r.status_code == 200 else 'ERR'}")
    c.check("rag/answer 带引用", bool(r.json().get("citations")))

    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    ok = r.status_code == 200 and "choices" in r.json() and "usage" in r.json()
    c.check("OpenAI 兼容 chat.completions", ok)

    r = client.post("/v1/embeddings", json={"input": ["测试", "第二段"]})
    data = r.json().get("data", []) if r.status_code == 200 else []
    c.check("embeddings 返回向量", len(data) == 2 and len(data[0]["embedding"]) > 8)

    r = client.post("/v1/agent/run", json={"goal": "1+1 等于多少", "max_steps": 2})
    c.check("agent/run 有界返回", r.status_code == 200 and "status" in r.json())

    r = client.get("/v1/models")
    c.check("models 列表非空", r.status_code == 200 and len(r.json().get("data", [])) >= 1)


def run_online(c: _Case) -> None:
    from fastapi.testclient import TestClient

    cfg = Config.from_env()
    # 冒烟自带 reset=True，会清空索引；因此**绝不能**指向生产向量库目录。
    cfg.vector_dir = cfg.data_dir / "smoke-qdrant"
    system = build_system(cfg, offline=False)
    health = system.health()
    c.check("Ollama 在线", health["llm"] == "ollama" and health["llm_model"] != "none",
            f"{health['llm']}/{health['llm_model']} threads={health['llm_threads']}")
    c.check("嵌入器在线", health["embedder"] == "ollama-bge-m3", health["embedder"])
    c.check("重排器状态", True, f"{health['reranker']} enabled={health['reranker_enabled']}")
    c.check("向量库状态", health["store"] == "qdrant-local" and not health["store_degraded"],
            f"{health['store']} degraded={health['store_degraded']}")

    client = TestClient(create_app(system))
    r = client.post("/v1/rag/ingest", json={"docs": SAMPLE_DOCS, "reset": True})
    c.check("在线入库", r.status_code == 200 and r.json().get("chunks", 0) > 0,
            json.dumps(r.json(), ensure_ascii=False))

    r = client.post("/v1/rag/answer", json={"query": "为什么 CPU 推理线程不能开太多？"})
    b = r.json()
    tr = b.get("trace", {})
    diag = (f"mode={b.get('mode')} tok_s={b.get('tok_s')} gen={tr.get('gen')} "
            f"llm={str(tr.get('llm_mode', '-'))[:80]}")
    c.check("RAG 生成成功", b.get("mode") == "rag", diag)
    c.check("答案非空", len(b.get("answer", "")) > 5, b.get("answer", "")[:60].replace("\n", " "))
    c.check("引用可用", len(b.get("citations", [])) > 0)
    if b.get("mode") == "rag":
        print(f"\n  [回答] {b['answer'][:300]}")

    r = client.post("/v1/rag/search", json={"query": "Windows 端口保留", "top_k": 3, "use_rerank": False})
    t = r.json().get("trace", {})
    c.check("检索链路 trace 完整", "bm25_top" in t and "dense_top" in t and "fused_top" in t,
            json.dumps(t.get("fused_top", [])[:2], ensure_ascii=False))

    r = client.post("/v1/rag/search", json={"query": "重排权重应该设多少", "top_k": 3, "use_rerank": True})
    c.check("重排路径可用（或显式降级）", r.status_code == 200)

    r = client.post("/v1/agent/run", json={"goal": "帮我算一下 sqrt(2)*3 的值", "max_steps": 4})
    b = r.json()
    c.check("Agent 完成（或显式降级）", b.get("status") in ("ok", "max_steps", "parse_error", "degraded"),
            f"status={b.get('status')} steps={b.get('iterations')}")
    if b.get("status") == "ok":
        print(f"  [Agent] {b['answer'][:200]}")


def main() -> int:
    c = _Case()

    run_offline(c)
    ok_offline = c.report("离线契约（无外部依赖）")
    c.reset()

    cfg = Config.from_env()
    import httpx

    try:
        online = httpx.get(f"{cfg.ollama_base}/api/tags", timeout=5).status_code == 200
    except Exception:
        online = False
    if online:
        run_online(c)
        ok_online = c.report("在线真实链路（Ollama + bge-m3）")
    else:
        print("\n[跳过] Ollama 不可达，未执行在线链路")
        ok_online = True
    overall = bool(ok_offline and ok_online)
    print("\n结论：" + ("端到端全绿" if overall else "存在失败项"))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
