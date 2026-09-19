import pytest
from fastapi.testclient import TestClient

from nebula.config import Config
from nebula.gateway import create_app
from nebula.runtime import build_system

DOCS = [
    {"source": "a.md", "text": "线程数应锁在 2 到 4 之间，因为瓶颈是内存带宽。"},
    {"source": "b.md", "text": "重排应作为第三路信号以 0.5 权重参与二次 RRF 融合。"},
]


@pytest.fixture()
def client(tmp_path):
    cfg = Config.from_env()
    cfg.vector_dir = tmp_path / "q"
    cfg.data_dir = tmp_path
    system = build_system(cfg, offline=True)
    return TestClient(create_app(system))


def test_healthz_shape(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    for key in ("llm", "embedder", "reranker", "store", "tools"):
        assert key in body["modules"]
    assert "config" in body and "llm_model" in body["config"]


def test_ingest_then_search(client):
    r = client.post("/v1/rag/ingest", json={"docs": DOCS, "reset": True})
    assert r.status_code == 200 and r.json()["chunks"] > 0
    r = client.post("/v1/rag/search", json={"query": "线程数应该设多少", "top_k": 2})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits and hits[0]["source"] == "a.md"
    assert "contributions" in hits[0]


def test_openai_compatible_contract(client):
    client.post("/v1/rag/ingest", json={"docs": DOCS, "reset": True})
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3:4b", "messages": [{"role": "user", "content": "你好"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(body["usage"])
    assert "nebula" in body


def test_embeddings_contract(client):
    r = client.post("/v1/embeddings", json={"input": ["甲", "乙", "丙"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 3 and all(len(d["embedding"]) > 0 for d in data)
    assert r.json()["object"] == "list"


def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200 and len(r.json()["data"]) == 2


def test_rag_answer_offline_is_explicit_not_fabricated(client):
    client.post("/v1/rag/ingest", json={"docs": DOCS, "reset": True})
    r = client.post("/v1/rag/answer", json={"query": "线程数应该设多少"})
    assert r.status_code == 200
    body = r.json()
    # 离线（无 LLM）时必须显式标记降级，且引用仍然完整
    assert body["mode"] in ("degraded", "abstain")
    assert body["citations"]
    assert body["answer"]


def test_rag_answer_abstains_on_empty_index(client):
    client.delete("/v1/rag/index")
    r = client.post("/v1/rag/answer", json={"query": "任意问题"})
    assert r.status_code == 200
    assert r.json()["mode"] == "abstain"


def test_agent_endpoint_bounded(client):
    r = client.post("/v1/agent/run", json={"goal": "1+1", "max_steps": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "max_steps", "parse_error", "degraded")
    assert body["iterations"] <= 2
    assert "tools" in body


def test_agent_tools_listed(client):
    r = client.get("/v1/agent/tools")
    names = [t["name"] for t in r.json()["tools"]]
    assert {"rag_search", "calculator", "now", "read_file"} <= set(names)


def test_index_can_be_cleared(client):
    client.post("/v1/rag/ingest", json={"docs": DOCS, "reset": True})
    client.delete("/v1/rag/index")
    r = client.post("/v1/rag/search", json={"query": "任意"})
    assert r.json()["hits"] == []
