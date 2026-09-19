from nebula.chunking import split_text
from nebula.config import Config, auto_threads
from nebula.sparse import BM25Index, tokenize


def test_auto_threads_is_locked_2_to_4():
    """回归护栏：不许有人把线程数"优化"回 cpu_count-1。"""
    for logical in (1, 2, 4, 8, 16, 32, 64):
        t = auto_threads(logical)
        assert 2 <= t <= 4, f"logical={logical} -> {t}"
    assert auto_threads(16) == 4
    assert auto_threads(8) == 2


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("NEBULA_LLM_THREADS", "3")
    monkeypatch.setenv("NEBULA_RERANK_WEIGHT", "0.25")
    monkeypatch.setenv("NEBULA_API_PORT", "8899")
    cfg = Config.from_env()
    assert cfg.llm_threads == 3
    assert cfg.rerank_weight == 0.25
    assert cfg.api_port == 8899


def test_config_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("NEBULA_LLM_THREADS", "not-a-number")
    assert Config.from_env().llm_threads == auto_threads()


def test_tokenize_mixes_cjk_and_latin():
    toks = tokenize("重排 rerank 权重 0.5")
    assert "重排" in toks
    assert "rerank" in toks
    assert all(t == t.lower() for t in toks if t.isascii())
    assert tokenize("") == []


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.add("d1", "在 CPU 上运行量化模型时线程数应锁在 2 到 4 之间")
    idx.add("d2", "重排器不应直接接管最终排序，应以 0.5 权重二次融合")
    idx.add("d3", "Windows 保留端口会导致绑定失败")
    order = idx.ranking("线程数应该设多少")
    assert order[0] == "d1"
    assert idx.scores("完全无关的词组")["d1"] == 0.0


def test_bm25_empty_index_is_safe():
    idx = BM25Index()
    assert idx.ranking("任意") == []
    assert idx.scores("任意") == {}


def test_chunking_splits_and_overlaps():
    text = "\n\n".join([f"第{i}段" + "内容" * 100 for i in range(6)])
    chunks = split_text(text, size=200, overlap=40, source="t.md")
    assert len(chunks) >= 5
    assert all(len(c.text) <= 260 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "t.md" for c in chunks)


def test_chunking_handles_empty_and_tiny():
    assert split_text("") == []
    assert split_text("   ") == []
    one = split_text("短句", size=100, overlap=10)
    assert len(one) == 1 and one[0].text == "短句"
