from nebula.sparse import BM25Index, tokenize


def test_tokenize_cjk_unigram_bigram():
    toks = tokenize("混合检索")
    assert "混" in toks and "合" in toks and "混合" in toks and "合检" in toks


def test_tokenize_latin_lowercase():
    toks = tokenize("RAG Pipeline 检索")
    assert "rag" in toks and "pipeline" in toks


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.add("d1", "向量数据库用于存储嵌入向量，支持相似度检索")
    idx.add("d2", "公司三月份的差旅报销标准与流程说明")
    order = idx.ranking("向量检索")
    assert order[0] == "d1"


def test_bm25_no_hit_returns_zero():
    idx = BM25Index()
    idx.add("d1", "人工智能与机器学习")
    assert idx.scores("完全无关的量子纠缠")["d1"] == 0.0


def test_bm25_empty_index():
    idx = BM25Index()
    assert idx.scores("任意") == {}
    assert idx.ranking("任意") == []
