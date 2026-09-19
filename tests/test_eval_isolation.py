"""评测隔离护栏。

真实故障：`run_eval` 复用生产向量库，且只在 `store.count() == 0` 时灌语料。
冒烟跑完后库里已有 3 条无关分块 → 24 篇黄金语料一条都没入库 →
24 条查询全命中那 3 条干扰文档 → **在线评测 0/24**，而程序不报任何错。

这类"指标全错但流程正常"的缺陷只能靠隔离性断言发现。
"""
import json

import pytest

from nebula.eval_harness import build_isolated_retriever, run_eval
from nebula.runtime import build_system
from nebula.config import Config

MINI = {
    "name": "mini",
    "corpus": [
        {"source": "a.md", "text": "线程数应锁在 2 到 4 之间，瓶颈是内存带宽。"},
        {"source": "b.md", "text": "重排权重必须小于 1，否则会换掉一阶段冠军。"},
        {"source": "c.md", "text": "Windows 保留端口会导致绑定失败。"},
    ],
    "queries": [
        {"query": "线程数应该设多少", "expect_source": "a.md"},
        {"query": "重排权重为什么不能设成一", "expect_source": "b.md"},
        {"query": "Windows 端口绑定失败", "expect_source": "c.md"},
    ],
}


@pytest.fixture()
def mini_set(tmp_path):
    p = tmp_path / "mini.json"
    p.write_text(json.dumps(MINI, ensure_ascii=False), encoding="utf-8")
    return p


def test_eval_is_isolated_from_production_index(mini_set, tmp_path, monkeypatch):
    """即使生产索引里塞满干扰文档，评测结果也必须不变。"""
    monkeypatch.setenv("NEBULA_VECTOR_DIR", str(tmp_path / "prod"))
    first = run_eval(mini_set, offline=True)

    # 往生产索引里灌入干扰文档
    cfg = Config.from_env()
    cfg.vector_dir = tmp_path / "prod"
    s = build_system(cfg, offline=True)
    s.retriever.add_document("完全无关的干扰内容" * 20, "distractor.md")

    second = run_eval(mini_set, offline=True)
    assert first["top1"] == second["top1"], "评测结果被生产索引状态污染"
    assert first["top3"] == second["top3"]
    assert first["corpus_chunks"] == second["corpus_chunks"]


def test_eval_always_ingests_its_own_corpus(mini_set, tmp_path, monkeypatch):
    monkeypatch.setenv("NEBULA_VECTOR_DIR", str(tmp_path / "prod2"))
    report = run_eval(mini_set, offline=True)
    assert report["corpus_docs"] == 3
    assert report["corpus_chunks"] >= 3
    assert report["cases"] == 3
    assert report["top3"] == "3/3"
    assert report["adversarial_invariant"]["champion_kept"] is True


def test_isolated_retriever_is_fresh_each_call(mini_set, tmp_path, monkeypatch):
    monkeypatch.setenv("NEBULA_VECTOR_DIR", str(tmp_path / "prod3"))
    cfg = Config.from_env()
    system = build_system(cfg, offline=True)
    a = build_isolated_retriever(system, MINI["corpus"])
    b = build_isolated_retriever(system, MINI["corpus"])
    assert a.store is not b.store
    assert a.store.count() == b.store.count()
