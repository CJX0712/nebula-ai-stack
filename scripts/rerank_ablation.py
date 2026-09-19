"""重排消融实验：把「重排不得接管排序」这条结论在本项目自己的黄金集上复现。

    uv run python scripts/rerank_ablation.py
    uv run python scripts/rerank_ablation.py --offline     # 用哈希嵌入，秒级完成

对同一次召回（BM25 + 稠密 + 一阶段 RRF）比较五种最终排序策略：
  1) 纯 RRF（完全不用重排）
  2) 重排直接接管（最常见的错误写法：final = reranked）
  3)~5) 重排作为第三路信号，以 w = 0.25 / 0.5 / 0.75 二次融合

重排打分只计算一次（短名单相同），因此五种策略完全可比，且总耗时约等于一次评测。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nebula.config import Config  # noqa: E402
from nebula.eval_harness import build_isolated_retriever, _load  # noqa: E402
from nebula.fusion import rrf  # noqa: E402
from nebula.runtime import build_system  # noqa: E402


def rank_hit_ids(order: list[str], source_of: dict[str, str], expect: str) -> tuple[int, int]:
    top1 = 1 if order and source_of.get(order[0]) == expect else 0
    top3 = 1 if expect in [source_of.get(i) for i in order[:3]] else 0
    return top1, top3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "eval" / "set.json"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    data = _load(Path(args.data))
    cfg = Config.from_env()
    system = build_system(cfg, offline=args.offline)
    retriever = build_isolated_retriever(system, data["corpus"])
    if not retriever.reranker.enabled:
        print("重排器未启用（模型缺失或 --offline），本实验退化为对照基线。")

    strategies = {
        "纯 RRF（不用重排）": None,
        "重排直接接管（错误写法）": "takeover",
        "二次融合 w=0.25": 0.25,
        "二次融合 w=0.50": 0.50,
        "二次融合 w=0.75": 0.75,
        "二次融合 w=1.00（等权）": 1.00,
    }
    acc = {k: {"top1": 0, "top3": 0} for k in strategies}
    n = len(data["queries"])
    rerank_ms_total = 0.0

    for case in data["queries"]:
        q, expect = case["query"], case["expect_source"]
        base = retriever.search(q, top_k=args.top_k, use_rerank=False)
        order = [h.id for h in base.hits]
        source_of = {h.id: h.source for h in base.hits}
        if not order:
            continue

        # 重排打一次分，五种策略共用
        import time

        t0 = time.perf_counter()
        scores = retriever.reranker.score(q, [retriever._texts.get(i, "") for i in order])
        rerank_ms_total += (time.perf_counter() - t0) * 1000
        reranked = [i for i, _ in sorted(zip(order, scores), key=lambda p: (-p[1], p[0]))]

        t1, t3 = rank_hit_ids(order, source_of, expect)
        acc["纯 RRF（不用重排）"]["top1"] += t1
        acc["纯 RRF（不用重排）"]["top3"] += t3

        t1, t3 = rank_hit_ids(reranked, source_of, expect)
        acc["重排直接接管（错误写法）"]["top1"] += t1
        acc["重排直接接管（错误写法）"]["top3"] += t3

        for name, w in strategies.items():
            if not isinstance(w, float):
                continue
            fused = rrf([order, reranked], k=cfg.rrf_k, weights=[1.0, w],
                        names=["fused", "rerank"]).order
            t1, t3 = rank_hit_ids(fused, source_of, expect)
            acc[name]["top1"] += t1
            acc[name]["top3"] += t3

    print(f"\n模式={'offline' if args.offline else 'online'}  "
          f"嵌入={type(system.embedder).__name__}  重排={type(system.reranker).__name__}  "
          f"语料={len(data['corpus'])} 篇  查询={n} 条\n")
    print(f"{'最终排序策略':<26}{'top-1':>10}{'top-3':>10}")
    print("-" * 46)
    for name, v in acc.items():
        print(f"{name:<26}{v['top1']:>7}/{n}{v['top3']:>7}/{n}")

    base_top1 = acc["纯 RRF（不用重排）"]["top1"]
    take_top1 = acc["重排直接接管（错误写法）"]["top1"]
    best = max((v["top1"], k) for k, v in acc.items())
    print(f"\n重排直接接管 vs 纯 RRF：{take_top1} vs {base_top1} "
          f"→ {'净损害（本实验复现了该缺陷）' if take_top1 < base_top1 else '未复现净损害'}")
    print(f"最优策略：{best[1]}（top-1 {best[0]}/{n}）")
    print(f"重排总耗时：{rerank_ms_total / 1000:.1f}s（{n} 条查询，单条 {rerank_ms_total / max(1, n):.0f}ms）")

    out = {
        "mode": "offline" if args.offline else "online",
        "embedder": type(system.embedder).__name__,
        "reranker": type(system.reranker).__name__,
        "questions": n, "corpus": len(data["corpus"]),
        "results": {k: {**v, "n": n} for k, v in acc.items()},
        "rerank_ms_per_query": round(rerank_ms_total / max(1, n), 1),
    }
    (ROOT / "data" / "rerank_ablation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("原始数据已写入 data/rerank_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
