"""RRF（Reciprocal Rank Fusion）融合。

核心不变量：融合只使用名次，不使用原始分数，因此跨量纲（BM25 分数 / 余弦 / 重排 logits）
天然免校准。带权重时，只要 w<1，一阶段冠军在对抗性重排下必定仍为最终第一：

    冠军得分 = 1/(k+1) + w/(k+n)
    亚军得分 = 1/(k+n) + w/(k+1)
    差      = (1-w) * [1/(k+1) - 1/(k+n)] > 0     (n > 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionResult:
    order: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, dict[str, float]] = field(default_factory=dict)


def rrf(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
    names: list[str] | None = None,
) -> FusionResult:
    """把多路排序融合成一路。

    rankings: 每路是一个按相关度降序的 id 列表（允许长度不同、允许缺项）。
    weights:  每路话语权；缺省全 1.0。重排路建议 0.5。
    """
    if not rankings:
        return FusionResult(order=[])
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 长度必须与 rankings 一致")
    if names is None:
        names = [f"rank_{i}" for i in range(len(rankings))]
    if len(names) != len(rankings):
        raise ValueError("names 长度必须与 rankings 一致")
    if k <= 0:
        raise ValueError("k 必须为正")

    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    for name, ranking, w in zip(names, rankings, weights):
        for rank, doc_id in enumerate(ranking):
            delta = w / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + delta
            contributions.setdefault(doc_id, {})[name] = delta

    order = sorted(scores, key=lambda d: (-scores[d], d))
    return FusionResult(order=order, scores=scores, contributions=contributions)
