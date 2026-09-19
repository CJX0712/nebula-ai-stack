import pytest

from nebula.fusion import rrf


def test_rrf_orders_by_rank_consensus():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    res = rrf([a, b], k=60)
    assert res.order[0] in ("x", "y")
    assert set(res.order) == {"x", "y", "z", "w"}
    # 两路都排在前的应优于只在单路出现的
    assert res.order.index("x") < res.order.index("z")


def test_rrf_is_scale_free():
    """只吃名次，不吃原始分数：同序不同分数必须给出同序结果。"""
    res1 = rrf([["a", "b"], ["a", "b"]], k=60)
    res2 = rrf([["a", "b"], ["a", "b"]], k=60, weights=[1.0, 1.0])
    assert res1.order == res2.order == ["a", "b"]


def test_adversarial_invariant_holds_arithmetically():
    """核心不变量：w<1 时，一阶段冠军在完全倒序的重排下仍是最终第一。

    冠军分 = 1/(k+1) + w/(k+n)
    亚军分 = 1/(k+n) + w/(k+1)
    差     = (1-w)*[1/(k+1) - 1/(k+n)] > 0
    对任意 k>0、n>1、0<=w<1 恒成立 —— 所以这条测试不会偶发失败。
    """
    k = 60
    for n in (3, 5, 12, 40):
        for w in (0.0, 0.25, 0.5, 0.75, 0.999):
            fused = [f"d{i}" for i in range(n)]
            reranked = list(reversed(fused))  # 对抗性：完全倒序
            res = rrf([fused, reranked], k=k, weights=[1.0, w], names=["fused", "rerank"])
            assert res.order[0] == fused[0], f"n={n} w={w} 冠军被换掉了"


def test_equal_weight_can_tie_which_is_why_w_must_be_lt_1():
    """w=1 时置换型对抗会平局（差值为 0） —— 这是必须 w<1 的另一个理由。"""
    fused = ["a", "b", "c"]
    res = rrf([fused, list(reversed(fused))], k=60, weights=[1.0, 1.0])
    assert res.scores["a"] == pytest.approx(res.scores["c"])


def test_rrf_guards():
    with pytest.raises(ValueError):
        rrf([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)
    assert rrf([]).order == []


def test_contributions_are_traceable():
    res = rrf([["a", "b"], ["b", "a"]], k=60, names=["sparse", "dense"])
    assert set(res.contributions["a"]) == {"sparse", "dense"}
    assert res.contributions["a"]["sparse"] > 0
