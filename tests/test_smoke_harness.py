"""冒烟骨架自身的回归护栏。

真实故障：`_Case.report()` 忘记 `return`（实际返回 None），
调用方用 `ok = f() and report()` 串联时整体结论恒为假 ——
22 项全部 PASS 却输出"存在失败项"。这种"测试套件说谎"比测试失败更危险。
"""
from nebula.smoke import _Case


def test_report_returns_bool_not_none():
    c = _Case()
    c.check("ok", True)
    result = c.report("phase")
    assert result is True and isinstance(result, bool)


def test_report_detects_failure():
    c = _Case()
    c.check("ok", True)
    c.check("bad", False)
    assert c.report("phase") is False


def test_reset_prevents_cross_phase_contamination():
    c = _Case()
    c.check("p1", True)
    assert c.report("p1") is True
    c.reset()
    c.check("p2", False)
    # 若忘记 reset，p1 的通过项会与 p2 的失败项混在一起，阶段结论失真
    assert c.report("p2") is False
    assert len(c.rows) == 1
