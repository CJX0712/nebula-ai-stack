from nebula.agent import ReactAgent, ToolRegistry, _safe_eval, calculator_tool
from nebula.inference import LLMResult


class _ScriptedLLM:
    """按脚本返回响应，用于确定性测试 Agent 循环。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def achat(self, messages, **kw):
        self.calls += 1
        return LLMResult(content=self.script.pop(0), model="script", eval_tokens=5)


def test_safe_eval_whitelist():
    assert abs(_safe_eval("sqrt(16) + 2*3") - 10.0) < 1e-9
    for bad in ("__import__('os').system('dir')", "open('x')", "1;2"):
        try:
            _safe_eval(bad)
        except Exception:
            pass
        else:
            raise AssertionError(f"危险表达式未被拦截：{bad}")


def test_agent_runs_tool_then_final():
    tools = ToolRegistry()
    tools.register(calculator_tool())
    llm = _ScriptedLLM([
        '{"action":"tool","tool":"calculator","args":{"expr":"2+3"}}',
        '{"action":"final","answer":"结果是 5"}',
    ])
    agent = ReactAgent(llm=llm, tools=tools, max_steps=4)
    res = agent.run("2+3 等于多少")
    assert res.status == "ok"
    assert res.answer == "结果是 5"
    assert res.steps[0].tool == "calculator"
    assert "5" in res.steps[0].observation


def test_agent_parse_error_is_bounded():
    tools = ToolRegistry()
    tools.register(calculator_tool())
    llm = _ScriptedLLM(["这句话里没有 JSON"])
    agent = ReactAgent(llm=llm, tools=tools, max_steps=3)
    res = agent.run("随便问")
    assert res.status == "parse_error"
    assert res.iterations == 1


def test_agent_max_steps_guard():
    tools = ToolRegistry()
    tools.register(calculator_tool())
    llm = _ScriptedLLM(['{"action":"tool","tool":"calculator","args":{"expr":"1+1"}}'] * 5)
    agent = ReactAgent(llm=llm, tools=tools, max_steps=3)
    res = agent.run("一直算")
    assert res.status == "max_steps" and res.iterations == 3


def test_unknown_tool_reports_error():
    tools = ToolRegistry()
    tools.register(calculator_tool())
    assert "未知工具" in tools.run("nope", {})
