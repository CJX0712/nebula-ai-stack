"""ReAct 风格工具调用 Agent：单文件、无外部编排框架。

协议（要求模型输出一个 JSON 对象，取文本中最后一个合法 JSON）：
    {"action": "tool", "tool": "rag_search", "args": {"query": "..."}}
    {"action": "final", "answer": "..."}
解析失败 -> 记 trace 并终止；超过 max_steps -> 强制收口。
"""
from __future__ import annotations

import ast
import json
import math
import operator as op
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

_JSON_RE = re.compile(r"\{[^{}]*\{[^{}]*\}\s*\}|\{[^{}]*\}", re.S)


@dataclass
class Tool:
    name: str
    description: str
    args: dict  # {arg_name: "说明"}
    fn: Callable[[dict], str]

    def signature(self) -> str:
        a = ", ".join(f'{k}: {v}' for k, v in self.args.items())
        return f'- {self.name}({a}) - {self.description}'


@dataclass
class Step:
    step: int
    thought: str = ""
    action: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class AgentResult:
    goal: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    iterations: int = 0
    status: str = "ok"  # ok | max_steps | parse_error | degraded


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def prompt(self) -> str:
        return "\n".join(t.signature() for t in self._tools.values())

    def run(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未知工具 {name}。可用：{', '.join(self.names())}"
        try:
            return str(tool.fn(args or {}))
        except Exception as e:
            return f"工具执行失败：{e}"


# ---------------- 内置工具 ----------------
def make_rag_tool(rag_engine) -> Tool:
    def _run(args: dict) -> str:
        q = args.get("query", "")
        if not q:
            return "错误：缺少 query"
        res = rag_engine.answer(q) if hasattr(rag_engine, "answer") else None
        if res is None:
            import asyncio

            res = asyncio.run(rag_engine.aanswer(q))
        cites = "；".join(f"[{c['index']}]{c['source']}" for c in res.citations[:3])
        return f"答案：{res.answer}\n引用：{cites or '无'}（mode={res.mode}）"

    return Tool(name="rag_search", description="在本机知识库中检索并回答问题", args={"query": "检索问题"}, fn=_run)


def _safe_eval(expr: str) -> float:
    """白名单 AST 求值，杜绝 eval 注入。"""
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
    }

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in {"sqrt", "log", "exp", "sin", "cos", "abs", "round"}:
                return getattr(math, fname)(*[visit(a) for a in node.args])
        raise ValueError(f"不允许的表达式：{ast.dump(node)[:80]}")

    return float(visit(ast.parse(expr, mode="eval")))


def calculator_tool() -> Tool:
    def _run(args: dict) -> str:
        expr = str(args.get("expr", ""))
        try:
            return f"{expr} = {_safe_eval(expr)}"
        except Exception as e:
            return f"计算失败：{e}"

    return Tool(name="calculator", description="计算数学表达式，如 sqrt(2)*3", args={"expr": "表达式"}, fn=_run)


def now_tool() -> Tool:
    def _run(args: dict) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Tool(name="now", description="获取当前本机时间", args={}, fn=_run)


def file_read_tool(root: Path) -> Tool:
    def _run(args: dict) -> str:
        rel = str(args.get("path", ""))
        p = (Path(root) / rel).resolve()
        base = Path(root).resolve()
        if not str(p).startswith(str(base)):
            return "错误：路径越界"
        if not p.exists():
            return f"错误：文件不存在 {rel}"
        try:
            return p.read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception as e:
            return f"读取失败：{e}"

    return Tool(name="read_file", description="读取项目 data 目录下的文本文件", args={"path": "相对路径"}, fn=_run)


# ---------------- Agent ----------------
class ReactAgent:
    def __init__(self, llm, tools: ToolRegistry, max_steps: int = 6, max_tokens: int = 512) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    async def _think(self, messages: list[dict]) -> str:
        """取一轮模型输出。

        Qwen3 类模型会先推理，思维链与正文共用输出预算；预算被思维链吃满时
        content 为空而 thinking 非空 —— 此时放大预算重试一次，而不是判失败。
        """
        out = await self.llm.achat(messages, temperature=0.1, num_predict=self.max_tokens)
        if not out.content and out.thinking:
            out = await self.llm.achat(
                messages + [{"role": "user", "content": "只输出那一个 JSON 对象，不要解释。"}],
                temperature=0.1,
                num_predict=int(self.max_tokens * 1.75),
            )
        return out.content

    @staticmethod
    def _parse(text: str) -> dict | None:
        cands = _JSON_RE.findall(text or "")
        for c in reversed(cands):
            try:
                obj = json.loads(c)
                if isinstance(obj, dict) and "action" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    def _system(self) -> str:
        return (
            "你是一个使用工具解决问题的助手。每一轮必须只输出一个 JSON 对象，不要输出其他文字。\n"
            f"可用工具：\n{self.tools.prompt()}\n"
            '格式：{"action":"tool","tool":"<名称>","args":{...}} 或 {"action":"final","answer":"<最终回答>"}'
        )

    async def arun(self, goal: str) -> AgentResult:
        messages = [{"role": "system", "content": self._system()}, {"role": "user", "content": goal}]
        steps: list[Step] = []
        for i in range(1, self.max_steps + 1):
            try:
                raw = await self._think(messages)
            except Exception as e:
                return AgentResult(goal=goal, answer=f"LLM 不可用：{e}", steps=steps, iterations=i, status="degraded")
            if not raw:
                return AgentResult(goal=goal, answer="LLM 无输出", steps=steps, iterations=i, status="degraded")
            obj = self._parse(raw)
            if obj is None:
                steps.append(Step(step=i, thought=raw[:300], action="parse_error"))
                return AgentResult(
                    goal=goal, answer=raw[:500], steps=steps, iterations=i, status="parse_error"
                )
            action = str(obj.get("action", ""))
            if action == "final":
                steps.append(Step(step=i, thought=raw[:200], action="final"))
                return AgentResult(goal=goal, answer=str(obj.get("answer", "")), steps=steps, iterations=i)
            tool = obj.get("tool") or obj.get("name")
            args = obj.get("args") or {}
            obs = self.tools.run(str(tool), args if isinstance(args, dict) else {})
            steps.append(Step(step=i, thought=raw[:200], action="tool", tool=str(tool), args=args,
                              observation=obs[:1500]))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"观察结果：{obs}"})
        return AgentResult(
            goal=goal,
            answer="已达最大步数，未得到最终结论。",
            steps=steps,
            iterations=self.max_steps,
            status="max_steps",
        )

    def run(self, goal: str) -> AgentResult:
        import asyncio

        return asyncio.run(self.arun(goal))
