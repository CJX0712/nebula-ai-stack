"""命令行入口：serve / ingest / ask / search / agent / bench / eval / verify。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config, PROJECT_ROOT


def _get_system(args):
    from .runtime import build_system

    cfg = Config.from_env()
    return build_system(cfg, offline=getattr(args, "offline", False))


def cmd_serve(args):
    import uvicorn

    from .gateway import create_app
    from .runtime import build_system

    cfg = Config.from_env()
    system = build_system(cfg, offline=args.offline)
    app = create_app(system)
    print(f"[nebula] 配置：{json.dumps(cfg.summary(), ensure_ascii=False)}")
    print(f"[nebula] 模块：{json.dumps(system.health(), ensure_ascii=False)}")
    port = args.port or cfg.api_port
    uvicorn.run(app, host=args.host or cfg.api_host, port=port, log_level="info")


def cmd_ingest(args):
    sys_ = _get_system(args)
    total = 0
    for p in args.paths:
        path = Path(p)
        files = [path] if path.is_file() else sorted(path.rglob("*.md"))
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            n = sys_.retriever.add_document(text, source=str(f.relative_to(PROJECT_ROOT)))
            total += n
            print(f"  入库 {f.name}: {n} 块")
    print(f"完成：新增 {total} 块，库中合计 {sys_.store.count()}")


def cmd_ask(args):
    sys_ = _get_system(args)
    t0 = time.perf_counter()
    res = sys_.rag.answer(args.query)
    dt = time.perf_counter() - t0
    print(f"\n【模式】{res.mode}  【耗时】{dt:.2f}s  【吞吐】{res.tok_s} tok/s")
    print(f"\n【回答】\n{res.answer}")
    if res.citations:
        print("\n【引用】")
        for c in res.citations:
            print(f"  [{c['index']}] {c['source']}  score={c['score']}  稠密#{c['dense_rank']} 稀疏#{c['sparse_rank']}")
    if args.trace:
        print(f"\n【链路】{json.dumps(res.trace, ensure_ascii=False, indent=2)}")


def cmd_search(args):
    sys_ = _get_system(args)
    res = sys_.retriever.search(args.query, top_k=args.top_k)
    print(f"命中 {len(res.hits)} 条（重排{'启用' if res.trace.get('rerank_used') else '未启用'}）")
    for i, h in enumerate(res.hits, 1):
        print(f"  {i}. [{h.source}] score={h.score:.4f} 稠密#{h.dense_rank} 稀疏#{h.sparse_rank} "
              f"重排={h.rerank_score}\n     {h.text[:100]}")


def cmd_agent(args):
    sys_ = _get_system(args)
    res = sys_.agent.run(args.goal)
    print(f"\n【状态】{res.status}  迭代 {res.iterations} 步")
    for st in res.steps:
        line = f"  步骤{st.step}: {st.action}"
        if st.tool:
            line += f" -> {st.tool}({json.dumps(st.args, ensure_ascii=False)})"
        print(line)
        if st.observation:
            print(f"      观察: {st.observation[:200]}")
    print(f"\n【结论】{res.answer}")


def cmd_bench(args):
    from .inference import OllamaLLM, bench_once

    cfg = Config.from_env()
    llm = OllamaLLM(cfg.ollama_base, args.model or cfg.llm_model, threads=args.threads or cfg.llm_threads,
                    ctx=cfg.llm_ctx)
    print(f"模型={llm.model} 线程={llm.threads}")
    for i in range(args.rounds):
        r = bench_once(llm, n=2)
        print(f"  第{i + 1}轮: {r['tok_s']:.2f} tok/s  (tokens={r['tokens']})")


def cmd_eval(args):
    from .eval_harness import run_eval

    report = run_eval(Path(args.data), offline=args.offline, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_verify(args):
    """一键自检：依赖可导入 + 单元不变量 + 端到端链路（可选真实 LLM）。"""
    import subprocess

    steps = [
        ("单元测试", [sys.executable, "-m", "pytest", "-q", "tests"]),
    ]
    if not args.skip_e2e:
        steps.append(("端到端冒烟", [sys.executable, "-m", "nebula.smoke"]))
    ok = True
    for name, cmd in steps:
        print(f"\n=== {name} ===")
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        print(r.stdout[-4000:])
        if r.returncode != 0:
            print(r.stderr[-2000:])
            ok = False
    print("\n结论：" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


def cmd_download(args):
    from .models_download import main as dl

    return dl(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nebula", description=f"Nebula AI Stack v{__version__} by 晨星")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sub.add_parser(name, **kw)
        s.set_defaults(fn=fn)
        return s

    s = add("serve", cmd_serve, help="启动 API 网关")
    s.add_argument("--host", default=""); s.add_argument("--port", type=int, default=0)
    s.add_argument("--offline", action="store_true")

    s = add("ingest", cmd_ingest, help="入库文档/目录")
    s.add_argument("paths", nargs="+"); s.add_argument("--offline", action="store_true")

    s = add("ask", cmd_ask, help="RAG 问答")
    s.add_argument("query"); s.add_argument("--trace", action="store_true")
    s.add_argument("--offline", action="store_true")

    s = add("search", cmd_search, help="混合检索")
    s.add_argument("query"); s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--offline", action="store_true")

    s = add("agent", cmd_agent, help="工具调用 Agent")
    s.add_argument("goal"); s.add_argument("--offline", action="store_true")

    s = add("bench", cmd_bench, help="推理吞吐基准")
    s.add_argument("--model", default=""); s.add_argument("--threads", type=int, default=0)
    s.add_argument("--rounds", type=int, default=3)

    s = add("eval", cmd_eval, help="检索效果评测")
    s.add_argument("--data", default=str(PROJECT_ROOT / "data" / "eval" / "set.json"))
    s.add_argument("--top-k", type=int, default=5); s.add_argument("--offline", action="store_true")

    s = add("verify", cmd_verify, help="一键自检")
    s.add_argument("--skip-e2e", action="store_true")

    s = add("download-models", cmd_download, help="下载模型（Ollama + ModelScope 重排器）")
    s.add_argument("--skip-ollama", action="store_true")

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
