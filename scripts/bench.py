"""线程数吞吐基准：把「线程锁 2-4」这条工程结论在**本机**重新测一遍。

    uv run python scripts/bench.py                 # 默认扫描 2/4/6/8/12/16
    uv run python scripts/bench.py --threads 2,4,8

刻意绕过正式客户端的线程钳制（clamp），否则测不出"越开越慢"。
测量口径：eval_count / eval_duration（不含模型加载），每档跑两轮取最好。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMPT = (
    "请分五点说明为什么在 CPU 上运行量化小模型时，线程数不是越多越好，"
    "每点不少于三十个字，并给出可验证的测量方法。"
)


def measure(base: str, model: str, threads: int, rounds: int = 2, num_predict: int = 160) -> dict:
    body = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_thread": threads, "num_predict": num_predict, "seed": 42, "temperature": 0.0},
        "keep_alive": "5m",
    }
    best = {"threads": threads, "tok_s": 0.0, "tokens": 0}
    with httpx.Client(timeout=600.0) as c:
        # warm-up：排除首次缺页
        c.post(f"{base}/api/generate", json={**body, "prompt": "你好", "options": {**body["options"], "num_predict": 8}})
        for _ in range(rounds):
            r = c.post(f"{base}/api/generate", json=body)
            r.raise_for_status()
            d = r.json()
            ec = int(d.get("eval_count") or 0)
            ed = float(d.get("eval_duration") or 0) / 1e9
            tps = ec / ed if ed > 0 else 0.0
            if tps > best["tok_s"]:
                best = {"threads": threads, "tok_s": round(tps, 2), "tokens": ec}
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("NEBULA_OLLAMA_BASE", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default=os.environ.get("NEBULA_LLM_MODEL", "qwen3:4b"))
    ap.add_argument("--threads", default="2,4,6,8,12,16")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    try:
        httpx.get(f"{args.base}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Ollama 不可达（{args.base}）：{e}")
        return 1

    print(f"模型={args.model}  提示词固定 seed=42  num_predict=160  每档 {args.rounds} 轮取最好\n")
    rows = []
    for t in [int(x) for x in args.threads.split(",") if x.strip()]:
        r = measure(args.base, args.model, t, args.rounds)
        rows.append(r)
        print(f"  num_thread={t:>2}  ->  {r['tok_s']:>6.2f} tok/s  (tokens={r['tokens']})", flush=True)

    valid = [r for r in rows if r["tokens"] > 40]
    if valid:
        best = max(valid, key=lambda r: r["tok_s"])
        worst = min(valid, key=lambda r: r["tok_s"])
        print(f"\n最快：{best['threads']} 线程 {best['tok_s']} tok/s")
        print(f"最慢：{worst['threads']} 线程 {worst['tok_s']} tok/s")
        if worst["tok_s"] > 0:
            print(f"差距：{best['tok_s'] / worst['tok_s']:.2f}x")
        print(f"结论：{'与本项目工程结论一致（2-4 线程最优）' if best['threads'] <= 4 else '本机最优线程数与默认结论不同，请据实调整 NEBULA_LLM_THREADS'}")
    else:
        print("\n警告：有效 token 数不足，样本可能被截断，结论不可信。")
    (ROOT / "data" / "bench.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "bench.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始数据已写入 data/bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
