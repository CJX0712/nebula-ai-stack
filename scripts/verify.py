"""一键验证：干净环境下判定"这套系统是否真的能跑起来"的机器判据。

按顺序执行，任一步失败即整体失败：
  1) 依赖可导入      （版本锁定生效）
  2) 单元不变量      （pytest，含对抗性重排不变量）
  3) 端到端冒烟      （离线契约 + 在线真实链路，Ollama 不可达则跳过在线部分）
  4) 评测基线        （离线哈希嵌入，可复现的检索质量下限）

用法：
    uv run python scripts/verify.py
    uv run python scripts/verify.py --online     # 强制要求在线链路
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def step(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def run(cmd: list[str], env: dict | None = None) -> tuple[bool, str, int, str]:
    """返回 (是否成功, 输出尾部, 退出码, 完整 stdout)。

    为什么同时返回尾部与全文：尾部用于人眼快速定位失败原因，
    全文用于结构化解析（评测 JSON 远超尾部长度，只拿尾部会解析失败）。
    """
    import os

    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=e)
    stdout = r.stdout or ""
    tail = stdout[-1500:] + (r.stderr or "")[-800:]
    return r.returncode == 0, tail.strip(), r.returncode, stdout


def run_pytest() -> tuple[bool, str]:
    """在**本进程内**跑 pytest，用插件直接取计数，不抓 stdout。

    为什么不用子进程 + 抓 stdout（血泪教训，两次不同根因）：

    1) 早先用 `"passed" in tail` 判断，失败时打印「输出为空（疑似进程被终止）」，
       而真实输出里明明有 `ERROR: file or directory not found: tests` ——
       报告与事实不符，把排障引向错误方向。
    2) 修好措辞后又出现新的偶发形态：stdout 尾部是
       `.......(60 个点)........ [100%]`，即**测试全部跑完、进程在打印汇总行之前
       退出**（rc=1、stderr 为空）。此时任何基于 stdout 汇总行的判断都会误报失败。

    结论：**不要从人眼输出里提取机器判据**。改为进程内调用 pytest，
    通过插件按 `report.when == "call"` 精确统计 passed/failed/errors，
    并以计数（而不是退出码、也不是输出文本）作为唯一判据。

    注意：不因为「测试失败」而重试，只因为「测试没跑起来」而重试。
    """
    import os

    import pytest as _pytest

    class _Collector:
        def __init__(self) -> None:
            self.passed = 0
            self.failed = 0
            self.errors = 0
            self.skipped = 0

        def pytest_runtest_logreport(self, report) -> None:
            if report.when == "call":
                if report.passed:
                    self.passed += 1
                elif report.failed:
                    self.failed += 1
                elif report.skipped:
                    self.skipped += 1
            elif report.when == "setup" and report.failed:
                self.errors += 1

    c = _Collector()
    cwd = os.getcwd()
    os.chdir(ROOT)
    # 不使用 --basetemp：`tests/conftest.py` 已接管 `tmp_path`（改用 tempfile.mkdtemp
    # 且不做清理），pytest 的临时目录机制因此完全不被实例化 —— 也就不会有
    # 「会话收尾递归删除上百个文件 → 被环境删除守卫拒绝 → SystemExit: 1」这回事。
    # 保留 `-p no:cacheprovider` 只是为了不写 .pytest_cache（减少无谓的文件写入）。
    try:
        rc = _pytest.main(["-q", "tests", "-p", "no:cacheprovider"], plugins=[c])
    except SystemExit as e:  # noqa: BLE001
        return False, (f"pytest 以 SystemExit({e.code}) 中止（通常是会话收尾被外部拦截，"
                       f"而非测试失败）；已完成 passed={c.passed} failed={c.failed}")
    except BaseException as e:  # noqa: BLE001 - 收集阶段崩溃也要如实上报
        return False, f"pytest 进程内执行异常：{type(e).__name__}: {e}"
    finally:
        os.chdir(cwd)
    detail = (f"passed={c.passed} failed={c.failed} errors={c.errors} "
              f"skipped={c.skipped} exit={rc}")
    ok = rc == 0 and c.failed == 0 and c.errors == 0 and c.passed > 0
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="要求在线链路必须可用")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="跳过在线链路（把 Ollama 指向不可达端口，仅验证可复现的离线路径）")
    args = ap.parse_args()

    step("1/4 依赖导入")
    ok, out, rc, _ = run([sys.executable, "-c",
                          "import fastapi,uvicorn,httpx,pydantic,numpy,qdrant_client,onnxruntime,tokenizers,nebula;"
                          "print('all imports ok')"])
    record("模块可导入（版本锁定生效）", ok, out.splitlines()[-1] if out else f"rc={rc}")

    step("2/4 单元不变量")
    ok, detail = run_pytest()
    record("pytest 全绿", ok, detail)

    step("3/4 端到端冒烟" + ("（--fast：仅离线）" if args.fast else ""))
    env = {"NEBULA_OLLAMA_BASE": "http://127.0.0.1:1"} if args.fast else None
    ok, out, rc, _ = run([sys.executable, "-m", "nebula.smoke"], env=env)
    line = next((ln for ln in reversed(out.splitlines()) if ln.strip().startswith("结论")), "")
    online_ok = "在线真实链路" in out
    record("离线契约 + 在线链路", ok, line or f"rc={rc}")
    if args.online and not online_ok:
        record("在线链路（--online 强制）", False, "Ollama 不可达")

    if not args.skip_eval:
        step("4/4 评测基线（离线可复现）")
        ok, out, rc, stdout = run([sys.executable, "-m", "nebula.cli", "eval", "--offline",
                                   "--data", str(ROOT / "data" / "eval" / "set.json")])
        import json as _json

        metrics = ""
        invariant_ok = False
        try:
            report = _json.loads(stdout[stdout.index("{"):])
            metrics = (f"top1={report['top1']} top3={report['top3']} mrr={report['mrr']} "
                       f"对抗不变量={report['adversarial_invariant']['champion_kept']}")
            invariant_ok = bool(report["adversarial_invariant"]["champion_kept"])
        except Exception as e:
            metrics = f"rc={rc} 无法解析评测输出：{type(e).__name__}"
        record("离线评测可运行且不变量成立", ok and invariant_ok, metrics)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"总计 {passed}/{len(RESULTS)} 项通过")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    overall = passed == len(RESULTS)
    print("\n结论：" + ("干净环境验证全绿" if overall else "存在失败项"))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
