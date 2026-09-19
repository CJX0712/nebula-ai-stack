"""模型下载：LLM/嵌入走 Ollama，重排器走 ModelScope 镜像（国内直连）。

不依赖 HuggingFace（本机 HF 不可达）与 GitHub Releases（需代理）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .config import PROJECT_ROOT

RERANK_REPO = "BAAI/bge-reranker-base"
RERANK_FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
]
MS_FILE_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"


def _log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def pull_ollama(models: list[str]) -> bool:
    ok = True
    for m in models:
        _log(f"ollama pull {m}")
        try:
            r = subprocess.run(["ollama", "pull", m], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=3600)
            print(r.stdout[-1500:])
            if r.returncode != 0:
                print(r.stderr[-800:])
                ok = False
        except FileNotFoundError:
            _log("未找到 ollama 命令，请先安装 Ollama（https://ollama.com）")
            return False
        except Exception as e:
            _log(f"拉取失败：{e}")
            ok = False
    return ok


def _http_get(url: str, dest: Path, timeout: int = 120) -> bool:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            with c.stream("GET", url) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                t0 = time.perf_counter()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(65536):
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (4 << 20) < 65536:
                            pct = done * 100 // total
                            spd = done / max(1e-6, time.perf_counter() - t0) / 1e6
                            print(f"\r   {dest.name} {pct}%  {spd:.1f} MB/s", end="", flush=True)
        print()
        tmp.replace(dest)
        return True
    except Exception as e:
        _log(f"下载失败 {url} -> {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def fetch_reranker(target_dir: Path) -> bool:
    """从 ModelScope 拉取 bge-reranker-base 的 ONNX 与 tokenizer。"""
    target_dir = Path(target_dir)
    all_ok = True
    for rel in RERANK_FILES:
        dest = target_dir / rel
        if dest.exists() and dest.stat().st_size > 0:
            _log(f"已存在，跳过：{dest}")
            continue
        url = MS_FILE_API.format(repo=RERANK_REPO, path=rel)
        _log(f"下载 {RERANK_REPO}:{rel}")
        ok = False
        for attempt in range(3):
            if _http_get(url, dest):
                ok = True
                break
            time.sleep(2)
        if not ok:
            all_ok = False
    # 顶层也放一份 tokenizer.json，方便直接读取
    src = target_dir / "tokenizer.json"
    if src.exists():
        (target_dir / "tokenizer.json").exists() and None
    return all_ok


def main(args) -> int:
    ok = True
    if not args.skip_ollama:
        ok = pull_ollama(["qwen3:4b", "bge-m3:latest"]) and ok
    from .config import Config

    cfg = Config.from_env()
    _log(f"重排器目标目录：{cfg.rerank_model_dir}")
    ok = fetch_reranker(cfg.rerank_model_dir) and ok
    print("\n结论：" + ("全部就绪" if ok else "部分失败（系统会自动降级，不影响主链路）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(type("A", (), {"skip_ollama": "--skip-ollama" in sys.argv})()))
