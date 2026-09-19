# ADR-006: 模型与依赖来源全部走国内可达镜像

## Status
Accepted (2026-09-20)

## Background
本机网络实测：

| 目标 | 结果 |
|---|---|
| `huggingface.co` | 超时（不可达） |
| `github.com`（网页 HEAD 经代理） | 502 |
| `github.com`（git HTTPS / 443） | **可达**（`git ls-remote` 正常返回） |
| `api.github.com`（经代理） | 200 |
| `www.modelscope.cn` | 200 |
| `pypi.org` | 200 |

这意味着：模型不能从 HF 拉，预编译二进制不能从 GitHub Releases 拉（不可靠），
但 PyPI 与 ModelScope 完全可用。

## Decision
1. **Python 依赖**：只从 PyPI 装，且**禁止任何需要本机编译的包**
   （已因此弃用 LanceDB 与 jieba，见 ADR-001/ADR-002）
2. **LLM 与嵌入**：走本机 Ollama（模型来自其 registry），已在位的模型直接复用
3. **重排器**：从 **ModelScope** 拉 `BAAI/bge-reranker-base` 的 `onnx/model.onnx` +
   `tokenizer.json`，三步重试 + `.part` 断点残留清理
4. **不做**：不从 HF `snapshot_download`，不从 GitHub Releases 下载 llama.cpp 预编译包
5. 下载逻辑集中在 `models_download.py`，任何源变更只改一处

## Consequences
- 正面：一键引导在国内网络下可跑通；无代理依赖
- 负面：重排器只有 ModelScope 一个源，源挂则降级为 `NullReranker`（中性分数，链路不中断）
- 负面：无法使用需要 GitHub Releases 的工具链（如 llama.cpp 预编译二进制）

## Related ADRs
ADR-001、ADR-002
