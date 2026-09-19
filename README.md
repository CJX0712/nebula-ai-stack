# Nebula AI Stack · 星穹 AI 栈

> 一套**模块化、可独立验证、端到端可运行**的 AI 系统开发环境。
> 不自研模型，把业界领先开源成果（Ollama / GGUF LLM / bge-m3 / bge-reranker-base / Qdrant / ONNXRuntime / FastAPI）
> 按单一职责切成 10 个模块，拼成一条可复现链路。
>
> 作者：**晨星** · 许可：MIT

---

## 1. 它解决什么问题

在一台**没有 GPU、内存 15GB、Windows 11** 的机器上，把「文档 → 检索 → 重排 → 生成带引用 → 工具调用 Agent → 评测 → 一键复现」全链路跑通，
并且每一步都有**机械可验证的不变量**，而不是"看起来能跑"。

| 目标 | 做法 |
|---|---|
| 端到端可运行 | 单条命令 `nebula serve` 起网关，OpenAI 兼容 + 原生 RAG/Agent 接口 |
| 干净环境可一键复现 | `uv sync` + `uv.lock` + `requirements.txt` + `scripts/bootstrap.ps1` |
| 模块可独立验证 | 10 个模块零反向依赖，每个都有单元不变量（`pytest`） |
| 不依赖外网也能自检 | 内置确定性哈希嵌入 + 内存向量库，`--offline` 全绿 |
| 复用成熟开源 | 推理/嵌入/重排/向量库全部用现成 SOTA，不自研轮子 |

---

## 2. 架构（四层 · 10 模块）

```
                    ┌──────────────── console/index.html（单文件控制台） ────────────────┐
                    └─────────────────────────── HTTP ────────────────────────────────┘
                                                 │
L2  能力层        ┌──────────── gateway（FastAPI · OpenAI 兼容） ────────────┐
                  │   /v1/chat/completions   /v1/embeddings                 │
                  │   /v1/rag/{ingest,answer,search}   /v1/agent/run        │
                  └───────┬───────────────────────────────┬────────────────┘
                          │                               │
                     rag（带引用/拒答）              agent（ReAct + 工具注册表）
                          │                               │
L1  数据层        retrieval（BM25 + Dense → RRF → 重排二次融合 w=0.5）
                    ├── embedding（bge-m3 via Ollama / 哈希兜底）
                    ├── vectorstore（Qdrant 本地模式 / 内存兜底）
                    └── reranker（bge-reranker-base ONNX / 中性降级）
                          │
L0  底座层        inference（Ollama · 线程锁 2-4 · 流式 · 非法参数不发 think）

L3  交付层        eval（中文黄金集 · top1/top3/MRR + 对抗性重排不变量）
                  console（单文件 HTML）  ·  scripts / docs / tests
```

依赖方向严格单向，任何模块都可单独实例化与测试，见 `docs/ARCHITECTURE.md`。

---

## 3. 快速开始

### 3.1 前置

- Python 3.12（`uv` 会自动装）
- [uv](https://docs.astral.sh/uv/)（`pip install uv` 或 winget）
- [Ollama](https://ollama.com)（提供 LLM 与嵌入；离线模式可不装）

### 3.2 一键引导

```powershell
# Windows
pwsh -File scripts/bootstrap.ps1
# 等价手动步骤
uv sync --extra dev
uv run nebula download-models        # ollama pull qwen3:4b / bge-m3 + ModelScope 拉重排器
```

### 3.3 起服务

```powershell
uv run nebula serve                  # http://127.0.0.1:8765
uv run nebula serve --offline        # 无模型也能起，用于接前端联调
```

### 3.4 用

```powershell
uv run nebula ingest docs            # 文档入库（目录递归 *.md）
uv run nebula ask "CPU 上线程数应该设多少？" --trace
uv run nebula search "重排权重" --top-k 5
uv run nebula agent "帮我算 sqrt(2)*3"
uv run nebula bench --rounds 3       # 推理吞吐基准
uv run nebula eval                   # 检索效果评测
uv run nebula verify                 # 一键自检：单测 + 端到端冒烟
```

任何 OpenAI 客户端可直接接入：

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-needed")
print(c.chat.completions.create(model="qwen3:4b",
      messages=[{"role": "user", "content": "用一句话解释 RRF 融合"}]).choices[0].message.content)
```

---

## 4. 四条被钉死的不变量

| 不变量 | 为什么 | 在哪验证 |
|---|---|---|
| **线程数 ∈ [2,4]** | 小量化模型是内存带宽瓶颈；`cpu_count-1` 实测慢 2–4 倍（32.5 → 8.2 tok/s） | `tests/test_core.py` + `nebula bench` |
| **重排不得接管排序** | 语言能力不足的重排器会无人制衡地压掉冠军；改为第三路信号 w=0.5 后 top-1 从 10/12 回到 12/12 | `tests/test_retrieval.py::test_adversarial_rerank_keeps_champion` |
| **写入→检索往返一致** | 存储层是一切检索效果的地基 | `tests/test_vectorstore.py` |
| **超时随输出预算放大** | CPU 上 1024 token 需 200–500s，固定 180s 超时会掐断生成，表现为"正文永远为空"——调大预算反而更容易失败 | `tests/test_generation_retry.py` |

对抗性不变量是**数学确定性**的，不会偶发失败：设榜长 n、RRF 常数 k、重排权重 w<1，
冠军分 − 亚军分 = `(1-w)·[1/(k+1) − 1/(k+n)] > 0`。

---

## 5. 目录

```
src/nebula/          10 个模块（config/inference/embedding/reranker/vectorstore/
                     sparse/chunking/fusion/retrieval/rag/agent/gateway/eval_harness）
console/index.html   单文件控制台（内联 CSS/JS，无外部依赖）
tests/               60 个测试：单元不变量 + 契约 + 隔离性护栏
scripts/             bootstrap.ps1 / verify.py / bench.py / rerank_ablation.py / make_hard_set.py
docs/                ARCHITECTURE.md / DEPLOYMENT.md / USAGE.md / **EVIDENCE.md（实测证据）**
docs/decisions/      ADR-001…007 + OPEN-DECISIONS.md
data/eval/           中文黄金集 set.json + 困难集 set_hard.json
.github/workflows/   CI（干净 Ubuntu 环境跑同一套离线判据）
```

**想知道"它到底跑得怎么样"，直接看 [`docs/EVIDENCE.md`](docs/EVIDENCE.md)**：
端到端冒烟 22 项、单元不变量、检索指标、重排消融、线程基准，全部是实测数字。

---

## 6. 已知边界

- **延迟**：本机 8C16T 无 GPU，单次完整 RAG 回答约 **2–5 分钟**，单轮 Agent 约 1–2 分钟，
  检索（含重排）0.3–0.8 秒。这是 CPU 解码的物理上限，不是实现缺陷（见 ADR-007）。
- 无 GPU：默认 `qwen3:4b` Q4。换 8B 需约 5GB 额外内存，配置里留了 `NEBULA_LLM_MODEL` 开关。
- Windows 端口：默认 8765（8000–8123 常被 WinNAT/Hyper-V 保留，绑定会 WinError 10013）。
- HuggingFace 不可达：模型统一走 ModelScope 镜像与 Ollama registry。
- LanceDB 在 Windows 无 wheel，已改用 Qdrant 本地模式（纯 Python wheel）。
- jieba 仅发布 sdist 且本机构建失败，已移除，中文改单字+双字切分（见 ADR-002）。

详见 `docs/DEPLOYMENT.md` 与 `docs/decisions/`。
