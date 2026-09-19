# 使用指南 · Nebula AI Stack

> 作者：晨星

## 0. 三种使用姿势

| 场景 | 入口 | 是否需要模型 |
|---|---|---|
| 命令行快速验证 | `nebula ask/search/agent/eval` | RAG 需要，检索/离线不需要 |
| 程序集成 | HTTP `http://127.0.0.1:8765` | 是 |
| 图形界面 | `console/index.html` | 是（可 `--offline` 看链路） |
| OpenAI SDK 直连 | `base_url=http://127.0.0.1:8765/v1` | 是 |

---

## 1. 五条核心命令

```powershell
uv run nebula serve                                   # 起服务
uv run nebula ingest docs                             # 文档入库（目录递归 *.md）
uv run nebula ask "CPU 上线程数应该设多少？" --trace     # 带引用的 RAG 问答
uv run nebula search "重排权重" --top-k 5               # 只看检索链路
uv run nebula agent "帮我算 sqrt(2)*3"                  # 工具调用 Agent
uv run nebula bench --rounds 3                        # 推理吞吐基准
uv run nebula eval                                    # 检索评测
uv run nebula verify                                  # 一键自检
uv run nebula download-models                         # 拉模型
```

`--offline` 追加到 `ask/search/agent/serve` 后，可在无 Ollama 环境下验证链路。

---

## 2. 典型工作流

### 2.1 建知识库并问答
```powershell
uv run nebula ingest .\docs          # 输出：每篇文档切了多少块
uv run nebula ask "重排为什么不能接管排序" --trace
```
输出包含三部分：**回答**（带 `[1][2]` 引用编号）、**引用列表**（来源/融合分/两路名次/重排分）、
**链路 trace**（各阶段 top 列表）。

### 2.2 观察检索到底哪一步出的问题
```powershell
uv run nebula search "top-1 低但 top-3 高" --top-k 5
```
判读表：

| top1 | top3 | 结论 | 动作 |
|---|---|---|---|
| 高 | 高 | 正常 | 别动 |
| 低 | 高 | **排序问题** | 查重排权重（保持 <1） |
| 低 | 低 | **召回问题** | 查分块/分词/嵌入，别动重排 |

### 2.3 Agent 用例
```powershell
uv run nebula agent "先查一下知识库里线程数怎么设，再算 16/4 是多少"
```
输出 `步骤表`：每一步的 `action / tool / args / observation`，便于判断是模型不会规划还是工具不确定。
状态含义：`ok` 完成 ｜ `max_steps` 达上限 ｜ `parse_error` 模型没按 JSON 协议输出 ｜ `degraded` LLM 不可用。

---

## 3. HTTP API

### 3.1 OpenAI 兼容
```bash
curl http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3:4b",
  "messages": [{"role":"user","content":"用一句话解释 RRF"}]
}'
```
```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="x")
r = c.chat.completions.create(model="qwen3:4b",
      messages=[{"role": "user", "content": "你好"}])
```

### 3.2 原生 RAG
```bash
# 入库
curl -X POST http://127.0.0.1:8765/v1/rag/ingest -H "Content-Type: application/json" \
  -d '{"docs":[{"text":"线程数应锁在 2 到 4 之间","source":"a.md"}],"reset":true}'

# 问答（带引用）
curl -X POST http://127.0.0.1:8765/v1/rag/answer -H "Content-Type: application/json" \
  -d '{"query":"线程数应该设多少"}'

# 检索（看得到每一路排序）
curl -X POST http://127.0.0.1:8765/v1/rag/search -H "Content-Type: application/json" \
  -d '{"query":"重排权重","top_k":5,"use_rerank":true}'
```

### 3.3 Agent
```bash
curl -X POST http://127.0.0.1:8765/v1/agent/run -H "Content-Type: application/json" \
  -d '{"goal":"帮我算 sqrt(2)*3","max_steps":4}'
```

### 3.4 响应字段速查
| 字段 | 含义 |
|---|---|
| `mode` | `rag` 真实生成 ／ `degraded` 降级摘录 ／ `abstain` 无依据拒答 |
| `tok_s` | 生成吞吐（tokens/s） |
| `citations[].score` | 融合分（RRF，量纲无关） |
| `citations[].dense_rank` / `sparse_rank` | 稠密/稀疏两路名次，`null` 表示该路未召回 |
| `citations[].rerank_score` | 重排 sigmoid 分（仅展示，不决定最终顺序） |
| `trace.rerank_used` | 本轮是否走了重排融合 |

---

## 4. 控制台

打开 `console/index.html`（双击即可，无构建、无外部依赖），右上角填 `http://127.0.0.1:8765` → 点“连接”。

四个页签：
1. **问答** — 提问、看回答与引用、展开 trace
2. **检索** — 逐条看两路名次与重排分
3. **入库** — 直接粘贴文本入库
4. **Agent** — 输入目标，看步骤表

---

## 5. 调参速查

| 想改善 | 调什么 | 怎么调 |
|---|---|---|
| 生成太慢 | `NEBULA_LLM_THREADS` | 保持 2–4；用 `scripts/bench.py` 实测本机最优点 |
| **回答延迟数分钟** | `NEBULA_LLM_MAX_TOKENS` | CPU 上 1024 token ≈ 200–500s，这是解码速度决定的。要快就降到 512 并接受偶发重试 |
| 回答跑题 | 分块参数 | 中文技术文档 400 字 / 60 重叠起步 |
| top-1 不准 | `NEBULA_RERANK_WEIGHT` | 扫 0 / 0.25 / **0.5** / 0.75 / 1.0；**别设 1.0** |
| 细节丢失 | `NEBULA_TOP_K` | 3 → 5，注意上下文预算与预填耗时 |
| 内存不足 | `NEBULA_RERANK_ENABLED=0` | 省约 1.5GB，检索质量略降 |
| 想换更大模型 | `NEBULA_LLM_MODEL=qwen3:8b` | 需额外约 5GB 内存，速度减半 |

> **延迟预期（本机 8C16T 无 GPU，qwen3:4b Q4）**：单次完整 RAG 回答 2–5 分钟，
> 单轮 Agent 约 1–2 分钟，检索（含重排）约 0.3–0.8 秒，嵌入约 30ms/条。
> 这是 CPU 解码的物理上限，不是实现缺陷。要交互式体验，请换 GPU 或云端模型后端
> （实现同签名 `achat/astream` 注入即可，见 6.3）。

---

## 6. 二次开发

### 6.1 换一个向量库
```python
# 只需实现三个方法，注入即可，调用方零改动
class MyStore:
    name = "my-store"
    def upsert(self, records) -> int: ...
    def search(self, vector, k=5) -> list[VectorHit]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
```
在 `runtime.build_system()` 里替换 `build_store(...)` 即可。

### 6.2 加一个 Agent 工具
```python
from nebula.agent import Tool
tools.register(Tool(
    name="word_count",
    description="统计文本字数",
    args={"text": "待统计文本"},
    fn=lambda a: f"{len(a.get('text',''))} 字",
))
```

### 6.3 换对话模型后端
实现同签名 `achat / astream / ahealth` 即可（如 vLLM、OpenAI、TGI），
在 `runtime.build_system()` 中替换 `OllamaLLM(...)`。

---

## 7. 排错口诀

1. **先看 `/healthz`** —— 它不会撒谎：哪个模块降级、线程数、库中块数一眼可见。
2. **看 `mode`** 而不是猜文本长短 —— `degraded` 和 `rag` 都能返回非空文本。
3. **看 `trace`** 而不是怀疑模型 —— 双路名次能直接区分"召回坏了"还是"排序坏了"。
4. **改参数前先跑 `nebula eval`** —— 有基线才有资格谈优化。
