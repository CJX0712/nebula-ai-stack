# 系统架构 · Nebula AI Stack

> 作者：晨星 ｜ 版本：0.1.0

## 1. 设计目标与取舍

| 目标 | 实现方式 | 有意不做 |
|---|---|---|
| 端到端可运行 | FastAPI 网关 + 10 模块单向依赖 | 不做分布式、不做多租户 |
| 模块可独立验证 | 每个模块有 Protocol 接口 + 单元不变量 | 不引入编排框架（LangChain 等）避免抽象泄漏 |
| 干净环境可复现 | `uv.lock` + `requirements.txt` + 引导脚本 | 不依赖 GPU / 不依赖外网模型仓库 |
| 失败可降级 | 重排/向量库/LLM 三层都有中性降级路径 | 不静默返回编造内容（降级必须显式标注 mode） |

**核心取舍**：所有外部能力（模型、向量库、重排器）都通过**依赖注入**进入系统，
`runtime.build_system()` 是唯一装配点。这样单测可以注入确定性假实现，
离线环境可以跑通全链路，生产环境换成真实实现而不改任何调用代码。

---

## 2. 分层与依赖方向

```
L3 交付   console(HTML)   scripts(verify/bench/bootstrap)   docs
              │ HTTP
L2 能力   gateway ──┬──> rag ──┐
                    └──> agent ─┴──> retrieval ──┬──> embedding
                                                 ├──> vectorstore
                                                 └──> reranker
                                                        │
L0 底座                                        inference(LLM) ← Ollama
```

规则（CI 可校验）：
1. 依赖只能向下，`L0` 不得 import `L1+`
2. `gateway` 只做协议适配与参数校验，**不含业务逻辑**
3. `retrieval` 不得直接调用 `inference`（检索与生成解耦）
4. 任何模块不得读取全局单例；一切依赖从构造函数进入

---

## 3. 模块契约

### 3.1 inference（L0）
```python
class OllamaLLM:
    model: str ; threads: int(2..4) ; ctx: int
    async def achat(messages, temperature=, num_predict=, seed=) -> LLMResult
    async def astream(messages, ...) -> AsyncIterator[str]
    async def ahealth() -> bool
@dataclass LLMResult: content, thinking, eval_tokens, eval_seconds, mode
```
- **不变量**：`threads ∈ [2,4]`（`auto_threads()` 强制）；`mode != "llm"` 时 `content` 必须为空或明确声明降级原因。
- **不发 `think` 字段**：该字段不省 token，只会把思维链倒进正文。
- **超时随预算放大**：`_timeout_for(num_predict) = max(base, num_predict / 2tok·s⁻¹)`，
  否则调大预算会被固定超时掐断，表现为"正文永远为空"（见 `docs/decisions/ADR-007`）。
- 失败不锁死：每次调用重新尝试，避免"服务后启动导致永久降级"。

### 3.2 embedding（L0）
```python
class OllamaEmbedder: async def aembed(texts) -> list[list[float]]
class HashingEmbedder: dim=256  # 确定性、零依赖、离线可复现
```
- **不变量**：`len(embed(texts)) == len(texts)`；向量已完成 L2 归一化（哈希实现）。
- Ollama 不可达时**自动**降级到哈希嵌入，并在 `last_mode` 标注实际路径。

### 3.3 reranker（L0）
```python
class OnnxReranker: enabled: bool ; def score(query, documents) -> list[float]
class NullReranker:  def score(...) -> [0.0]*n
```
- **不变量**：任何异常都必须转成中性分数，绝不上抛；`enabled=False` 时融合顺序原样透传。
- 模型缺失 → 构造期即降级并记录 `_error`，不影响主链路启动。

### 3.4 vectorstore（L1）
```python
class QdrantVectorStore:  # 本地模式，零服务
    def upsert(records) -> int ; def search(vector, k) -> list[VectorHit]
    def count() -> int ; def clear() -> None
class MemoryVectorStore: ...  # 兜底
```
- **不变量**：往返一致（upsert 后同向量检索必在首位）。
- 点 ID 必须是 UUID 字符串（Qdrant 约束，见 `retrieval.add_chunks` 的 `uuid5`）。
- `clear()` **逐点删除**而非删目录：文件级删除在受管环境下可能被回收站策略拦截。

### 3.5 sparse / chunking / fusion（L1 纯函数层）
- `sparse.BM25Index`：Okapi BM25(k1=1.5,b=0.75)，中文按单字+双字切分（不依赖 jieba，保证零编译）。
- `chunking.split_text`：段落优先，超长硬切，块间重叠。
- `fusion.rrf`：只吃名次 → 跨量纲免校准。

### 3.6 retrieval（L1 编排核心）
```python
class HybridRetriever:
    search(query, top_k=5, use_rerank=None) -> RetrievalResult
```
流程：
```
query ──┬── BM25 排序 ──┐
        └── 稠密排序 ────┴── RRF(k=60, w=[1,1]) ── 一阶段融合榜 fused
                                   │
                       shortlist = fused[:max(rerank_top_n, top_k)]
                                   │
                       重排打分 → reranked 排序
                                   │
                  RRF(k=60, w=[1.0, 0.5]) ── 二次融合 ── final
```
- **不变量**：`w<1` 时，注入对抗性重排器（完全倒序）后一阶段冠军仍为最终第一。
  数学恒等式：`Δ = (1-w)·[1/(k+1) - 1/(k+n)] > 0, n>1`。
- **短名单约束**：`len(shortlist) >= top_k`，否则会静默丢候选。
- trace 同时保留 `bm25_top / dense_top / fused_top / rerank_top / final_top`，可解释检索。

### 3.7 rag（L2）
```python
class RagEngine: async def aanswer(query) -> RagAnswer(answer, citations, mode, trace)
```
- `mode` 三态：`rag`（真实生成）/ `degraded`（LLM 不可用 → 给原文摘录）/ `abstain`（无命中 → 拒绝回答）。
- 引用编号与命中一一致，不重排不丢弃。

### 3.8 agent（L2）
```python
class ToolRegistry: register/get/names/prompt/run
class ReactAgent:   async def arun(goal) -> AgentResult(steps, status, iterations)
```
- 输出协议：单轮只输出一个 JSON（`tool` 或 `final`），解析取**最后一个**合法 JSON。
- **有界**：`max_steps` 硬上限；解析失败立刻收口并标注 `parse_error`，不空转。
- 内置工具：`rag_search` / `calculator`（AST 白名单，禁 `eval`）/ `now` / `read_file`（路径越界校验）。

### 3.9 gateway（L2 协议层）
| 端点 | 用途 | 契约要点 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI 兼容 | `choices[].message`、`usage` 三件套、支持 `stream=true` |
| `POST /v1/embeddings` | OpenAI 兼容 | `data[].embedding` |
| `POST /v1/rag/ingest` | 入库 | 返回新增块数与总块数 |
| `POST /v1/rag/answer` | 带引用问答 | 返回 `mode` + `citations` + `trace` |
| `POST /v1/rag/search` | 检索 | 返回每路的 `dense_rank/sparse_rank/rerank_score` |
| `POST /v1/agent/run` | Agent | 返回 `steps[]` 与 `status` |
| `GET /healthz` | 健康 | 逐模块真实状态，不撒谎 |

> 请求模型必须定义在**模块级**：`from __future__ import annotations` 下若定义在函数内，
> FastAPI 无法解析前向引用，端点会静默 422。（已踩，见 `docs/decisions/ADR-004`）

### 3.10 eval（L3）
`run_eval(data, offline, top_k)` → `{top1, top3, mrr, avg_latency_ms, adversarial_invariant}`。
黄金集 `data/eval/set.json`：12 篇文档 × 24 条中文查询，每篇文档的关键词互不重叠。

---

## 4. 启动时序

```
nebula serve
  └─ Config.from_env()            # 环境变量 → 配置
  └─ build_system(cfg, offline)
       ├─ OllamaLLM(threads=2..4)
       ├─ OllamaEmbedder(bge-m3) / HashingEmbedder
       ├─ build_reranker(models/bge-reranker-base) → OnnxReranker | NullReranker
       ├─ build_store(data/qdrant)  → QdrantVectorStore | 内存
       ├─ HybridRetriever(...)
       ├─ RagEngine(retriever, llm)
       └─ ToolRegistry + ReactAgent
  └─ create_app(system).add_middleware(CORS)
  └─ uvicorn.run(host=127.0.0.1, port=8765)
```
装配失败不会崩溃：任何一层不可用都退化为可观测的降级态，`/healthz` 会如实上报。

---

## 5. 可验证性设计（Harness）

| 层次 | 判据 | 位置 |
|---|---|---|
| 单元 | 49 个断言，含 3 条确定性不变量 | `tests/` |
| 契约 | 离线契约 + 在线链路两张清单 | `nebula.smoke` |
| 效果 | top1/top3/MRR + 对抗不变量 | `nebula eval` |
| 性能 | 线程扫描表 + tok/s | `scripts/bench.py` |
| 复现 | 依赖导入 → 单测 → 冒烟 → 评测 | `scripts/verify.py` |

**没有机器判据的"完成"，不算完成。**
