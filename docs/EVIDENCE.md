# 验证证据 · Nebula AI Stack

> 全部数据在**本机实测**产生：Windows 11 / AMD Ryzen 7 8C16T / 15GB RAM / **无 GPU**。
> 采集时间：2026-09-20。复现命令写在每节末尾。

---

## 1. 端到端冒烟：22 项全绿

```
=== 离线契约（无外部依赖）===
  [PASS] healthz 200            status=200
  [PASS] healthz 含 modules
  [PASS] rag/ingest 200         {"chunks": 3, "documents": 3}
  [PASS] 入库块数 > 0            chunks=3
  [PASS] rag/search 命中         top=docs/threads.md
  [PASS] rag/answer 200         mode=degraded（离线无 LLM，显式降级而非编造）
  [PASS] rag/answer 带引用
  [PASS] OpenAI 兼容 chat.completions
  [PASS] embeddings 返回向量
  [PASS] agent/run 有界返回
  [PASS] models 列表非空

=== 在线真实链路（Ollama + bge-m3）===
  [PASS] Ollama 在线             ollama/qwen3:4b threads=4
  [PASS] 嵌入器在线              ollama-bge-m3
  [PASS] 重排器状态              bge-reranker-base-onnx enabled=True
  [PASS] 向量库状态              qdrant-local degraded=False
  [PASS] 在线入库                {"chunks": 3, "documents": 3}
  [PASS] RAG 生成成功            mode=rag tok_s=5.10 gen=thinking_truncated_retry
  [PASS] 答案非空
  [PASS] 引用可用
  [PASS] 检索链路 trace 完整
  [PASS] 重排路径可用
  [PASS] Agent 完成              status=ok steps=2

结论：端到端全绿
```

真实生成的回答（原文摘录，未编辑）：

> CPU 推理线程不能开太多，因为内存带宽成为瓶颈。实测显示，在 8 核 16 线程机器上运行 Q4 量化模型时，
> 4 线程性能达 32.5 tok/s，而 16 线程仅 8.2 tok/s，故线程数应锁在 2–4 之间。[1]

`gen=thinking_truncated_retry` 表示：第一次 1024 token 预算被思维链吃满，系统按设计放大预算重试后成功
（对应 `docs/decisions/ADR-007`）。**这条痕迹被刻意保留在 trace 里**，不做美化。

复现：`uv run python -m nebula.smoke`

---

## 2. 单元不变量

```
60 passed in 3.75s
```

覆盖 5 条硬不变量：

| 不变量 | 测试 |
|---|---|
| 线程数 ∈ [2,4] | `test_core.py::test_auto_threads_is_locked_2_to_4` |
| 对抗性重排不换冠军 | `test_retrieval.py::test_adversarial_rerank_keeps_champion` |
| 写入→检索往返一致 | `test_vectorstore.py::test_qdrant_or_fallback_roundtrip` |
| 超时随预算缩放 | `test_generation_retry.py::test_http_timeout_scales_with_token_budget` |
| **评测与生产索引隔离** | `test_eval_isolation.py::test_eval_is_isolated_from_production_index` |
| 冒烟骨架不得说谎 | `test_smoke_harness.py::test_report_returns_bool_not_none` |
| 真实向量后端不得静默降级 | `test_vectorstore.py::test_qdrant_backend_is_not_silently_degraded` |

对抗性不变量另有一组纯算术证明（`test_fusion.py`）：对 `n ∈ {3,5,12,40}`、`w ∈ {0,0.25,0.5,0.75,0.999}`
的全部组合，完全倒序的重排都换不掉一阶段冠军。

复现：`uv run pytest -q tests`

---

## 3. 重排器（真实 ONNX 推理）

```
模型：BAAI/bge-reranker-base  onnx/model.onnx  1060.92 MB（ModelScope 镜像）
load_s=5.03   enabled=True
score_s=0.25（3 条候选，单条约 83ms）
query = "CPU 推理线程数应该设多少"
docs  = ["线程数应锁在 2 到 4 之间，因为瓶颈是内存带宽而不是算力",
         "Windows 保留端口会导致绑定失败并报 WinError 10013",
         "重排权重必须小于 1，否则对抗性重排会换掉冠军"]
score = [0.9628, 0.0002, 0.0334]   argmax=0（正确）
```

---

## 4. 检索评测（黄金集 12 文档 × 24 中文查询）

评测在**独立内存索引**上运行，每次重建，既不读也不写生产索引（见 `test_eval_isolation.py`）。

| 口径 | top-1 | top-3 | MRR | 平均延迟 | 嵌入 | 重排 |
|---|---|---|---|---|---|---|
| **在线**（bge-m3 + bge-reranker-base） | **23/24** | **24/24** | 0.9792 | 1019 ms | OllamaEmbedder | OnnxReranker |
| **离线**（哈希嵌入 + 关闭重排） | 21/24 | 24/24 | 0.9375 | 1.1 ms | HashingEmbedder | NullReranker |

两种口径的 `adversarial_invariant.champion_kept` 均为 **true**。

> 离线口径的意义：**不装任何模型也能跑通全链路并给出可复现的质量下限**，
> 因此 CI 与代码评审可以完全不依赖外部服务。

---

## 5. 重排消融实验（把 ADR-005 在本项目数据上复现）

原始黄金集**没有区分度**（六种策略全部 24/24），因此另建困难集
（原 12 篇 + 4 篇"关键词枢纽"干扰文档，`scripts/make_hard_set.py`）。

困难集（16 篇语料，24 条查询，online）：

| 最终排序策略 | top-1 | top-3 |
|---|---|---|
| 纯 RRF（不用重排） | **23/24** | 24/24 |
| 重排直接接管（最常见的错误写法） | **22/24** | 24/24 |
| 二次融合 w=0.25 | 23/24 | 24/24 |
| **二次融合 w=0.50（本项目默认）** | **23/24** | 24/24 |
| 二次融合 w=0.75 | 23/24 | 24/24 |
| 二次融合 w=1.00（等权） | 23/24 | 24/24 |

单条查询重排耗时 **981 ms**（16 篇语料，CPU ONNX）。

**结论（不美化）**：
1. 「重排接管 = 净损害」在本项目复现：23 → 22。
2. 二次融合（w<1）与纯 RRF 持平，**没有观察到正增益** ——
   `bge-reranker-base` 在这套中文集上不添乱，但也没帮上忙。
3. 本设计的收益是**风险消除**，不是精度提升。要精度增益需换更强的中文重排器，
   并用 `scripts/rerank_ablation.py` 在困难集上验证后再切换。
4. 教训：**评测集必须能区分被评对象**。只看原始黄金集会得出"怎么配都行"的错误结论。

复现：`uv run python scripts/make_hard_set.py && uv run python scripts/rerank_ablation.py --data data/eval/set_hard.json`

---

## 6. 线程吞吐基准（把"线程锁 2–4"在本机重测）

固定提示词 + `seed=42` + `num_predict=160`，每档两轮取最好，用
`eval_count / eval_duration` 计速（排除模型加载）：

| num_thread | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| tok/s | **5.85** | 5.72 | 5.45 | 4.97 | 3.56 | **2.71** |

- 单调递减，**16 线程比 2 线程慢 2.16×** —— 与工程结论一致（内存带宽瓶颈）。
- 本机最优点是 2，与 4 仅差 2.3%（两轮采样内属噪声区间）；
  钳制区间取 `[2,4]` 覆盖两类机器，`NEBULA_LLM_THREADS=2` 也是合法选择。
- 绝对值 5.8 tok/s 解释了为什么单次 RAG 回答需要数分钟：**这是 CPU 解码的物理上限**。

复现：`uv run python scripts/bench.py --threads 2,4,6,8,12,16 --rounds 2`

---

## 7. 一键复现判据

```powershell
uv run python scripts/verify.py
```

输出为逐项 PASS/FAIL 汇总表，只有当「依赖导入 / 单元不变量 / 端到端冒烟 / 评测基线」
四项全部通过时结论才是"干净环境验证全绿"。

CI 亦在干净 Ubuntu 环境执行同一套离线判据（`.github/workflows/ci.yml`，
不装 Ollama、不下载模型）。
