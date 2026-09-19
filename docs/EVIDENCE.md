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

## 7. 干净环境可复现（两份独立证据）

### 7.1 本机一键自检

```powershell
uv run python scripts/verify.py          # 含在线链路（约 13 分钟，CPU 推理）
uv run python scripts/verify.py --fast   # 仅离线路径（约 5 秒）
```

```
=== 1/4 依赖导入 ===      [PASS] 模块可导入（版本锁定生效）  all imports ok
=== 2/4 单元不变量 ===    [PASS] pytest 全绿                passed=60 failed=0 errors=0 skipped=0 exit=0
=== 3/4 端到端冒烟 ===    [PASS] 离线契约 + 在线链路        结论：端到端全绿
=== 4/4 评测基线 ===      [PASS] 离线评测可运行且不变量成立  top1=21/24 top3=24/24 mrr=0.9375 对抗不变量=True

总计 4/4 项通过
结论：干净环境验证全绿
```

**连续三次 `--fast` 复跑，结果逐位相同**（`passed=60 failed=0 errors=0` × 3）——
判据本身必须是稳定的，否则它没有资格当判据。

#### 判据设计上的硬约束（三层根因，都是踩出来的）

**① 不从人眼输出里提取机器判据。**
第一版用 `"passed" in stdout` 判断，失败时打印「输出为空（疑似进程被终止）」，
而真实输出里明明有 `ERROR: file or directory not found: tests` ——
**报告与事实不符**，把排障引向错误方向。

**② 判据改用进程内插件计数，不用退出码也不用输出文本。**
修好措辞后又出现新的偶发形态：stdout 尾部是
`.......(60 个点)...... [100%]`，也就是**测试全部跑完、进程在打印汇总行之前退出**
（rc=1、stderr 为空）。任何基于 stdout 汇总行的判断都会误报失败。

**③ 真凶：pytest 收尾时的批量删除被环境守卫拦截。**
把异常类型打进报告后才看见：
```
[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED] {"count":337,"threshold":50,...,
 "targets":["...\\Temp\\pytest-of-Administrator\\garbage-<uuid>"]}
→ SystemExit: 1
```
pytest 会话收尾要递归删除 337 个历史临时文件，被环境的删除守卫拒绝，
于是**测试全过、汇总行永远打不出来**。这也解释了"偶发"：累积超阈值后才必然触发。

**④ 最终修法：不让 pytest 使用它的临时目录机制。**
`tests/conftest.py` 用同名夹具覆盖内建 `tmp_path`，改为 `tempfile.mkdtemp()`
且**不做任何清理** → 测试进程不再发起递归删除，问题从根上消失。
（中途试过 `--basetemp` 指向仓库内目录，只是把被拦的位置从"收尾"挪到了"setup"。）

最终实现：`verify.py` 在自身进程内 `pytest.main(..., plugins=[collector])`，
按 `report.when == "call"` 统计计数，**以计数作为唯一判据**；
并把「运行没跑起来」与「测试没通过」严格区分：只对前者重试，绝不因后者重试
（后者重试等于掩盖缺陷）。完整复盘见 `docs/decisions/ADR-008`。

### 7.2 GitHub Actions（真正的"干净环境"）

在完全独立的干净 Ubuntu runner 上（不装 Ollama、不下载任何模型、只用锁文件）：

```
workflow: verify   触发: push   结论: success   耗时: 21s
```

CI 依次执行：`uv python install 3.12` → `uv sync --extra dev --frozen`
→ `pytest -q tests` → `python -m nebula.smoke`（自动跳过在线部分）
→ `nebula eval --offline` → `python scripts/verify.py --skip-eval`。

**这一条是"干净环境一键复现"最硬的证据**：它与开发机无关、与本地缓存无关、
与已下载的模型无关，全靠 `uv.lock` + 离线兜底实现（`HashingEmbedder` + `MemoryVectorStore`）。

查看：`gh run list -R CJX0712/nebula-ai-stack`
