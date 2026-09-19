# ADR-007: 输出预算与 HTTP 超时必须联动放大

## Status
Accepted (2026-09-20, 由两次线上冒烟失败倒逼)

## Background
`nebula.smoke` 的在线链路连续两次在同一个断言上失败：

```
[FAIL] RAG 生成成功  mode=degraded tok_s=0.0
```

排查路径（**注意：这个故障有三种外观相同、根因不同的形态**）：

| 观察 | 根因 | 修法 |
|---|---|---|
| `content=""`、`thinking` 非空 | 思维链吃满 `num_predict` | 放大预算重试一次（ADR 见 `rag._generate`） |
| `mode="degraded: ReadTimeout..."` | **HTTP 超时**先于生成完成 | 超时随预算线性放大（本 ADR） |
| `mode="degraded: ConnectError"` | Ollama 未启动 | 运维问题，不属本 ADR |

第一次只修了形态 1（预算 512 → 1024），仍然失败。真实原因是形态 2：
CPU 上实测解码 2–5 tok/s，`num_predict=1024` 需要 **200–500 秒**，
而客户端超时是固定的 180 秒 —— 请求在生成中途被掐断，
表现为 `content=""`，看起来和"思维链吃满"一模一样。

**这是本项目最隐蔽的一类故障**：调大预算反而更容易失败。

## Decision
1. 默认超时从 180s 提到 **300s**（`NEBULA_LLM_TIMEOUT`）。
2. **超时随预算动态放大**，写成代码而不是文档约定：
   ```python
   MIN_TOK_S = 2.0        # CPU 上量化 4B 的保守解码下界
   MAX_TIMEOUT_S = 1800.0
   def _timeout_for(num_predict):
       if not num_predict: return self.timeout
       return max(self.timeout, min(MAX_TIMEOUT_S, num_predict / MIN_TOK_S))
   ```
3. 降级回答**必须**回显 `out.mode`（含异常类型），否则排障时只能靠猜。
4. 写成回归断言 `test_http_timeout_scales_with_token_budget`。

## Consequences
- 正面：调大预算不再有隐藏惩罚；失败信息自带根因
- 负面：最坏情况下单次请求会挂住 5–15 分钟（CPU 推理的物理现实）。
  缓解手段：控制 `NEBULA_TOP_K`（缩短预填）、生产环境换 GPU 或云端模型后端
- **诚实的代价**：本机 4B 模型单次完整 RAG 回答耗时约 2–5 分钟，
  这是 CPU 解码速度决定的，不是实现缺陷

## Related ADRs
ADR-003（同样是"按解码而非预填调优"的推论）
