# ADR-003: 推理线程数强制钳制在 2–4，并写成回归测试

## Status
Accepted (2026-09-20)

## Background
几乎所有示例代码都写 `n_threads = os.cpu_count() - 1`。在 8C16T 机器上这是 15，
恰好落在**最差**性能点。

实测（Q4 量化小模型，固定提示词 seed=42，两次取最好）：

| threads | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| tok/s | 32.3 | **32.5** | 25.6 | 23.6 | 25.1 | **8.2** |

根因：小量化模型的推理瓶颈是**内存带宽**，不是算力。线程超过物理核后，
多出来的线程只在抢内存总线。

## Decision
1. `config.auto_threads()` 统一计算：`max(2, min(4, n // 4))`，全局唯一入口。
2. `OllamaLLM` 构造时再次钳制 `max(1, min(4, threads))`，防绕过。
3. 写成回归断言 `tests/test_core.py::test_auto_threads_is_locked_2_to_4`，
   防止被"优化"回 cpu_count-1。
4. 提供 `scripts/bench.py` 让任何人在自己机器上重测（不走钳制路径）。
5. 重排器 ONNX 同理：`intra_op_num_threads = 2`。

## Consequences
- 正面：默认即最优；结论可复现；新机器也能一键复测
- 负面：**预填（prompt eval）**阶段实测 8 线程最快（69.3 vs 56.3 tok/s），
  被迫放弃该阶段的局部最优
- 理由：预填比解码快一个数量级，端到端耗时由解码主导 → 按解码调优

## Related ADRs
无
