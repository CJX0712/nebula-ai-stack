# 悬而未决登记册（OPEN-DECISIONS）

> 规则：只追加 + 就地关闭（OPEN → RESOLVED）。每次会话开始先扫一遍，逐条判断能否关闭。
> 三类固定 slug：`waiting-on-external-condition` / `design-decision-to-evaluate` / `existing-design-boundary`

| Date | Source | Open Item | Related Constraints | Current Leaning | Blocked By | Resolves When | Status |
|---|---|---|---|---|---|---|---|
| 2026-09-20 | 需求确认 | LLM 档位是否升到 8B | 内存空闲 1–5GB；Docker 已占 7.9GB | 保持 qwen3:4b，配置留 `NEBULA_LLM_MODEL` 开关 | 需实测 8B 常驻内存与 tok/s | 用户提出"更聪明"诉求或内存释放后 | OPEN |
| 2026-09-20 | 设计评估 | 重排器是否换更小的多语言模型 | 当前 bge-reranker-base ONNX 1.06GB，常驻约 1.5GB | 先用现状，按 w=0.5 融合；不因"体积大≠准"贸然更换 | 需在黄金集上做 A/B | bge-reranker 系列出现明显更小且更准的替代 | OPEN |
| 2026-09-20 | 设计评估 | 是否引入 Langfuse/OpenTelemetry 做可观测 | 内存紧张；当前用 JSONL trace + `/healthz` | 暂不引入，trace 已足够定位链路问题 | 需要跨进程/跨会话追踪时 | 出现多服务编排需求 | OPEN |
| 2026-09-20 | 现有边界 | 向量库限本地模式 | Qdrant 本地模式不支持多进程并发写 | 单进程服务；如需并发需切服务端模式 | 需要水平扩容时 | 部署多实例时切 `url=` 模式 | OPEN |
| 2026-09-20 | 现有边界 | 中文 BM25 未用词典分词 | 已移除 jieba（无 wheel） | 单字+双字切分，稠密检索兜底 | 出现中文召回瓶颈证据 | 评测显示 BM25 路显著拖后腿 | OPEN |
| 2026-09-20 | 设计评估 | Agent 工具集是否扩展（HTTP/代码执行） | 安全边界与沙箱缺失 | 暂只保留 4 个内置工具 | 需沙箱方案 | 明确要接外部系统时 | OPEN |
| 2026-09-20 | 需求确认 | 是否需要 Docker Compose 全容器化部署 | 本机 Docker 已占 7.9GB 内存 | 本机进程为主，容器作为可选档 | 目标机器内存条件 | 需要在服务器上交付时 | OPEN |
