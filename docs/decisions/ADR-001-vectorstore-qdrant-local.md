# ADR-001: 向量库选 Qdrant 本地模式，弃用 LanceDB

## Status
Accepted (2026-09-20)

## Background
初版计划用 LanceDB 作为嵌入式向量库（零服务、落盘简单）。实测 `uv sync` 直接失败：

```
error: Distribution `lancedb==0.39.0` can't be installed because it doesn't have a
source distribution or wheel for the current platform
hint: You're on Windows (win_amd64), but `lancedb` only has wheels for
manylinux_2_28_x86_64 / manylinux_2_28_aarch64 / macosx_11_0_arm64
```

目标平台是 Windows，且项目第一原则是「干净环境一键复现」——不能依赖本地编译。

## Decision
改用 `qdrant-client` 的**本地模式**（`QdrantClient(path=...)`）：
- 纯 Python wheel（`py3-none-any`），全平台可装
- 本地模式无需服务端，直接落盘，内存占用可控
- 与 Qdrant 服务端 API 一致，未来若要上服务化无需改代码

同时保留 `MemoryVectorStore` 作为导入失败时的兜底实现。

## Consequences
- 正面：可在 Windows 一键复现；无需 Docker；升级路径清晰
- 负面：多引入 `grpcio`/`protobuf` 依赖（体积增大）；点 ID 必须为合法 UUID 字符串
- 约束：`retrieval.add_chunks` 用 `str(uuid.uuid5(...))` 生成 ID，不能再截断为 hex

## Related ADRs
ADR-002（同为"无编译依赖"约束下的取舍）
