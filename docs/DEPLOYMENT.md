# 部署指南 · Nebula AI Stack

> 作者：晨星 ｜ 目标平台：Windows 11 / Linux / macOS（CPU 优先，无 GPU 亦可）

## 1. 环境要求

| 项 | 最低 | 本机实测参考 |
|---|---|---|
| CPU | 4 核 | AMD Ryzen 7（8C16T） |
| 内存 | 8GB | 15GB（空闲 1–5GB 波动） |
| 磁盘 | 10GB | C 盘剩余 25GB |
| Python | 3.12（uv 自动安装） | CPython 3.12.14 |
| GPU | **不需要** | 无 |
| 可选 | Ollama（提供 LLM 与嵌入） | 0.34.0 |
| 可选 | Docker（仅用于可选的 Qdrant/Langfuse 服务化） | 29.7.2 |

**纯离线也能跑**：`--offline` 模式使用确定性哈希嵌入 + 内存向量库，不下载任何模型。

---

## 2. 一键部署（推荐）

```powershell
git clone <repo-url> nebula-ai-stack
cd nebula-ai-stack
pwsh -File scripts/bootstrap.ps1
uv run nebula serve
```

`bootstrap.ps1` 依次完成：检查 uv → 固定 Python 3.12 → `uv sync` → 导出 `requirements.txt`
→ 建数据目录 → 拉模型 → 跑自检。

参数：
- `-SkipModels`：不拉模型（仅代码环境）
- `-Offline`：跳过模型，直接跑离线自检

---

## 3. 手动部署（逐步，便于排障）

```powershell
# 3.1 Python 与依赖
uv python pin 3.12
uv sync --extra dev
uv export --no-hashes --format requirements-txt --output-file requirements.txt

# 3.2 模型
#   LLM + 嵌入：走 Ollama
ollama pull qwen3:4b        # 2.5GB，对话
ollama pull bge-m3:latest   # 1.2GB，多语言嵌入
#   重排器：走 ModelScope 镜像（HF 在本网络不可达）
uv run nebula download-models --skip-ollama

# 3.3 目录
mkdir data, data\qdrant, models   # PowerShell: New-Item -ItemType Directory

# 3.4 校验
uv run nebula verify
```

### 3.5 不使用 uv 的复现路径
```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
`requirements.txt` 由锁文件导出，含全部传递依赖与精确版本。

---

## 4. 启动与验证

```powershell
uv run nebula serve                 # 127.0.0.1:8765
uv run nebula serve --port 9000     # 换端口
uv run nebula serve --offline       # 无模型接前端
```

启动后自检：
```powershell
curl http://127.0.0.1:8765/healthz
uv run python -m nebula.smoke       # 端到端冒烟
```

控制台：浏览器打开 `console/index.html`（单文件，直接双击即可），右上角填网关地址后点“连接”。

---

## 5. 端口与 Windows 特殊处理

**8000–8123 是被 WinNAT / Hyper-V 保留的高频区间，绑定会报 `WinError 10013`。**
这不是程序缺陷。默认端口因此选 **8765**，可覆盖：

```powershell
$env:NEBULA_API_PORT = "9000"; uv run nebula serve
netsh int ipv4 show excludedportrange protocol=tcp   # 查看保留区间
```

---

## 6. 环境变量总表

| 变量 | 默认 | 说明 |
|---|---|---|
| `NEBULA_OLLAMA_BASE` | http://127.0.0.1:11434 | Ollama 服务地址 |
| `NEBULA_LLM_MODEL` | qwen3:4b | 对话模型（8B 可换 `qwen3:8b`） |
| `NEBULA_EMBED_MODEL` | bge-m3:latest | 嵌入模型 |
| `NEBULA_LLM_THREADS` | auto（钳制 2–4） | 推理线程数，**不要调大** |
| `NEBULA_LLM_CTX` | 4096 | 上下文窗口 |
| `NEBULA_LLM_MAX_TOKENS` | 1024 | 单次输出预算。**CPU 上 1024 token ≈ 200–500s**，调大前先读 ADR-007 |
| `NEBULA_LLM_TIMEOUT` | 300 | HTTP 超时基线（秒）。实际超时 = `max(此值, num_predict/2)`，上限 1800s |
| `NEBULA_RERANK_ENABLED` | 1 | 是否启用重排 |
| `NEBULA_RERANK_WEIGHT` | 0.5 | 重排二次融合权重，**必须 < 1** |
| `NEBULA_RERANK_DIR` | models/bge-reranker-base | 重排模型目录 |
| `NEBULA_RRF_K` | 60 | RRF 常数 |
| `NEBULA_TOP_K` | 5 | 默认返回条数 |
| `NEBULA_CHUNK_SIZE` / `_OVERLAP` | 420 / 60 | 分块参数 |
| `NEBULA_VECTOR_DIR` | data/qdrant | 向量库目录 |
| `NEBULA_API_HOST` / `_PORT` | 127.0.0.1 / 8765 | 服务绑定 |
| `NEBULA_AGENT_MAX_STEPS` | 6 | Agent 最大迭代 |

---

## 7. 可选：服务化部署（Docker）

本机 Docker 已占用较多内存，Nebula 默认**不**依赖 Docker。需要多机共享时可用：

```yaml
# docker-compose.optional.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.4
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
volumes: { qdrant_data: {} }
```
此时把 `NEBULA_VECTOR_DIR` 指向远端即可（需在 `vectorstore.py` 增加 `url=` 分支，当前为本地模式）。

**内存预算参考**（16GB 机器）：Ollama qwen3:4b 常驻约 3–4GB，bge-m3 约 1.5GB，
重排 ONNX 约 1.5GB（fp32）。若内存吃紧：
1. 只跑 `--offline` 做开发联调；
2. 关闭重排（`NEBULA_RERANK_ENABLED=0`）省 1.5GB；
3. 停掉与本项目无关的容器。

---

## 8. 故障排查

| 症状 | 根因 | 处置 |
|---|---|---|
| 端点返回 422，body 说字段缺失 | Pydantic 模型定义在 `create_app` 内部，FastAPI 无法解析前向引用 | 请求模型必须放模块级（ADR-004） |
| 绑定失败 `WinError 10013` | 端口落在 WinNAT/Hyper-V 保留区间 | 换 8765 或其他端口 |
| 推理只有个位数 tok/s | 线程数被设成 `cpu_count-1` | 锁 2–4；`scripts/bench.py` 复测 |
| `rag/answer` 一直 `degraded` | ①Ollama 未启动 ②思维链吃满预算 ③**HTTP 超时先于生成完成** | 看响应里 `trace.llm_mode` 与 `trace.gen`：`ConnectError`=①；`gen=thinking_truncated_retry`=②；`ReadTimeout`=③（调 `NEBULA_LLM_TIMEOUT`） |
| 检索全为新文档但答案无关 | 嵌入走了哈希降级（`embed_mode=fallback-hashing`） | 修好 Ollama 后重新入库 |
| `pip install lancedb` 失败 | LanceDB 无 Windows wheel | 已改用 Qdrant 本地模式 |
| `pip install jieba` 构建失败 | jieba 仅发布 sdist，本机编译受限 | 已移除该依赖，中文改单字+双字切分 |
| HF/GitHub 下载超时 | 本网络不可达 | 模型走 ModelScope；代码走 PyPI |

## 9. 卸载 / 清理

```powershell
Remove-Item -Recurse -Force .venv, data\qdrant    # 只删虚拟环境与向量索引
# models\ 是模型缓存，删掉需重新下载（约 1.1GB 重排器 + Ollama 侧模型另计）
```
