# Nebula AI Stack · Windows 一键引导
# 用法： pwsh -File scripts/bootstrap.ps1 [-Offline] [-SkipModels]
param(
    [switch]$SkipModels,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

function Step($name) { Write-Host "`n==> $name" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "    [!!] $msg" -ForegroundColor Yellow }

Step "检查 uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Warn "未找到 uv，尝试通过 pip 安装"
    python -m pip install --user uv
}
Ok ("uv " + (uv --version))

Step "固定 Python 3.12 并同步依赖（版本锁定）"
uv python pin 3.12 | Out-Null
uv sync --extra dev
Ok "依赖安装完成，锁文件 uv.lock 已生成"

Step "导出 requirements.txt（便于无 uv 环境复现）"
uv export --no-hashes --format requirements-txt --output-file requirements.txt | Out-Null
Ok "requirements.txt 已导出"

Step "创建数据目录"
New-Item -ItemType Directory -Force -Path "data", "data\qdrant", "data\eval", "models" | Out-Null
Ok "data/ models/ 就绪"

if (-not $SkipModels) {
    Step "拉取模型（Ollama: qwen3:4b + bge-m3；ModelScope: bge-reranker-base ONNX）"
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        uv run nebula download-models
    } else {
        Warn "未安装 Ollama，跳过模型拉取。系统仍可用 --offline 模式运行。"
        Warn "安装地址：https://ollama.com/download"
    }
}

if ($Offline) {
    Step "离线模式自检"
    uv run python -m nebula.smoke
} else {
    Step "自检（单元不变量 + 端到端冒烟）"
    uv run nebula verify
}

Write-Host "`n引导完成。启动服务： uv run nebula serve" -ForegroundColor Green
