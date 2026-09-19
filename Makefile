> Windows 用户请优先使用 scripts/bootstrap.ps1 / scripts/verify.py；本文件为 Unix 与 CI 提供等价入口。

PY ?= uv run python

.PHONY: help bootstrap sync download serve ingest ask search bench eval test verify clean

help:
	@echo "make bootstrap  安装依赖并导出锁定清单"
	@echo "make download   拉取模型（Ollama + ModelScope）"
	@echo "make serve      启动 API 网关 (127.0.0.1:8765)"
	@echo "make ingest     文档入库 (DIR=docs)"
	@echo "make ask        提问 (Q='...')"
	@echo "make bench      线程吞吐基准"
	@echo "make eval       检索评测"
	@echo "make test       单元测试"
	@echo "make verify     一键自检"

bootstrap:
	uv python pin 3.12
	uv sync --extra dev
	uv export --no-hashes --format requirements-txt --output-file requirements.txt
	mkdir -p data/qdrant data/eval models

sync:
	uv sync --extra dev

download:
	uv run nebula download-models

serve:
	uv run nebula serve

ingest:
	uv run nebula ingest $(DIR)

ask:
	uv run nebula ask "$(Q)" --trace

search:
	uv run nebula search "$(Q)"

bench:
	$(PY) scripts/bench.py

eval:
	uv run nebula eval

test:
	uv run pytest -q tests

verify:
	$(PY) scripts/verify.py

clean:
	rm -rf data/qdrant/* .pytest_cache .ruff_cache
