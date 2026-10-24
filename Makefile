# 常用入口。解释器默认用仓库内 .venv，可 `make PY=python3 test` 覆盖
PY      ?= .venv/bin/python
BIN     ?= .venv/bin
HOST    ?= 127.0.0.1
PORT    ?= 8000

.PHONY: install index demo test fixtures eval samples

install:  ## 建虚拟环境并安装（含稠密检索与测试依赖）
	test -x $(PY) || python3 -m venv .venv
	$(PY) -m pip install -e '.[dense,dev]'

index:  ## 建 rag.db。显式开稠密路：BM25 单路的充分性判断不可靠
	USE_DENSE=1 $(BIN)/ragwriter index

demo:  ## 起服务：API + web/ 静态页；无 LLM_API_KEY 时自动进入回放模式
	$(BIN)/uvicorn ragwriter.api.app:app --host $(HOST) --port $(PORT)

test:
	$(PY) -m pytest -q

fixtures:  ## 录制演示用回放素材（需要 LLM_API_KEY）
	$(PY) -m ragwriter.agent.replay record --name blog_geo --message "写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？"

eval:  ## 19 条多轮用例的真链路评估，结果写入 docs/eval-report.md（需要 LLM_API_KEY）
	USE_DENSE=1 $(BIN)/ragwriter eval --live --no-judge

samples:  ## 重新生成 samples/ 下的示例输出（需要 LLM_API_KEY）
	USE_DENSE=1 $(PY) scripts/make_samples.py samples
