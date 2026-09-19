"""交叉编码器重排层（可选）。

模型：BAAI/bge-reranker-base，ONNXRuntime CPU 推理，文件来自 ModelScope 镜像。
不可用时**必须**降级为中性分数 [0.0]*n，使融合顺序原样透传，而不是抛异常。

重要：重排输出**不直接接管最终排序**，而是作为第三路信号以权重 w=0.5
与一阶段 RRF 二次融合（见 fusion.rrf 的数学不变量）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class NullReranker:
    name = "null"
    enabled = False

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)


class OnnxReranker:
    """bge-reranker-base ONNX。延迟加载，文件缺失即降级。"""

    name = "bge-reranker-base-onnx"

    def __init__(self, model_dir: Path, max_length: int = 512) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.enabled = False
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self._error: str | None = None
        self._try_load()

    # ---------- 加载 ----------
    def _find(self, *names: str) -> Path | None:
        for n in names:
            for cand in (self.model_dir / n, self.model_dir / "onnx" / n):
                if cand.exists():
                    return cand
        return None

    def _try_load(self) -> None:
        onnx_path = self._find("model.onnx", "model_quantized.onnx")
        tok_path = self._find("tokenizer.json")
        if onnx_path is None or tok_path is None:
            self._error = f"模型文件缺失（{self.model_dir}）：需要 model.onnx 与 tokenizer.json"
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            so = ort.SessionOptions()
            so.intra_op_num_threads = 2  # 与 LLM 同理：小模型是带宽瓶颈
            so.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self.enabled = True
        except Exception as e:  # 加载失败 -> 降级，绝不向上抛
            self._error = f"重排器加载失败：{e}"
            self._session = None
            self.enabled = False

    # ---------- 打分 ----------
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not self.enabled or not documents:
            return [0.0] * len(documents)
        out: list[float] = []
        for doc in documents:
            try:
                out.append(float(self._score_one(query, doc)))
            except Exception:
                out.append(0.0)
        return out

    def _score_one(self, query: str, doc: str) -> float:
        enc = self._tokenizer.encode(query or "", doc or "")
        ids = list(enc.ids)[: self.max_length]
        types = list(enc.type_ids)[: self.max_length]
        mask = [1] * len(ids)
        arr_ids = np.array([ids], dtype=np.int64)
        arr_mask = np.array([mask], dtype=np.int64)
        feeds = {"input_ids": arr_ids, "attention_mask": arr_mask, "token_type_ids": np.array([types], dtype=np.int64)}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        logits = self._session.run(None, feeds)[0]
        val = float(np.asarray(logits).reshape(-1)[0])
        return 1.0 / (1.0 + math.exp(-val))  # sigmoid，仅用于展示；融合只用名次


def build_reranker(model_dir: Path, enabled: bool = True):
    if not enabled:
        return NullReranker()
    r = OnnxReranker(model_dir)
    if not r.enabled:
        return NullReranker()
    return r
