"""稀疏检索：中文 BM25（Okapi）。

为什么自己写而不用 rank-bm25：中文必须先分词，且需要把「分词器」这层
显式暴露出来供测试替换。实现约 80 行，完全可控。
"""
from __future__ import annotations

import math
import re
from collections import Counter

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文用「一元 + 二元」切分，英文数字按词切并小写归一。

    为什么不用 jieba：jieba 在 Windows 只有 sdist，干净环境构建易失败；
    而 CJK 二元组是信息检索里的标准做法（Solr/Whoosh CJK 同款），
    零依赖、确定性、对新词不敏感，效果足以支撑 BM25 稀疏路。
    """
    if not text:
        return []
    tokens: list[str] = []
    text = text.lower()
    for m in _LATIN.findall(text):
        tokens.append(m)
    # 逐段 CJK：一元 + 二元
    buf: list[str] = []
    for ch in text:
        if _CJK.match(ch):
            buf.append(ch)
        else:
            tokens.extend(_cjk_terms(buf))
            buf = []
    tokens.extend(_cjk_terms(buf))
    return tokens


def _cjk_terms(chars: list[str]) -> list[str]:
    if not chars:
        return []
    out = list(chars)
    out.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return out


class BM25Index:
    """Okapi BM25，k1=1.5, b=0.75。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._ids: list[str] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._len: list[int] = []
        self._avg_len: float = 0.0

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._ids.append(doc_id)
        self._docs.append(tokens)
        tf = Counter(tokens)
        self._tf.append(tf)
        self._len.append(len(tokens))
        for t in tf:
            self._df[t] += 1
        total = sum(self._len)
        self._avg_len = total / max(1, len(self._len))

    def clear(self) -> None:
        self._docs.clear()
        self._ids.clear()
        self._tf.clear()
        self._df.clear()
        self._len.clear()
        self._avg_len = 0.0

    def scores(self, query: str) -> dict[str, float]:
        """返回 {doc_id: bm25 分数}；语料为空或查询无词命中时返回全 0。"""
        if not self._ids:
            return {}
        q_tokens = tokenize(query)
        if not q_tokens:
            return {d: 0.0 for d in self._ids}
        n = len(self._ids)
        out: dict[str, float] = {}
        for i, doc_id in enumerate(self._ids):
            dl = self._len[i]
            tf = self._tf[i]
            s = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self._avg_len))
                s += idf * f * (self.k1 + 1) / max(1e-9, denom)
            out[doc_id] = s
        return out

    def ranking(self, query: str, limit: int | None = None) -> list[str]:
        sc = self.scores(query)
        order = sorted(sc, key=lambda d: (-sc[d], d))
        return order[:limit] if limit else order
