"""文本分块：按段落优先、字符数兜底的重叠切分。"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PARA_RE = re.compile(r"\n\s*\n|\r\n\s*\r\n")


@dataclass
class Chunk:
    text: str
    index: int
    source: str


def split_text(text: str, size: int = 420, overlap: int = 60, source: str = "") -> list[Chunk]:
    """先按段落聚合到接近 size，再对超长段落做硬切，块间保留 overlap。"""
    if not text or not text.strip():
        return []
    if overlap >= size:
        overlap = max(0, size // 4)

    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[Chunk] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}".strip() if buf else p
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(p) <= size:
            buf = p
            continue
        # 超长段落硬切
        start = 0
        while start < len(p):
            end = min(start + size, len(p))
            chunks.append(p[start:end])
            if end >= len(p):
                break
            start = end - overlap
    if buf:
        chunks.append(buf)

    out: list[Chunk] = []
    for i, c in enumerate(chunks):
        c = c.strip()
        if c:
            out.append(Chunk(text=c, index=len(out), source=source))
    return out
