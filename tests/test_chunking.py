from nebula.chunking import split_text


def test_split_short_text_single_chunk():
    out = split_text("这是一段很短的文本。", size=100, overlap=10, source="s")
    assert len(out) == 1 and out[0].source == "s"


def test_split_respects_size():
    text = "段落一。\n\n" + "内容" * 500 + "\n\n段落二。"
    out = split_text(text, size=200, overlap=20, source="doc")
    assert len(out) > 1
    assert all(len(c.text) <= 200 + 1 for c in out)


def test_split_empty():
    assert split_text("", source="x") == []
    assert split_text("   \n  ", source="x") == []


def test_index_monotonic():
    out = split_text("a。\n\n b。\n\n c。", size=5, overlap=1, source="s")
    assert [c.index for c in out] == list(range(len(out)))
