from app.knowledge.documents import extract_pages, split_text


def test_split_text_keeps_overlap_and_content() -> None:
    # 小尺寸参数用于快速验证切分不会丢失正文首尾。
    text = "第一段设备说明。\n第二段维护步骤。\n第三段故障处理。"
    chunks = split_text(text, chunk_size=12, overlap=3)
    assert len(chunks) > 1
    assert chunks[0].startswith("第一段")
    assert chunks[-1].endswith("故障处理。")


def test_extract_txt_supports_utf8_bom() -> None:
    # 系统导出的带 BOM 中文 TXT 应保持原文内容。
    pages = extract_pages("设备知识".encode("utf-8-sig"), ".txt")
    assert pages == [(1, "设备知识")]


def test_split_text_keeps_numbered_answer_with_question() -> None:
    # 编号问题和其后回答应优先落在同一文本块，避免只检索到问题标题。
    text = "1. 维护设备前要做什么？\n必须先切断电源并悬挂警示牌。\n2. 故障如何处理？\n先检查电源。"
    chunks = split_text(text, chunk_size=80, overlap=10)
    assert any("维护设备前要做什么" in chunk and "切断电源" in chunk for chunk in chunks)


if __name__ == "__main__":
    test_split_text_keeps_overlap_and_content()
    test_extract_txt_supports_utf8_bom()
    print("ok")
