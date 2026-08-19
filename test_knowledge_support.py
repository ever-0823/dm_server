from app.knowledge.service import extract_pages, split_text


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


if __name__ == "__main__":
    test_split_text_keeps_overlap_and_content()
    test_extract_txt_supports_utf8_bom()
    print("ok")
