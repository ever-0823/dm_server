from app.routers.devices import decode_csv_content


def test_decode_csv_content_supports_chinese_encodings() -> None:
    """中文 CSV 在 UTF-8 和 GB18030 编码下都应得到相同内容。"""
    content = "设备编号,设备名称\nDEV-001,生产设备\n"

    assert decode_csv_content(content.encode("utf-8-sig")) == content
    assert decode_csv_content(content.encode("gb18030")) == content


if __name__ == "__main__":
    test_decode_csv_content_supports_chinese_encodings()
