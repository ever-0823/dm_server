"""验证 GLM-OCR 表格 HTML、合并结构及字段映射，不加载推理模型。"""

from io import BytesIO

from PIL import Image

from app.ocr import glm_ocr as ocr
from app.knowledge import table_merge
from app.core.exceptions import AppException


def test_html_table_keeps_row_and_column_spans():
    """GLM-OCR 的 HTML 合并行列应转换为稳定的网格坐标。"""
    cells = ocr._html_cells(
        "<table><tr><th colspan='3'>审批表</th></tr>"
        "<tr><td rowspan='2'>相关信息</td><td>单位</td><td>甲</td></tr>"
        "<tr><td>联系人</td><td>乙</td></tr></table>"
    )
    assert [(item["row"], item["column"], item["row_span"], item["column_span"]) for item in cells] == [
        (0, 0, 1, 3),
        (1, 0, 2, 1),
        (1, 1, 1, 1),
        (1, 2, 1, 1),
        (2, 1, 1, 1),
        (2, 2, 1, 1),
    ]


def test_table_text_is_searchable():
    """HTML 表格进入知识库前应变成保留行关系的普通文本。"""
    assert ocr._table_text("<table><tr><td>设备</td><td>状态</td></tr></table>") == "设备 | 状态"


def test_markdown_table_is_normalized_and_restored_in_layout_order() -> None:
    """SDK 漏写表格时应补回标准 HTML，并去掉裁剪图引用。"""
    table = "<table style='width:100%'><tr><td rowspan='2'>参数</td><td>型号A</td></tr><tr><td>7</td></tr></table>"
    pages = [[
        {"index": 0, "label": "table", "content": table},
        {"index": 1, "label": "image", "content": "", "image_path": "imgs/device.jpg"},
    ]]
    markdown = ocr._normalize_markdown_tables("![Image](imgs/device.jpg)", pages)
    assert "参数" in markdown
    assert "![Image]" not in markdown
    assert "style=" not in markdown
    assert 'rowspan="2"' in markdown
    assert markdown.count("</table>") == 1


def test_parse_image_returns_markdown_and_regions(monkeypatch) -> None:
    """统一识别入口应返回可编辑 Markdown 和后台区域坐标。"""
    image = BytesIO()
    Image.new("RGB", (100, 100), "white").save(image, format="PNG")
    monkeypatch.setattr(
        ocr,
        "_parse_result",
        lambda _content: (
            [[{
                "label": "table",
                "content": (
                    "<table><tr><td colspan='3'>审批表</td></tr>"
                    "<tr><td rowspan='1'>相关信息</td><td>单位</td><td>甲</td></tr></table>"
                ),
                "bbox_2d": [0, 0, 1000, 1000],
            }]],
            "<table><tr><td>参数</td><td>型号A</td></tr></table>",
        ),
    )
    result = ocr.parse_image(image.getvalue())
    assert "参数" in result["markdown"]
    assert result["regions"][0]["bbox"][-1] == [0.0, 100.0]
    assert result["provider"] in {"maas", "selfhosted"}


def test_markdown_to_text_keeps_table_cells() -> None:
    """向量化前应从 Markdown 表格提取可见单元格文字。"""
    text = ocr.markdown_to_text("# 标题\n\n<table><tr><td>设备</td><td>在线</td></tr></table>")
    assert "标题" in text
    assert "设备 | 在线" in text


def test_table_merge_removes_each_selected_header_explicitly() -> None:
    """跨图片续表必须逐张明确跳过表头，合并结果不能重复写入表头。"""
    snapshot = {
        "id": 1,
        "name": "设备参数",
        "metadata": {},
        "images": [
            {"id": 11, "original_name": "第1页.png", "page_number": 1,
             "markdown_content": "| 设备 | 状态 |\n| --- | --- |\n| A | 正常 |"},
            {"id": 12, "original_name": "第2页.png", "page_number": 2,
             "markdown_content": "| 设备 | 状态 |\n| --- | --- |\n| B | 维修 |"},
        ],
    }
    payload = {
        "title": "设备参数续表",
        "headers": [],
        "selections": [
            {"image_id": 11, "table_index": 0, "skip_header": True},
            {"image_id": 12, "table_index": 0, "skip_header": True},
        ],
    }

    result = table_merge.build_merge(snapshot, payload)

    assert result["group"]["headers"] == ["设备", "状态"]
    assert result["group"]["rows"] == [["A", "正常"], ["B", "维修"]]
    assert result["group"]["markdown"].count("| 设备 | 状态 |") == 1


def test_table_merge_rejects_unconfirmed_duplicate_header() -> None:
    """检测到重复表头但用户未确认时必须阻止入库。"""
    snapshot = {
        "id": 1,
        "name": "设备参数",
        "metadata": {},
        "images": [
            {"id": 11, "original_name": "第1页.png", "page_number": 1,
             "markdown_content": "| 设备 | 状态 |\n| --- | --- |\n| A | 正常 |"},
            {"id": 12, "original_name": "第2页.png", "page_number": 2,
             "markdown_content": "| 设备 | 状态 |\n| --- | --- |\n| B | 维修 |"},
        ],
    }
    payload = {
        "title": "设备参数续表",
        "headers": [],
        "selections": [
            {"image_id": 11, "table_index": 0, "skip_header": True},
            {"image_id": 12, "table_index": 0, "skip_header": False},
        ],
    }

    try:
        table_merge.build_merge(snapshot, payload)
    except AppException as exc:
        assert "请勾选跳过首行表头" in str(exc)
    else:
        raise AssertionError("未确认的重复表头不应被当作数据写入")


def test_table_merge_rejects_different_column_counts() -> None:
    """跨图片表格列数不同不能静默拼接，避免生成错位数据。"""
    snapshot = {
        "id": 1,
        "name": "设备参数",
        "metadata": {},
        "images": [
            {"id": 11, "original_name": "第1页.png", "page_number": 1,
             "markdown_content": "| 设备 | 状态 |\n| --- | --- |\n| A | 正常 |"},
            {"id": 12, "original_name": "第2页.png", "page_number": 2,
             "markdown_content": "| 设备 | 状态 | 备注 |\n| --- | --- | --- |\n| B | 维修 | 待检 |"},
        ],
    }
    payload = {
        "title": "设备参数续表",
        "headers": [],
        "selections": [
            {"image_id": 11, "table_index": 0, "skip_header": True},
            {"image_id": 12, "table_index": 0, "skip_header": True},
        ],
    }

    try:
        table_merge.build_merge(snapshot, payload)
    except AppException as exc:
        assert "列数与表头不同" in str(exc)
    else:
        raise AssertionError("列数不同的表格不应被合并")


def test_long_merged_table_chunks_repeat_headers(monkeypatch) -> None:
    """长续表拆分后，每个文本块都必须包含完整表头。"""
    monkeypatch.setattr(table_merge.settings, "KNOWLEDGE_CHUNK_SIZE", 55)
    group = {
        "title": "设备参数续表",
        "headers": ["参数", "型号"],
        "rows": [["长度", "690x450x300mm"], ["重量", "35kg"], ["自由度", "23"]],
        "row_sources": [
            {"image_id": 11, "page_number": 1},
            {"image_id": 12, "page_number": 2},
            {"image_id": 12, "page_number": 2},
        ],
    }

    chunks = table_merge._merged_table_chunks(group)

    assert len(chunks) > 1
    assert all("| 参数 | 型号 |" in chunk["display_content"] for chunk in chunks)
    assert all("| --- | --- |" in chunk["display_content"] for chunk in chunks)
    assert "| 长度 | 690x450x300mm |" in chunks[0]["display_content"]
    assert "型号：690x450x300mm" in chunks[0]["content"]


def test_old_table_parse_route_is_removed() -> None:
    """表格 JSON 接口必须移除，避免继续暴露已废弃能力。"""
    from fastapi.testclient import TestClient
    from app.app_factory import create_app
    from app.dependencies.auth import current_user

    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    with TestClient(app) as client:
        response = client.post("/api/knowledge/table/parse", files={"file": ("sample.png", b"image", "image/png")})
    assert response.status_code == 404
