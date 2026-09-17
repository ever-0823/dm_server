"""纵向合并类别的跨图续表回归，不调用数据库或模型。"""

from copy import deepcopy

import pytest

from app.core.exceptions import AppException
from app.knowledge.table_merge import build_merge, extract_tables


def test_merge_repeats_categories_without_changing_source():
    """不带电检查续页不含表头，展开类别时保留页码且不混入上电检查。"""
    first = ("<table><tr><th>类型</th><th>要点</th></tr>"
             '<tr><td rowspan="2">整机外观</td><td>检查外观</td></tr>'
             "<tr><td>检查镜片</td></tr></table>")
    second = ('<table><tr><td rowspan="3">电池包</td><td>检查接口</td></tr>'
              "<tr><td>检查安装</td></tr><tr><td>检查损伤</td></tr>"
              '<tr><td rowspan="2">遥控器</td><td>检查摇杆</td></tr>'
              "<tr><td>检查按键</td></tr></table>"
              "<h3>上电检查</h3><table><tr><td>类型</td><td>要点</td></tr>"
              "<tr><td>电池</td><td>确认电量</td></tr></table>")
    snapshot = {"id": 1, "metadata": {}, "images": [
        {"id": 1, "original_name": "前页", "page_number": 10, "markdown_content": first},
        {"id": 2, "original_name": "续页", "page_number": 11, "markdown_content": second},
    ]}
    before = deepcopy(snapshot)
    result = build_merge(snapshot, {"title": "不带电检查", "selections": [
        {"image_id": 1, "table_index": 0, "skip_header": True},
        {"image_id": 2, "table_index": 0, "skip_header": False},
    ]})
    group = result["group"]
    assert group["headers"] == ["类型", "要点"]
    assert group["rows"] == [
        ["整机外观", "检查外观"], ["整机外观", "检查镜片"],
        ["电池包", "检查接口"], ["电池包", "检查安装"], ["电池包", "检查损伤"],
        ["遥控器", "检查摇杆"], ["遥控器", "检查按键"],
    ]
    assert [s["page_number"] for s in group["row_sources"]] == [10, 10, 11, 11, 11, 11, 11]
    assert snapshot == before
    assert "确认电量" not in group["markdown"]
    assert second[result["parts"][1][1]["end"]:].startswith("<h3>上电检查")


def test_multiple_vertical_groups_keep_original_column_coordinates():
    """不同列同时跨行时仍按坐标展开，显式空值保留而不猜测填充。"""
    tables = extract_tables(
        '<table><tr><td rowspan="2">A</td><td>B</td><td rowspan="2">C</td></tr>'
        '<tr><td></td></tr></table>'
    )
    assert tables[0]["error"] == ""
    assert tables[0]["rows"] == [["A", "B", "C"], ["A", "", "C"]]


@pytest.mark.parametrize("raw,message", [
    ('<table><tr><td colspan="2">标题</td></tr></table>', "横向合并"),
    ('<table><tr><td rowspan="3">A</td><td>B</td></tr></table>', "实际行数"),
    ('<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>', "缺失单元格"),
])
def test_ambiguous_structure_is_rejected_without_index_shift(raw, message):
    """不完整结构不得自动补猜，同时必须保留选表序号。"""
    tables = extract_tables(raw + "<table><tr><td>X</td><td>Y</td></tr></table>")
    assert message in tables[0]["error"]
    assert tables[1]["index"] == 1
    assert tables[1]["rows"] == [["X", "Y"]]


def test_rowspanned_data_cannot_be_mistaken_for_header():
    """用户错误勾选续页数据为表头时提供明确错误。"""
    snapshot = {"metadata": {}, "images": [
        {"id": i, "original_name": str(i), "page_number": i,
         "markdown_content": '<table><tr><td rowspan="2">类别</td><td>内容</td></tr>'
                             '<tr><td>另一项</td></tr></table>'}
        for i in (1, 2)
    ]}
    with pytest.raises(AppException, match="单层表头"):
        build_merge(snapshot, {"title": "测试", "selections": [
            {"image_id": 1, "table_index": 0, "skip_header": True},
            {"image_id": 2, "table_index": 0, "skip_header": False},
        ]})
