"""验证真实转换入口的配置、合并结构及字段映射，不加载推理模型。"""

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from app.ocr import ppocrv6 as ocr


def cell(text, row, col, rows=1, cols=1, table=0):
    """最小单元格样本，行列偏移与 Docling 保持一致。"""
    return dict(text=text, row=row, column=col, row_span=rows,
                column_span=cols, table_index=table, score=0.96)


def test_converter_and_pixel_mapping(monkeypatch):
    """无文本层时必须关闭匹配，且页面坐标必须换算为像素坐标。"""
    import docling.document_converter as dc
    from docling.datamodel.base_models import InputFormat

    captured = {}
    class Converter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(dc, "DocumentConverter", Converter)
    ocr._table_converter.cache_clear()
    try:
        ocr._table_converter()
        options = captured['format_options'][InputFormat.IMAGE].pipeline_options
        assert options.do_ocr is False
        assert options.table_structure_options.do_cell_matching is False
    finally:
        ocr._table_converter.cache_clear()
    box = BoundingBox(l=10, t=90, r=30, b=80, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert ocr._docling_bbox(box, Size(width=100, height=100), 200, 200) == [
        [20, 20], [60, 20], [60, 40], [20, 40]]


def test_nested_pairs_quotes_blanks_and_multiple_tables():
    """同行两组字段应分开，报价应为对象，末行和第二张表不能丢失。"""
    cells = [cell('审批表', 0, 0, cols=5), cell('相关信息', 1, 0, rows=2),
             cell('申请单位', 1, 1), cell('甲', 1, 2),
             cell('对方单位', 1, 3), cell('乙', 1, 4),
             cell('备注', 2, 1), cell('', 2, 2, cols=3),
             cell('价格审批', 3, 0, rows=3), cell('对方报价', 3, 1, rows=3),
             cell('订书器：395620元', 3, 2), cell('', 3, 3),
             cell('起钉器：526300元', 4, 2), cell('', 4, 3),
             cell('合计：921920元', 5, 2), cell('', 5, 3),
             cell('分管人审批', 6, 0, cols=2), cell('签名', 6, 2, cols=3),
             cell('另一表', 0, 0, cols=2, table=1),
             cell('独立字段', 1, 0, table=1), cell('独立值', 1, 1, table=1)]
    result = ocr._to_nested_business_table(cells)
    info = result['sections'][0]['fields']
    assert [(f['name'], f['value']) for f in info] == [('申请单位', '甲'), ('对方单位', '乙'), ('备注', '')]
    quote = result['sections'][1]['fields'][0]
    assert quote['name'] == '对方报价'
    assert quote['value'] == {'订书器': '395620元', '起钉器': '526300元', '合计': '921920元'}
    fields = [f for s in result['sections'] for f in s['fields']]
    assert any(f['name'] == '分管人审批' and f['value'] == '签名' for f in fields)
    assert any(f['name'] == '独立字段' and f['value'] == '独立值' for f in fields)
    assert all(f['status'] == '待确认' for f in fields)


def test_empty_structure_must_not_fall_back(monkeypatch):
    """图像有效但 Docling 网格为空时必须抛错，不能返回未分类成功结果。"""
    image = BytesIO()
    Image.new('RGB', (10, 10)).save(image, format='PNG')
    monkeypatch.setattr(ocr, '_table_converter', lambda: SimpleNamespace(
        convert=lambda _: SimpleNamespace(document=SimpleNamespace(tables=[]))))
    with pytest.raises(ocr.OcrUnavailable, match='未识别出表格单元格'):
        ocr._recognize_docling_cells(image.getvalue(), '.png', [])


def test_ocr_mapping_is_unique_and_retains_empty_cell():
    """边缘重叠的文字只归属一格，缺失置信度不能伪造成满分。"""
    cells = [{'bbox': [[0, 0], [20, 0], [20, 20], [0, 20]]},
             {'bbox': [[15, 0], [40, 0], [40, 20], [15, 20]]}]
    ocr._assign_ocr_cells(cells, [{'text': '合同', 'score': None,
                                 'bbox': [[2, 2], [19, 2], [19, 16], [2, 16]]}])
    assert [c['text'] for c in cells] == ['合同', '']
    assert all(c['score'] is None for c in cells)


def test_trailing_blank_and_vertical_value():
    """末尾空列不能把经办人值变成字段，多行金额不是子字段名。"""
    result = ocr._nested_fields([
        cell('经办人', 0, 0), cell('', 0, 1), cell('姓名', 0, 2),
        cell('核准价格', 1, 0, rows=2), cell('大写金额', 1, 1), cell('921920', 2, 1)])
    assert [(f['name'], f['value']) for f in result] == [
        ('经办人', '姓名'), ('核准价格', '大写金额\n921920')]


def test_table_api_reports_structure_failure(monkeypatch):
    """后端结构异常应到达前端，不能被成功响应隐藏。"""
    from fastapi.testclient import TestClient
    from app.app_factory import create_app
    from app.dependencies.auth import current_user

    def fail(*_args):
        raise ocr.OcrUnavailable('Docling 未识别出表格单元格')

    monkeypatch.setattr('app.routers.ocr.recognize_table', fail)
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {'username': 'tester'}
    with TestClient(app) as client:
        response = client.post('/api/knowledge/table/parse', files={'file': ('sample.png', b'image', 'image/png')})
    assert response.status_code == 503
    assert 'Docling' in response.json()['message']
