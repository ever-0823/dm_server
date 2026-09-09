import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings


class OcrUnavailable(Exception):
    """本地 OCR 引擎无法处理请求时抛出的异常。"""


# PaddleOCR 复用同一个模型实例，通过锁避免多个请求同时执行 predict。
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _ocr():
    """只加载一次 PP-OCRv6，避免每次请求都重复初始化模型。"""
    Path(settings.OCR_MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", settings.OCR_MODEL_CACHE_DIR)

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OcrUnavailable("OCR 依赖未安装，请安装 paddleocr 和 paddlepaddle") from exc

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # 文字检测像素阈值。调低可以检测更浅、更模糊的文字。
        text_det_thresh=0.3,
        # 文字框置信度阈值。调低可以保留更多疑似文字区域。
        text_det_box_thresh=0.29,
        # 当前 Windows CPU 环境的 oneDNN 不支持部分 PP-OCRv6 属性，关闭后使用普通 CPU 推理。
        enable_mkldnn=False,
    )


def recognize_text(content: bytes, suffix: str) -> list[dict]:
    """识别上传的图片，并返回统一格式的文本行。"""
    temp_path: Path | None = None
    try:
        # PaddleOCR 需要文件路径，因此先把上传内容写入临时文件。
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        with _inference_lock:
            # predict 可能返回惰性迭代器，必须在锁内完成推理与结果读取。
            results = list(_ocr().predict(str(temp_path)))
        return _to_lines(results)
    except OcrUnavailable:
        raise
    except Exception as exc:
        raise OcrUnavailable(f"OCR 识别失败：{exc}") from exc
    finally:
        # Windows 无法删除正在使用的文件，因此在识别结束后统一清理。
        if temp_path:
            temp_path.unlink(missing_ok=True)


def recognize_table(content: bytes, suffix: str) -> dict:
    """表格结构必须成功；识别失败时不再静默伪装成普通 OCR 行结果。"""
    lines = recognize_text(content, suffix)
    cells = _recognize_docling_cells(content, suffix, lines)
    structured = _to_nested_business_table(cells)
    return {
        **structured,
        "cells": cells,
        "ocr_lines": lines,
        "parser": "docling_tableformer+ppocrv6",
        # OCR 分数只表示文字识别可信程度，不代表字段映射已经人工确认。
        "need_review": True,
    }


@lru_cache(maxsize=1)
def _table_converter():
    """复用离线表格模型，禁用依赖文本层的单元格匹配。"""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption

    options = PdfPipelineOptions(
        artifacts_path=Path(settings.DOCLING_ARTIFACTS_PATH),
        do_ocr=False,
        table_structure_options=TableStructureOptions(do_cell_matching=False),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)},
    )


# 表格转换器同样复用模型；避免并发请求交叉初始化管线。
_table_lock = Lock()


def _recognize_docling_cells(content: bytes, suffix: str, lines: list[dict]) -> list[dict]:
    """保留空单元格、合并跨度，并把页面坐标还原到原图像素。"""
    from io import BytesIO
    from PIL import Image

    temp_path = None
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        with _table_lock:
            document = _table_converter().convert(str(temp_path)).document
        cells = []
        for table_index, table in enumerate(document.tables):
            page_no = table.prov[0].page_no
            size = document.pages[page_no].size
            for cell in table.data.table_cells:
                bbox = _docling_bbox(cell.bbox, size, width, height)
                cells.append({
                    "table_index": table_index,
                    "page_number": page_no,
                    "text": "",
                    "score": None,
                    "bbox": bbox,
                    "row": int(cell.start_row_offset_idx),
                    "column": int(cell.start_col_offset_idx),
                    "row_span": int(cell.row_span),
                    "column_span": int(cell.col_span),
                })
        if not cells:
            raise OcrUnavailable("Docling 未识别出表格单元格，请使用更清晰、正面的表格图片。")
        _assign_ocr_cells(cells, lines)
        if not any(cell["text"] for cell in cells):
            raise OcrUnavailable("表格结构已识别，但未匹配到文字，请检查图片与 OCR 坐标。")
        return cells
    except OcrUnavailable:
        raise
    except Exception as exc:
        raise OcrUnavailable(f"Docling 表格结构识别失败：{exc}") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def _docling_bbox(value, size, width: int, height: int) -> list[list[float]]:
    """兼容底部原点和图片 DPI，不能假定 Docling 坐标等于图片像素。"""
    if value is None:
        return []
    box = value.to_top_left_origin(size.height)
    sx, sy = width / size.width, height / size.height
    return [[box.l * sx, box.t * sy], [box.r * sx, box.t * sy],
            [box.r * sx, box.b * sy], [box.l * sx, box.b * sy]]


def _assign_ocr_cells(cells: list[dict], lines: list[dict]) -> None:
    """每个文字框只匹配重叠最大的一格，避免边界文字被重复使用。"""
    matches = [[] for _ in cells]
    for line in lines:
        points = line.get("bbox") or []
        if not points:
            continue
        left, right = min(p[0] for p in points), max(p[0] for p in points)
        top, bottom = min(p[1] for p in points), max(p[1] for p in points)
        area = max((right - left) * (bottom - top), 1)
        overlaps = []
        for cell in cells:
            box = cell["bbox"]
            if not box:
                overlaps.append(0)
                continue
            overlap = max(0, min(right, box[2][0]) - max(left, box[0][0]))
            overlap *= max(0, min(bottom, box[2][1]) - max(top, box[0][1]))
            overlaps.append(overlap / area)
        if overlaps and max(overlaps) >= 0.3:
            matches[overlaps.index(max(overlaps))].append(line)
    for cell, matched in zip(cells, matches):
        matched.sort(key=lambda line: (min(p[1] for p in line["bbox"]), min(p[0] for p in line["bbox"])))
        cell["text"] = " ".join(str(line["text"]).strip() for line in matched)
        scores = [line.get("score") for line in matched]
        cell["score"] = min(scores) if scores and all(s is not None for s in scores) else None


def _nested_fields(cells: list[dict]) -> list[dict]:
    """按横向字段对及纵向合并范围递归整理；保留空值和重复子字段。"""
    remaining = sorted(cells, key=lambda c: (c["row"], c["column"]))
    fields = []
    while remaining:
        label = remaining.pop(0)
        if not label["text"].strip():
            # 没有文字的占位格留在原始 cells 中，不虚构业务字段名。
            continue
        row_end = label["row"] + label["row_span"]
        col_start = label["column"] + label["column_span"]
        # 下一条同高度的跨行标签是右侧另一组字段的边界。
        boundaries = [c["column"] for c in remaining
                      if c["row"] == label["row"] and c["row_span"] >= label["row_span"]
                      and c["column"] >= col_start]
        if label["row_span"] > 1:
            boundary = min(boundaries) if boundaries else float("inf")
            children = [c for c in remaining if label["row"] <= c["row"] < row_end
                        and col_start <= c["column"] < boundary]
        else:
            children = []
        if children:
            nonempty = [c for c in children if c["text"].strip()]
            if (len({c["column"] for c in nonempty}) == 1
                    and not any(re.search(r"[:：]", c["text"]) for c in nonempty)):
                # 合并标签右侧只有一列内容时，例如金额大写和数字，保留多行值。
                value = "\n".join(c["text"] for c in sorted(nonempty, key=lambda c: c["row"]))
            else:
                child_fields = _nested_fields(children)
                names = [f["name"] for f in child_fields]
                # 同名字段不能相互覆盖，用列表保留它们的来源顺序。
                value = ({f["name"]: f["value"] for f in child_fields}
                         if len(names) == len(set(names)) else child_fields)
        else:
            # 印刷的“订书器：395620元”可独占一格，空白邻格仍需消耗掉。
            candidates = [c for c in remaining if c["row"] == label["row"] and c["column"] == col_start]
            children = candidates[:1]
            if children and not children[0]["text"]:
                # TableFormer 偶尔把末尾值拆成空格与有字格，只合并末尾的相邻一格。
                trailing = [c for c in remaining if c["row"] == label["row"]
                            and c["column"] >= col_start]
                if len(trailing) == 2 and trailing[1]["column"] == col_start + children[0]["column_span"]:
                    children = trailing
            value = "\n".join(c["text"] for c in children if c["text"])
        remaining = [c for c in remaining if all(c is not child for child in children)]
        name = label["text"]
        if not value:
            pair = re.split(r"[:：]", name, maxsplit=1)
            if len(pair) == 2 and pair[1].strip():
                name, value = pair[0].strip(), pair[1].strip()
        scores = [c.get("score") for c in [label, *children]]
        confidence = min(scores) if scores and all(s is not None for s in scores) else None
        fields.append({
            "name": name or "未识别字段",
            "value": value,
            "confidence": round(confidence, 2) if confidence is not None else None,
            "status": "待确认",
        })
    return fields


def _to_nested_business_table(cells: list[dict]) -> dict:
    """逐表处理合并首列，隔离多表行号并保留未分组字段。"""
    title = ""
    sections = []
    # ponytail: 首列分组适用于审批表；其他版式保持未分类，后续再增加业务模板。
    for table_id in sorted({c.get("table_index", 0) for c in cells}):
        body = [c for c in cells if c.get("table_index", 0) == table_id]
        columns = max(c["column"] + c["column_span"] for c in body)
        titles = [c for c in body if c["row"] == 0 and c["column"] == 0
                  and c["column_span"] == columns and c["row_span"] == 1]
        if titles and not title:
            title = titles[0]["text"]
        body = [c for c in body if c not in titles]
        groups = [c for c in body if c["column"] == 0 and c["row_span"] > 1]
        used = set()
        for group in groups:
            children = [c for c in body if group["row"] <= c["row"]
                        and c["row"] + c["row_span"] <= group["row"] + group["row_span"]
                        and c["column"] >= group["column_span"]]
            used.update(id(c) for c in [group, *children])
            sections.append({"name": group["text"] or "未识别分组", "fields": _nested_fields(children)})
        remaining = [c for c in body if id(c) not in used]
        if remaining:
            sections.append({"name": "未分类", "fields": _nested_fields(remaining)})
    return {"document_title": title, "sections": sections}




def _to_lines(results: list[Any]) -> list[dict]:
    """把 PaddleOCR 结果转换为文本和置信度字段。"""
    lines: list[dict] = []
    for item in results:
        data = getattr(item, "json", item)
        result = data.get("res", data) if isinstance(data, dict) else {}
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        # Paddle 返回 NumPy 数组时不能使用布尔 or 判断。
        boxes = result.get("rec_boxes")
        if boxes is None or len(boxes) == 0:
            boxes = result.get("dt_polys")
        if boxes is None:
            boxes = []

        for index, text in enumerate(texts):
            line = {"text": str(text), "score": None}
            if index < len(scores):
                line["score"] = float(scores[index])
            if index < len(boxes):
                # 坐标保留为普通列表，便于接口返回和后续写入 JSONB。
                line["bbox"] = _to_bbox(boxes[index])
            lines.append(line)
    return lines


def _to_bbox(value: Any) -> list[list[float]]:
    """把矩形或多边形坐标统一转换为四个顶点。"""
    try:
        points = value.tolist() if hasattr(value, "tolist") else value
        if len(points) == 4 and not isinstance(points[0], (list, tuple)):
            left, top, right, bottom = (float(number) for number in points)
            return [[left, top], [right, top], [right, bottom], [left, bottom]]
        return [[float(point[0]), float(point[1])] for point in points]
    except (TypeError, IndexError, ValueError):
        return []
