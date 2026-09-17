"""GLM-OCR 官方 API 与本地 Ollama 的统一应用层适配。"""

import json
import os
import re
from functools import lru_cache
from html import escape, unescape
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from PIL import Image

from app.core.config import settings


class OcrUnavailable(Exception):
    """GLM-OCR 无法完成识别时抛出的统一异常。"""


# ponytail: 本地 CPU 推理串行运行；需要并发时再改为独立 OCR 服务队列。
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _parser():
    """按当前配置创建并缓存一个 GLM-OCR 解析器，不在官方和本地之间自动切换。"""
    try:
        from glmocr import GlmOcr
    except ImportError as exc:
        raise OcrUnavailable("GLM-OCR 依赖未安装，请重新安装后端依赖") from exc

    if settings.GLM_OCR_MODE == "maas":
        if not settings.ZHIPU_API_KEY:
            raise OcrUnavailable("官方 GLM-OCR 模式缺少 ZHIPU_API_KEY")
        try:
            return GlmOcr(
                mode="maas",
                api_key=settings.ZHIPU_API_KEY,
                api_url=settings.GLM_OCR_API_URL,
                model=settings.GLM_OCR_API_MODEL,
                timeout=settings.GLM_OCR_TIMEOUT,
            )
        except Exception as exc:
            raise OcrUnavailable(f"官方 GLM-OCR 初始化失败：{exc}") from exc

    model_cache = Path(settings.GLM_OCR_MODEL_CACHE_DIR)
    model_cache.mkdir(parents=True, exist_ok=True)
    # 本地布局模型使用项目缓存，避免启动时重复访问 Hugging Face。
    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    endpoint = urlparse(settings.OLLAMA_BASE_URL)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise OcrUnavailable("OLLAMA_BASE_URL 配置无效")
    try:
        return GlmOcr(
            mode="selfhosted",
            model=settings.GLM_OCR_MODEL,
            ocr_api_host=endpoint.hostname,
            ocr_api_port=endpoint.port or (443 if endpoint.scheme == "https" else 80),
            layout_device=settings.GLM_OCR_LAYOUT_DEVICE,
            _dotted={
                "pipeline.ocr_api.api_path": "/api/generate",
                "pipeline.ocr_api.api_mode": "ollama_generate",
                "pipeline.ocr_api.connect_timeout": settings.GLM_OCR_CONNECT_TIMEOUT,
                "pipeline.ocr_api.request_timeout": settings.GLM_OCR_TIMEOUT,
                "pipeline.ocr_api.retry_max_attempts": 0,
                "pipeline.max_workers": 1,
                "pipeline.page_loader.max_tokens": 4096,
                "pipeline.page_loader.pdf_dpi": 200,
                "pipeline.page_loader.pdf_max_pages": settings.GLM_OCR_MAX_PDF_PAGES,
                "pipeline.layout.batch_size": 1,
                "pipeline.layout.workers": 1,
            },
        )
    except Exception as exc:
        raise OcrUnavailable(f"本地 GLM-OCR 初始化失败：{exc}") from exc


class _TableHTMLParser(HTMLParser):
    """把 GLM-OCR 表格 HTML 转成保留合并行列的单元格。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[dict]] = []
        self._table_depth = 0
        self._row = -1
        self._next_column = 0
        self._occupied: set[tuple[int, int]] = set()
        self._cell: dict | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self.tables.append([])
                self._row = -1
                self._next_column = 0
                self._occupied = set()
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row += 1
            self._next_column = 0
            return
        if tag not in {"td", "th"}:
            if self._cell is not None and tag in {"br", "div", "p"}:
                self._cell_parts.append("\n")
            return
        if self._row < 0:
            self._row = 0
        while (self._row, self._next_column) in self._occupied:
            self._next_column += 1
        attributes = {name.lower(): value for name, value in attrs}
        row_span = _positive_span(attributes.get("rowspan"))
        column_span = _positive_span(attributes.get("colspan"))
        column = self._next_column
        for row in range(self._row, self._row + row_span):
            for occupied_column in range(column, column + column_span):
                self._occupied.add((row, occupied_column))
        self._cell = {
            "text": "",
            "row": self._row,
            "column": column,
            "row_span": row_span,
            "column_span": column_span,
        }
        self._cell_parts = []
        self._next_column = column + column_span

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._table_depth == 1 and self._cell is not None:
            self._cell["text"] = re.sub(r"\s+", " ", "".join(self._cell_parts)).strip()
            self.tables[-1].append(self._cell)
            self._cell = None
            self._cell_parts = []
            return
        if tag == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell_parts.append(data)


def _positive_span(value: str | None) -> int:
    """将非法或缺失的 HTML 合并跨度恢复为 1。"""
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _html_cells(content: str, page_number: int = 1, table_offset: int = 0) -> list[dict]:
    """解析一个区域中的全部 HTML 表格，并补充分页和表格编号。"""
    parser = _TableHTMLParser()
    parser.feed(content)
    cells: list[dict] = []
    for local_table_index, table in enumerate(parser.tables):
        for cell in table:
            cells.append({
                **cell,
                "table_index": table_offset + local_table_index,
                "page_number": page_number,
            })
    return cells


def _table_html(content: str) -> str:
    """把模型表格生成为 Qt 可稳定显示的标准 HTML。"""
    cells = _html_cells(content)
    if not cells:
        return content
    tables = []
    for table_index in sorted({cell["table_index"] for cell in cells}):
        rows: dict[int, list[dict]] = {}
        for cell in cells:
            if cell["table_index"] == table_index:
                rows.setdefault(cell["row"], []).append(cell)
        parts = ['<table border="1" cellspacing="0" cellpadding="6">']
        for row_index in sorted(rows):
            parts.append("<tr>")
            for cell in sorted(rows[row_index], key=lambda item: item["column"]):
                spans = ""
                if cell["row_span"] > 1:
                    spans += f' rowspan="{cell["row_span"]}"'
                if cell["column_span"] > 1:
                    spans += f' colspan="{cell["column_span"]}"'
                parts.append(f'<td{spans}>{escape(cell["text"])}</td>')
            parts.append("</tr>")
        parts.append("</table>")
        tables.append("\n".join(parts))
    return "\n\n".join(tables)


def _normalize_markdown_tables(markdown: str, pages: list[list[dict]]) -> str:
    """规范表格 HTML，并去掉裁剪图引用，原图由知识库单独保存。"""
    missing_table = False
    for page in pages:
        for region in page:
            content = str(region.get("content") or "").strip()
            is_table = str(region.get("label") or "").lower() == "table" or "<table" in content.lower()
            if not is_table or not content:
                continue
            normalized = _table_html(content)
            if content in markdown:
                markdown = markdown.replace(content, normalized, 1)
            else:
                missing_table = True
    if missing_table:
        markdown = "\n\n".join(
            content
            for page in pages
            for region in sorted(page, key=lambda item: item.get("index", 0))
            if (content := (
                _table_html(str(region.get("content") or "").strip())
                if str(region.get("label") or "").lower() == "table" or "<table" in str(region.get("content") or "").lower()
                else str(region.get("content") or "").strip()
            ))
            and str(region.get("label") or "").lower() != "image"
        )
    markdown = re.sub(r"!\[[^\]]*]\([^\n)]*\)", "", markdown)
    markdown = re.sub(r"<img\b[^>]*>", "", markdown, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def _table_text(content: str) -> str:
    """把 HTML 表格转换为按行排列的可检索文本。"""
    cells = _html_cells(content)
    if not cells:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content)).strip()
    rows: dict[tuple[int, int], list[dict]] = {}
    for cell in cells:
        rows.setdefault((cell["table_index"], cell["row"]), []).append(cell)
    return "\n".join(
        " | ".join(cell["text"] for cell in sorted(row, key=lambda item: item["column"]))
        for _key, row in sorted(rows.items())
    )


def markdown_to_text(markdown: str) -> str:
    """提取 Markdown 可见文字，供切分和向量化使用。"""
    # 富文本编辑器导出的 HTML 带样式头，不能把 CSS 作为知识正文入库。
    markdown = re.sub(r"<(head|style|script)\b[^>]*>.*?</\1>", "", markdown,
                      flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<table\b.*?</table>",
        lambda match: "\n" + _table_text(match.group(0)) + "\n",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"!\[[^\]]*]\([^\n)]*\)", "", text)
    text = re.sub(r"\[([^]]+)]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}(?:#{1,6}|[-*+] |\d+[.)] )", "", text, flags=re.MULTILINE)
    text = re.sub(r"[`*_~]", "", text)
    return unescape(re.sub(r"\n{3,}", "\n\n", text)).strip()


def _bbox_points(value, width: int, height: int) -> list[list[float]]:
    """把 SDK 的 0-1000 矩形换算为原图四点像素坐标。"""
    try:
        left, top, right, bottom = (float(number) for number in value)
    except (TypeError, ValueError):
        return []
    sx, sy = width / 1000, height / 1000
    return [
        [round(left * sx, 2), round(top * sy, 2)],
        [round(right * sx, 2), round(top * sy, 2)],
        [round(right * sx, 2), round(bottom * sy, 2)],
        [round(left * sx, 2), round(bottom * sy, 2)],
    ]


def _regions(pages: list[list[dict]], width: int, height: int) -> list[dict]:
    """将 SDK 布局区域转换为后台保存的坐标格式，前端默认不展示。"""
    result = []
    for page_number, page in enumerate(pages, start=1):
        for region in page:
            content = str(region.get("content") or "").strip()
            label = str(region.get("native_label") or region.get("label") or "text")
            item = {
                "text": markdown_to_text(content) if content else "",
                "label": label,
                "page_number": page_number,
                "score": None,
            }
            bbox = _bbox_points(region.get("bbox_2d"), width, height)
            if bbox:
                item["bbox"] = bbox
            if item["text"] or label == "image":
                result.append(item)
    return result


def _parse_result(content: bytes) -> tuple[list[list[dict]], str]:
    """调用当前模型提供方，并返回统一分页结果和 Markdown。"""
    try:
        with _inference_lock:
            result = _parser().parse(content, save_layout_visualization=False)
        pages = json.loads(result.json_result) if isinstance(result.json_result, str) else result.json_result
        if not isinstance(pages, list):
            raise TypeError("json_result 不是分页列表")
        normalized_pages = [page if isinstance(page, list) else [] for page in pages]
        return normalized_pages, _normalize_markdown_tables(str(result.markdown_result or ""), normalized_pages)
    except OcrUnavailable:
        raise
    except Exception as exc:
        message = str(exc)
        provider = "官方 GLM-OCR" if settings.GLM_OCR_MODE == "maas" else "本地 GLM-OCR"
        if "connection" in message.lower() or "refused" in message.lower():
            message = "无法连接模型服务"
        elif "model" in message.lower() and "not found" in message.lower():
            message = f"模型 {settings.GLM_OCR_MODEL} 未安装"
        raise OcrUnavailable(f"{provider} 识别失败：{message}") from exc


def parse_pages(content: bytes) -> list[str]:
    """解析 PDF 或图片字节，并返回每页可检索文本。"""
    pages, markdown = _parse_result(content)
    if not pages:
        text = markdown_to_text(markdown)
        return [text] if text else []
    result = []
    for page in pages:
        parts = []
        for region in sorted(page, key=lambda item: item.get("index", 0)):
            content_text = str(region.get("content") or "").strip()
            if not content_text or str(region.get("label") or "").lower() == "image":
                continue
            if str(region.get("label") or "").lower() == "table" or "<table" in content_text.lower():
                parts.append(_table_text(content_text))
            else:
                parts.append(markdown_to_text(content_text))
        result.append("\n\n".join(part for part in parts if part).strip())
    return result


def parse_image(content: bytes, _suffix: str = "") -> dict:
    """一次识别图片，返回可编辑 Markdown 和后台区域坐标。"""
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
    except Exception as exc:
        raise OcrUnavailable("上传内容不是有效图片") from exc
    pages, markdown = _parse_result(content)
    if not markdown:
        raise OcrUnavailable("GLM-OCR 未识别到可用内容")
    return {
        "markdown": markdown,
        "regions": _regions(pages, width, height),
        "provider": settings.GLM_OCR_MODE,
    }
