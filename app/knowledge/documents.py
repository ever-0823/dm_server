"""知识文档文本提取与切分实现。"""

import re

from app.core.config import settings
from app.core.exceptions import AppException
from app.ocr.glm_ocr import OcrUnavailable, parse_pages


def _pdf_page_count(content: bytes) -> int:
    """在进入耗时 OCR 前读取 PDF 页数并验证文件结构。"""
    try:
        import pymupdf
    except ImportError as exc:
        raise AppException(503, "GLM-OCR PDF 依赖未安装，请重新安装后端依赖") from exc
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            return len(document)
    except Exception as exc:
        raise AppException(400, "PDF 文件损坏或格式无法识别") from exc


def _pdf_native_text(content: bytes) -> list[str]:
    """优先读取 PDF 文本层，避免可复制文档再走一遍 OCR。"""
    try:
        import pymupdf
    except ImportError as exc:
        raise AppException(503, "GLM-OCR PDF 依赖未安装，请重新安装后端依赖") from exc
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            return [page.get_text("text").strip() for page in document]
    except Exception as exc:
        raise AppException(400, "PDF 文件损坏或格式无法识别") from exc


def extract_pages(content: bytes, suffix: str) -> list[tuple[int, str]]:
    """使用标准库读取 TXT；PDF 先取文本层，扫描页再调用 GLM-OCR。"""
    if suffix == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise AppException(400, "TXT 编码无法识别，请使用 UTF-8 或 GB18030") from exc
        return [(1, text)]

    if suffix != ".pdf":
        raise AppException(400, "仅支持 PDF、TXT 文档")

    page_count = _pdf_page_count(content)
    if page_count <= 0:
        raise AppException(400, "PDF 文件不包含可解析页面")
    if page_count > settings.GLM_OCR_MAX_PDF_PAGES:
        raise AppException(
            400,
            f"PDF 最多支持 {settings.GLM_OCR_MAX_PDF_PAGES} 页，请拆分后重新上传",
        )
    try:
        native_pages = _pdf_native_text(content)
        if all(native_pages):
            return [(page_number, text) for page_number, text in enumerate(native_pages, start=1) if text]
        pages = parse_pages(content)
    except OcrUnavailable as exc:
        raise AppException(503, str(exc)) from exc
    # 有文本层的页面保留原文字，空页才使用 OCR 结果。
    merged = []
    for page_number, native in enumerate(native_pages, start=1):
        text = native or (pages[page_number - 1].strip() if page_number - 1 < len(pages) else "")
        if text:
            merged.append((page_number, text))
    return merged


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按章节和条目切分正文，尽量让标题与对应说明保留在同一块。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("文本块大小必须大于重叠长度")

    # 只压缩多余空白，保留换行用于识别章节和编号条目。
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # PDF 目录通常独占一页；即使超过普通块大小，也要整体保留为一个可识别的目录块。
    if re.match(r"^目\s*录(?:\s|$)", normalized):
        return [normalized]

    # 中文章节、数字条目通常是问题和答案的语义边界，不在普通句号处强行拆散。
    heading_pattern = re.compile(
        r"^(?:第[一二三四五六七八九十百]+[章节部分]|[一二三四五六七八九十百]+[、.．]|\d+[、.．)])"
    )
    blocks: list[str] = []
    current_lines: list[str] = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            if current_lines:
                blocks.append("\n".join(current_lines))
                current_lines = []
            continue
        if current_lines and heading_pattern.match(line):
            blocks.append("\n".join(current_lines))
            current_lines = []
        current_lines.append(line)
    if current_lines:
        blocks.append("\n".join(current_lines))

    chunks: list[str] = []

    def split_long_block(block: str) -> list[str]:
        """超长条目仍按句号或字符边界切分，并保留少量重叠。"""
        result: list[str] = []
        start = 0
        while start < len(block):
            end = min(start + chunk_size, len(block))
            if end < len(block):
                minimum_end = start + chunk_size // 2
                sentence_end = max(
                    block.rfind("。", minimum_end, end),
                    block.rfind("；", minimum_end, end),
                    block.rfind("\n", minimum_end, end),
                )
                if sentence_end >= minimum_end:
                    end = sentence_end + 1
            result.append(block[start:end].strip())
            if end >= len(block):
                break
            start = max(end - overlap, start + 1)
        return result

    current = ""
    for block in blocks:
        if len(block) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_long_block(block))
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            # 给相邻语义块保留尾部上下文，避免边界查询丢失关键信息。
            current = f"{current[-overlap:]}\n\n{block}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current.strip())

    return chunks
