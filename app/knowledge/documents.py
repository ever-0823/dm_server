"""知识文档文本提取与切分实现。"""

import io
import re

from app.core.exceptions import AppException


def extract_pages(content: bytes, suffix: str) -> list[tuple[int, str]]:
    """从 TXT 或可复制文本的 PDF 中提取按页组织的正文。"""
    if suffix == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise AppException(400, "TXT 编码无法识别，请使用 UTF-8 或 GB18030") from exc
        return [(1, text)]

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AppException(503, "PDF 解析依赖 pypdf 未安装") from exc

    try:
        reader = PdfReader(io.BytesIO(content))
        # 页码从 1 开始，便于前端直接展示来源位置。
        return [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1)]
    except Exception as exc:
        raise AppException(400, f"PDF 文件解析失败：{exc}") from exc


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按章节和条目切分正文，尽量让标题与对应说明保留在同一块。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("文本块大小必须大于重叠长度")

    # 只压缩多余空白，保留换行用于识别章节和编号条目。
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

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
