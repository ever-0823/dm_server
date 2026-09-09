"""知识文档文本提取与切分实现。"""

import re
import tempfile
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import AppException


@lru_cache(maxsize=1)
def _document_converter():
    """复用 Docling 转换器，避免每次导入文档都重新初始化解析管线。"""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise AppException(503, "Docling 解析依赖未安装，请执行 pip install docling") from exc

    artifacts_path = Path(settings.DOCLING_ARTIFACTS_PATH)
    if not artifacts_path.is_dir():
        raise AppException(503, f"Docling 模型未下载，请先准备本地模型目录：{artifacts_path}")

    # PDF 解析固定读取本地模型，防止业务请求期间临时联网下载而超时。
    pdf_options = PdfPipelineOptions(artifacts_path=artifacts_path)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )


def extract_pages(content: bytes, suffix: str) -> list[tuple[int, str]]:
    """使用标准库读取 TXT，使用 Docling 解析结构化文档。"""
    if suffix == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise AppException(400, "TXT 编码无法识别，请使用 UTF-8 或 GB18030") from exc
        return [(1, text)]

    if suffix not in {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}:
        raise AppException(400, "仅支持 PDF、DOCX、PPTX、XLSX、HTML、TXT 文档")

    temp_path: Path | None = None
    try:
        # Docling 接收文件路径，临时文件结束后立即清理，不把用户文档留在服务器临时目录。
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        document = _document_converter().convert(str(temp_path)).document
        page_numbers = sorted(int(page_number) for page_number in (getattr(document, "pages", {}) or {}))
        if page_numbers:
            # PDF 等分页文档按页导出，继续为知识库来源保留真实页码。
            pages = [
                (page_number, document.export_to_markdown(page_no=page_number).strip())
                for page_number in page_numbers
            ]
            return [(page_number, text) for page_number, text in pages if text]

        markdown = document.export_to_markdown().strip()
        if not markdown:
            return []
        # DOCX、HTML 等没有稳定分页信息时统一作为第 1 页处理。
        return [(1, markdown)]
    except AppException:
        # 保留依赖或模型缺失的原始状态码，方便前端展示准确处理建议。
        raise
    except Exception as exc:
        raise AppException(400, f"Docling 文档解析失败：{exc}") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


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
