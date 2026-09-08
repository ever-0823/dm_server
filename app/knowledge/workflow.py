"""知识库导入、检索和回答的工作流实现。"""

import logging
import time
from hashlib import sha256
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.core.exceptions import AppException
from app.knowledge import answering, documents, embedding, qa, store


logger = logging.getLogger(__name__)
IMAGE_QUERY_MARKERS = (
    "这张图片",
    "这张图",
    "这幅图",
    "图片里的",
    "图片中的",
    "图片里",
    "图片中",
    "图中的",
    "图中",
    "照片里的",
    "照片中的",
    "照片里",
    "照片中",
)


def _normalize_match_text(value: str | None) -> str:
    """去掉空白后统一比较查询词，兼容 PDF 提取产生的换行和空格。"""
    return "".join(str(value or "").split())


def _prioritize_exact_matches(query: str, rows: list[dict], top_k: int) -> list[dict]:
    """把正文精确命中排在目录命中前，避免目录标题遮蔽正文内容。"""
    normalized_query = _normalize_match_text(query)
    if len(normalized_query) < 2:
        return rows[:top_k]

    exact_rows = [
        row
        for row in rows
        if normalized_query in _normalize_match_text(row.get("content"))
        or normalized_query in _normalize_match_text(row.get("context"))
    ]
    body_rows = [
        row
        for row in exact_rows
        if not _normalize_match_text(row.get("content")).startswith("目录")
    ]
    preferred_rows = body_rows or exact_rows
    preferred_ids = {id(row) for row in preferred_rows}
    remaining_rows = [row for row in rows if id(row) not in preferred_ids]
    if body_rows:
        # 已经找到正文时移除目录精确命中，避免目录条目再次混入来源和上下文。
        remaining_rows = [
            row
            for row in remaining_rows
            if not (
                _normalize_match_text(row.get("content")).startswith("目录")
                and normalized_query
                in _normalize_match_text(row.get("content") or row.get("context"))
            )
        ]
    ordered_rows = preferred_rows + remaining_rows
    return ordered_rows[:top_k]


def _extract_chunks(filename: str, content: bytes) -> tuple[list[tuple[int, str]], list[dict]]:
    """提取文档正文并按现有规则生成文本块。"""
    suffix = Path(filename).suffix.lower()
    pages = documents.extract_pages(content, suffix)
    chunks: list[dict] = []
    for page_number, page_text in pages:
        for chunk_text in documents.split_text(
            page_text,
            settings.KNOWLEDGE_CHUNK_SIZE,
            settings.KNOWLEDGE_CHUNK_OVERLAP,
        ):
            chunks.append({"page_number": page_number, "content": chunk_text})

    if not chunks:
        raise AppException(400, "文档未提取到文字，扫描版 PDF 请先进行 OCR")
    return pages, chunks


def preview_document(filename: str, content: bytes) -> dict:
    """返回提取和切分预览，不生成向量也不写入数据库。"""
    pages, chunks = _extract_chunks(filename, content)
    # 预览最多返回前 10 个文本块，避免大文档把桌面端响应撑得过大。
    preview_chunks = chunks[:10]
    return {
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "items": preview_chunks,
        "truncated": len(chunks) > len(preview_chunks),
    }


def import_document(
    filename: str,
    content_type: str,
    content: bytes,
    username: str,
    qa_split: bool = False,
) -> dict:
    """提取、切分并向量化文档，最后在一个事务中写入 pgvector。"""
    _pages, chunks = _extract_chunks(filename, content)

    if qa_split:
        # 每个普通文本块生成一组 QA，保留原页码以便检索结果继续定位来源。
        chunks = [
            {
                "page_number": chunk["page_number"],
                "content": qa.generate_qa(chunk["content"])["content"],
            }
            for chunk in chunks
        ]

    vectors = embedding.encode_documents([chunk["content"] for chunk in chunks])
    if any(len(vector) != settings.EMBEDDING_DIMENSIONS for vector in vectors):
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

    # 将用户在导入向导中选择的处理方式随文档元数据一起保存。
    processing_mode = "问答对提取" if qa_split else "正常分割"
    return store.save_document(
        filename,
        content_type,
        len(content),
        chunks,
        vectors,
        username,
        processing_mode,
    )


def import_image(
    filename: str,
    knowledge_name: str,
    content_type: str,
    image_content: bytes,
    corrected_text: str,
    ocr_lines: list[dict],
    username: str,
    document_id: int | None = None,
) -> dict:
    """新建图片知识库，或把一张 OCR 图片追加到现有知识库。"""
    final_text = corrected_text.strip()
    if not final_text:
        raise AppException(400, "OCR 校正文本不能为空")
    if document_id is None and not knowledge_name.strip():
        raise AppException(400, "知识库名称不能为空")

    display_chunks = documents.split_text(
        final_text,
        settings.KNOWLEDGE_CHUNK_SIZE,
        settings.KNOWLEDGE_CHUNK_OVERLAP,
    )
    if not display_chunks:
        raise AppException(400, "OCR 文本未生成有效文本块")

    # 检索文本附带图片文件名和来源类型，展示文本仍保持用户校正后的正文。
    prefix = (
        f"知识库名称：{knowledge_name}\n原始图片：{filename}\n"
        f"来源类型：图片 OCR\n图片大小：{len(image_content)} 字节\n内容："
    )
    chunks = [
        {
            "page_number": 1,
            "content": prefix + text,
            "display_content": text,
        }
        for text in display_chunks
    ]
    vectors = embedding.encode_documents([chunk["content"] for chunk in chunks])
    if any(len(vector) != settings.EMBEDDING_DIMENSIONS for vector in vectors):
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

    image_dir = Path(settings.UPLOAD_FOLDER) / "knowledge_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    image_path = image_dir / f"{uuid4().hex}{suffix}"
    image_path.write_bytes(image_content)
    try:
        result = store.append_image(
            document_id,
            knowledge_name.strip(),
            filename,
            content_type,
            len(image_content),
            final_text,
            ocr_lines,
            sha256(image_content).hexdigest(),
            str(image_path.resolve()),
            chunks,
            vectors,
            username,
        )
        # 重复图片不需要保留第二份物理文件，已有知识数据保持不变。
        if result.get("duplicate"):
            image_path.unlink(missing_ok=True)
        return result
    except Exception:
        # 数据库写入失败时清理刚保存的原图，避免留下无主文件。
        image_path.unlink(missing_ok=True)
        raise


def search(query: str, top_k: int) -> list[dict]:
    """生成查询向量并从 pgvector 返回相关文本及相邻上下文。"""
    started_at = time.perf_counter()
    image_query = any(marker in query for marker in IMAGE_QUERY_MARKERS)
    clean_query = query
    if image_query:
        # 去掉口语化图片指代词，让 Embedding 聚焦真正需要检索的内容。
        for marker in IMAGE_QUERY_MARKERS:
            clean_query = clean_query.replace(marker, "")
        clean_query = clean_query.strip(" ，。？?") or query
    query_vector = embedding.encode_query(clean_query)
    if len(query_vector) != settings.EMBEDDING_DIMENSIONS:
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

    # 多取少量候选，给正文关键词命中一次重新排序的空间，最终仍只返回 top_k 条。
    candidate_rows = store.search_chunks(query_vector, min(top_k * 4, 80))
    rows = _prioritize_exact_matches(clean_query, candidate_rows, len(candidate_rows))
    if image_query:
        # 用户明确询问图片时优先展示 OCR 图片知识，分数仍决定同类来源内部顺序。
        rows.sort(key=lambda row: row.get("source_type") not in {"image", "image_set"})
    rows = rows[:top_k]
    logger.info(
        "知识库检索完成 query_length=%d top_k=%d result_count=%d elapsed_ms=%.1f",
        len(query),
        top_k,
        len(rows),
        (time.perf_counter() - started_at) * 1000,
    )
    return rows


def stream_answer(query: str, top_k: int) -> Iterator[dict]:
    """检索知识并返回 Ollama 逐段生成的答案事件。"""
    return answering.stream_answer(query, search(query, top_k))


def ask(query: str, top_k: int) -> dict:
    """汇总流式事件，兼容原有一次性问答入口。"""
    return answering.collect_answer(stream_answer(query, top_k))


def list_documents() -> list[dict]:
    """返回已导入的全部知识文档。"""
    return store.list_documents()


def get_document_chunks(document_id: int, page: int, page_size: int) -> dict | None:
    """分页返回指定知识文档的元数据和文本块。"""
    return store.get_document_chunks(document_id, page, page_size)


def get_source_image(document_id: int) -> dict | None:
    """返回图片知识来源对应的原图文件信息。"""
    return store.get_source_image(document_id)


def list_document_images(document_id: int) -> list[dict]:
    """返回指定知识库关联的全部 OCR 图片。"""
    return store.list_document_images(document_id)


def get_document_image(document_id: int, image_id: int) -> dict | None:
    """返回知识库中指定的一张 OCR 原图。"""
    return store.get_source_image(document_id, image_id)


def delete_document(document_id: int) -> bool:
    """删除文档及其全部文本块和向量。"""
    return store.delete_document(document_id)
