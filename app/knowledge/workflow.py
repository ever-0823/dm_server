"""知识库导入、检索和回答的工作流实现。"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import AppException
from app.knowledge import answering, documents, embedding, store


logger = logging.getLogger(__name__)


def import_document(filename: str, content_type: str, content: bytes, username: str) -> dict:
    """提取、切分并向量化文档，最后在一个事务中写入 pgvector。"""
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

    vectors = embedding.encode_documents([chunk["content"] for chunk in chunks])
    if any(len(vector) != settings.EMBEDDING_DIMENSIONS for vector in vectors):
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

    return store.save_document(filename, content_type, len(content), chunks, vectors, username)


def search(query: str, top_k: int) -> list[dict]:
    """生成查询向量并从 pgvector 返回相关文本及相邻上下文。"""
    started_at = time.perf_counter()
    query_vector = embedding.encode_query(query)
    if len(query_vector) != settings.EMBEDDING_DIMENSIONS:
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

    rows = store.search_chunks(query_vector, top_k)
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


def delete_document(document_id: int) -> bool:
    """删除文档及其全部文本块和向量。"""
    return store.delete_document(document_id)
