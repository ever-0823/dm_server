"""知识文档和向量的 pgvector 持久化实现。"""

import logging
import time
from threading import Lock

from app.core.config import settings
from app.core.exceptions import AppException


logger = logging.getLogger(__name__)

# 首次访问知识库时再初始化表，后端可在 pgvector 暂时离线时正常启动。
_schema_lock = Lock()
_schema_ready = False


def _connect(register_types: bool = True):
    """创建数据库连接，并在扩展已存在时注册向量字段适配器。"""
    try:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise AppException(503, "知识库依赖未安装，请先安装 requirements.txt") from exc

    try:
        connection = psycopg.connect(
            host=settings.VECTOR_DB_HOST,
            port=settings.VECTOR_DB_PORT,
            dbname=settings.VECTOR_DB_NAME,
            user=settings.VECTOR_DB_USER,
            password=settings.VECTOR_DB_PASSWORD,
            row_factory=dict_row,
        )
        # 首次建表前数据库尚无 vector 类型，此时必须先跳过类型注册。
        if register_types:
            register_vector(connection)
        return connection
    except Exception as exc:
        raise AppException(503, f"向量数据库连接失败：{exc}") from exc


def ensure_schema() -> None:
    """创建知识库表和向量索引；同一进程只执行一次。"""
    global _schema_ready
    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return
        dimensions = settings.EMBEDDING_DIMENSIONS
        if dimensions <= 0:
            raise AppException(500, "EMBEDDING_DIMENSIONS 必须大于 0")

        with _connect(register_types=False) as connection, connection.cursor() as cursor:
            # vector 扩展和表使用 IF NOT EXISTS，重复启动不会覆盖已有知识数据。
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id BIGSERIAL PRIMARY KEY,
                    original_name VARCHAR(255) NOT NULL,
                    content_type VARCHAR(100) NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_by VARCHAR(100) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dimensions}) NOT NULL,
                    UNIQUE(document_id, chunk_index)
                )
                """
            )
            # HNSW 适合知识库持续检索，无需像 IVFFlat 一样预先训练索引。
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_hnsw_idx
                ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        _schema_ready = True


def save_document(
    filename: str,
    content_type: str,
    size_bytes: int,
    chunks: list[dict],
    vectors: list[list[float]],
    username: str,
) -> dict:
    """在一个事务中保存文档元数据、文本块和向量。"""
    ensure_schema()
    from pgvector import Vector

    with _connect() as connection, connection.cursor() as cursor:
        # 文档和全部文本块在同一事务写入，任一块失败时不会留下半成品。
        cursor.execute(
            """
            INSERT INTO knowledge_documents
                (original_name, content_type, size_bytes, chunk_count, created_by)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, original_name, chunk_count, created_at
            """,
            (filename, content_type, size_bytes, len(chunks), username),
        )
        document = cursor.fetchone()
        cursor.executemany(
            """
            INSERT INTO knowledge_chunks
                (document_id, chunk_index, page_number, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (document["id"], index, chunk["page_number"], chunk["content"], Vector(vectors[index]))
                for index, chunk in enumerate(chunks)
            ],
        )
    return document


def search_chunks(query_vector: list[float], top_k: int) -> list[dict]:
    """使用余弦距离查询相关文本块及其相邻上下文。"""
    ensure_schema()
    from pgvector import Vector

    database_started_at = time.perf_counter()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            WITH ranked AS (
                SELECT c.id AS chunk_id,
                       c.document_id,
                       c.chunk_index,
                       d.original_name,
                       c.page_number,
                       c.content,
                       1 - (c.embedding <=> %s) AS score,
                       ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s) AS result_rank
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.id = c.document_id
            )
            SELECT r.chunk_id,
                   r.document_id,
                   r.original_name,
                   r.page_number,
                   r.content,
                   COALESCE(
                       (
                           SELECT STRING_AGG(context_chunk.content, E'\n\n' ORDER BY context_chunk.chunk_index)
                           FROM knowledge_chunks context_chunk
                           WHERE context_chunk.document_id = r.document_id
                             AND context_chunk.chunk_index BETWEEN r.chunk_index - 1 AND r.chunk_index + 1
                       ),
                       r.content
                   ) AS context,
                   r.score
            FROM ranked r
            WHERE r.result_rank <= %s
            ORDER BY r.result_rank
            """,
            (Vector(query_vector), Vector(query_vector), top_k),
        )
        rows = cursor.fetchall()

    logger.info(
        "pgvector 查询完成 top_k=%d result_count=%d elapsed_ms=%.1f",
        top_k,
        len(rows),
        (time.perf_counter() - database_started_at) * 1000,
    )
    # psycopg 可能返回 Decimal，统一转成 JSON 可序列化的 float。
    for row in rows:
        row["score"] = float(row["score"])
    return rows


def list_documents() -> list[dict]:
    """按上传时间倒序返回知识文档。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, original_name, content_type, size_bytes,
                   chunk_count, created_by, created_at
            FROM knowledge_documents
            ORDER BY id DESC
            """
        )
        return cursor.fetchall()


def delete_document(document_id: int) -> bool:
    """删除文档；外键级联同步删除全部向量块。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM knowledge_documents WHERE id = %s", (document_id,))
        return cursor.rowcount > 0
