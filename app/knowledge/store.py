"""知识文档和向量的 pgvector 持久化实现。"""

import json
import logging
import time
from pathlib import Path
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
            # 数据库不可达时快速返回明确错误，避免知识库页面长时间卡住。
            connect_timeout=5,
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
                    processing_mode VARCHAR(30) NOT NULL DEFAULT '正常分割',
                    source_type VARCHAR(30) NOT NULL DEFAULT 'document',
                    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source_path TEXT,
                    size_bytes BIGINT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_by VARCHAR(100) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # 为已经存在的知识文档表补充处理模式字段，旧数据按普通分割兼容。
            cursor.execute(
                """
                ALTER TABLE knowledge_documents
                ADD COLUMN IF NOT EXISTS processing_mode VARCHAR(30) NOT NULL DEFAULT '正常分割'
                """
            )
            # 图片来源需要保存原图路径和 OCR 坐标，历史文档继续按普通文档处理。
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS source_type VARCHAR(30) NOT NULL DEFAULT 'document'")
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS source_path TEXT")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_images (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    image_index INTEGER NOT NULL,
                    original_name VARCHAR(255) NOT NULL,
                    content_type VARCHAR(100) NOT NULL,
                    source_path TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    corrected_text TEXT NOT NULL,
                    ocr_lines JSONB NOT NULL DEFAULT '[]'::jsonb,
                    content_hash VARCHAR(64),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(document_id, image_index),
                    UNIQUE(document_id, content_hash)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    source_image_id BIGINT REFERENCES knowledge_images(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    display_content TEXT,
                    embedding vector({dimensions}) NOT NULL,
                    UNIQUE(document_id, chunk_index)
                )
                """
            )
            # 检索文本可带元数据，展示时仍使用用户校正后的原文。
            cursor.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS display_content TEXT")
            cursor.execute(
                """
                ALTER TABLE knowledge_chunks
                ADD COLUMN IF NOT EXISTS source_image_id BIGINT REFERENCES knowledge_images(id) ON DELETE CASCADE
                """
            )
            # 把旧版单图片知识迁移为图片子记录，保留原图和 OCR 坐标数据。
            cursor.execute(
                """
                INSERT INTO knowledge_images
                    (document_id, image_index, original_name, content_type, source_path,
                     size_bytes, corrected_text, ocr_lines)
                SELECT document.id,
                       1,
                       COALESCE(NULLIF(document.source_metadata->>'original_filename', ''), document.original_name),
                       document.content_type,
                       document.source_path,
                       document.size_bytes,
                       COALESCE(
                           (
                               SELECT STRING_AGG(
                                   COALESCE(chunk.display_content, chunk.content),
                                   E'\n\n' ORDER BY chunk.chunk_index
                               )
                               FROM knowledge_chunks AS chunk
                               WHERE chunk.document_id = document.id
                           ),
                           ''
                       ),
                       COALESCE(document.source_metadata->'ocr_lines', '[]'::jsonb)
                FROM knowledge_documents AS document
                WHERE document.source_type IN ('image', 'image_set')
                  AND document.source_path IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM knowledge_images AS image
                      WHERE image.document_id = document.id
                        AND image.source_path = document.source_path
                  )
                """
            )
            # 旧图片生成的文本块统一关联到迁移后的第一张来源图片。
            cursor.execute(
                """
                UPDATE knowledge_chunks AS chunk
                SET source_image_id = image.id
                FROM knowledge_documents AS document
                JOIN knowledge_images AS image
                  ON image.document_id = document.id
                 AND image.source_path = document.source_path
                WHERE chunk.document_id = document.id
                  AND chunk.source_image_id IS NULL
                  AND document.source_type IN ('image', 'image_set')
                """
            )
            # 兼容处理模式字段上线前已完成 QA 拆分的文档，避免历史数据被误显示为普通分割。
            cursor.execute(
                """
                UPDATE knowledge_documents AS document
                SET processing_mode = '问答对提取'
                WHERE document.processing_mode = '正常分割'
                  AND EXISTS (
                      SELECT 1
                      FROM knowledge_chunks AS chunk
                      WHERE chunk.document_id = document.id
                        AND chunk.content LIKE '问题：%'
                        AND chunk.content LIKE '%答案：%'
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
    processing_mode: str = "正常分割",
    source_type: str = "document",
    source_metadata: dict | None = None,
    source_path: str | None = None,
) -> dict:
    """在一个事务中保存文档元数据、文本块和向量。"""
    ensure_schema()
    from pgvector import Vector

    with _connect() as connection, connection.cursor() as cursor:
        # 文档和全部文本块在同一事务写入，任一块失败时不会留下半成品。
        cursor.execute(
            """
            INSERT INTO knowledge_documents
                (original_name, content_type, processing_mode, source_type, source_metadata,
                 source_path, size_bytes, chunk_count, created_by)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id, original_name, processing_mode, source_type, source_metadata,
                      source_path IS NOT NULL AS source_available, chunk_count, created_at
            """,
            (
                filename,
                content_type,
                processing_mode,
                source_type,
                json.dumps(source_metadata or {}, ensure_ascii=False),
                source_path,
                size_bytes,
                len(chunks),
                username,
            ),
        )
        document = cursor.fetchone()
        cursor.executemany(
            """
            INSERT INTO knowledge_chunks
                (document_id, chunk_index, page_number, content, display_content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document["id"],
                    index,
                    chunk["page_number"],
                    chunk["content"],
                    chunk.get("display_content") or chunk["content"],
                    Vector(vectors[index]),
                )
                for index, chunk in enumerate(chunks)
            ],
        )
    return document


def append_image(
    document_id: int | None,
    knowledge_name: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    corrected_text: str,
    ocr_lines: list[dict],
    content_hash: str,
    source_path: str,
    chunks: list[dict],
    vectors: list[list[float]],
    username: str,
) -> dict:
    """新建图片知识库或向现有图片知识库追加一张图片。"""
    ensure_schema()
    from pgvector import Vector

    with _connect() as connection, connection.cursor() as cursor:
        if document_id is None:
            # 知识库主体只保存汇总信息，原图明细统一写入 knowledge_images。
            cursor.execute(
                """
                INSERT INTO knowledge_documents
                    (original_name, content_type, processing_mode, source_type, source_metadata,
                     source_path, size_bytes, chunk_count, created_by)
                VALUES (%s, %s, 'OCR 图片', 'image', '{}'::jsonb, NULL, 0, 0, %s)
                RETURNING id, original_name, source_type, created_at
                """,
                (knowledge_name, content_type, username),
            )
            document = cursor.fetchone()
            document_id = document["id"]
            image_index = 1
            chunk_index = 0
        else:
            # 锁定主体行，保证两个追加请求不会生成重复的图片或文本块序号。
            cursor.execute(
                """
                SELECT id, original_name, source_type, created_at
                FROM knowledge_documents
                WHERE id = %s
                FOR UPDATE
                """,
                (document_id,),
            )
            document = cursor.fetchone()
            if document is None:
                raise AppException(404, "目标知识库不存在")
            if document["source_type"] not in {"image", "image_set"}:
                raise AppException(400, "当前仅支持向图片知识库追加图片")

            cursor.execute(
                """
                SELECT id
                FROM knowledge_images
                WHERE document_id = %s AND content_hash = %s
                """,
                (document_id, content_hash),
            )
            duplicate = cursor.fetchone()
            if duplicate:
                return {
                    **document,
                    "image_id": duplicate["id"],
                    "duplicate": True,
                    "added_chunk_count": 0,
                }

            cursor.execute(
                """
                SELECT COALESCE(MAX(image_index), 0) + 1 AS next_image_index
                FROM knowledge_images
                WHERE document_id = %s
                """,
                (document_id,),
            )
            image_index = cursor.fetchone()["next_image_index"]
            cursor.execute(
                """
                SELECT COALESCE(MAX(chunk_index), -1) + 1 AS next_chunk_index
                FROM knowledge_chunks
                WHERE document_id = %s
                """,
                (document_id,),
            )
            chunk_index = cursor.fetchone()["next_chunk_index"]

        cursor.execute(
            """
            INSERT INTO knowledge_images
                (document_id, image_index, original_name, content_type, source_path,
                 size_bytes, corrected_text, ocr_lines, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            RETURNING id
            """,
            (
                document_id,
                image_index,
                filename,
                content_type,
                source_path,
                size_bytes,
                corrected_text,
                json.dumps(ocr_lines, ensure_ascii=False),
                content_hash,
            ),
        )
        image_id = cursor.fetchone()["id"]
        cursor.executemany(
            """
            INSERT INTO knowledge_chunks
                (document_id, source_image_id, chunk_index, page_number, content, display_content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    image_id,
                    chunk_index + offset,
                    image_index,
                    chunk["content"],
                    chunk.get("display_content") or chunk["content"],
                    Vector(vectors[offset]),
                )
                for offset, chunk in enumerate(chunks)
            ],
        )
        cursor.execute(
            """
            UPDATE knowledge_documents
            SET size_bytes = size_bytes + %s,
                chunk_count = chunk_count + %s,
                source_type = CASE WHEN %s > 1 THEN 'image_set' ELSE 'image' END
            WHERE id = %s
            RETURNING id, original_name, processing_mode, source_type, size_bytes,
                      chunk_count, created_at
            """,
            (size_bytes, len(chunks), image_index, document_id),
        )
        result = cursor.fetchone()
    return {
        **result,
        "image_id": image_id,
        "image_index": image_index,
        "duplicate": False,
        "added_chunk_count": len(chunks),
    }


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
                       c.source_image_id,
                       c.chunk_index,
                       d.original_name,
                       d.source_type,
                       image.original_name AS source_image_name,
                       c.page_number,
                       c.content,
                       c.display_content,
                       1 - (c.embedding <=> %s) AS score,
                       ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s) AS result_rank
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.id = c.document_id
                LEFT JOIN knowledge_images image ON image.id = c.source_image_id
            )
            SELECT r.chunk_id,
                   r.document_id,
                   r.original_name,
                   r.source_type,
                   r.source_image_id,
                   r.source_image_name,
                   r.page_number,
                   COALESCE(r.display_content, r.content) AS content,
                   COALESCE(
                       (
                           SELECT STRING_AGG(
                               COALESCE(context_chunk.display_content, context_chunk.content),
                               E'\n\n' ORDER BY context_chunk.chunk_index
                           )
                           FROM knowledge_chunks context_chunk
                           WHERE context_chunk.document_id = r.document_id
                             AND (r.source_image_id IS NULL OR context_chunk.source_image_id = r.source_image_id)
                             AND context_chunk.chunk_index BETWEEN r.chunk_index - 1 AND r.chunk_index + 1
                       ),
                       COALESCE(r.display_content, r.content)
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
                   processing_mode, source_type,
                   (source_path IS NOT NULL OR EXISTS (
                       SELECT 1 FROM knowledge_images image WHERE image.document_id = knowledge_documents.id
                   )) AS source_available,
                   chunk_count, created_by, created_at
            FROM knowledge_documents
            ORDER BY id DESC
            """
        )
        return cursor.fetchall()


def get_document_chunks(document_id: int, page: int, page_size: int) -> dict | None:
    """分页返回指定知识文档及其文本块。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        # 先读取文档元数据，以便调用方区分空文档和不存在的文档。
        cursor.execute(
            """
            SELECT id, original_name, content_type, size_bytes,
                   processing_mode, source_type, source_metadata,
                   (source_path IS NOT NULL OR EXISTS (
                       SELECT 1 FROM knowledge_images image WHERE image.document_id = knowledge_documents.id
                   )) AS source_available,
                   chunk_count, created_by, created_at
            FROM knowledge_documents
            WHERE id = %s
            """,
            (document_id,),
        )
        document = cursor.fetchone()
        if document is None:
            return None

        cursor.execute(
            """
            SELECT chunk.id, chunk.chunk_index, chunk.page_number,
                   chunk.source_image_id, image.original_name AS source_image_name,
                   COALESCE(chunk.display_content, chunk.content) AS content
            FROM knowledge_chunks AS chunk
            LEFT JOIN knowledge_images AS image ON image.id = chunk.source_image_id
            WHERE chunk.document_id = %s
            ORDER BY chunk.chunk_index
            LIMIT %s OFFSET %s
            """,
            (document_id, page_size, (page - 1) * page_size),
        )
        return {
            "document": document,
            "items": cursor.fetchall(),
            "page": page,
            "page_size": page_size,
            "total": document["chunk_count"],
        }


def list_document_images(document_id: int) -> list[dict]:
    """返回知识库关联的全部图片，不暴露服务器本地路径。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, document_id, image_index, original_name, content_type,
                   size_bytes, created_at
            FROM knowledge_images
            WHERE document_id = %s
            ORDER BY image_index
            """,
            (document_id,),
        )
        return cursor.fetchall()


def get_source_image(document_id: int, image_id: int | None = None) -> dict | None:
    """返回指定图片来源；未指定时返回知识库中的第一张图片。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, original_name, content_type, source_path
            FROM knowledge_images
            WHERE document_id = %s AND (%s IS NULL OR id = %s)
            ORDER BY image_index
            LIMIT 1
            """,
            (document_id, image_id, image_id),
        )
        image = cursor.fetchone()
        if image is None and image_id is None:
            # 兼容尚未执行迁移的旧版单图片知识记录。
            cursor.execute(
                """
                SELECT NULL AS id,
                       COALESCE(NULLIF(source_metadata->>'original_filename', ''), original_name) AS original_name,
                       content_type,
                       source_path
                FROM knowledge_documents
                WHERE id = %s AND source_type IN ('image', 'image_set')
                """,
                (document_id,),
            )
            image = cursor.fetchone()
    if not image or not image.get("source_path"):
        return None

    candidate = Path(image["source_path"]).resolve()
    uploads_root = Path(settings.UPLOAD_FOLDER).resolve()
    if not candidate.is_relative_to(uploads_root) or not candidate.is_file():
        return None
    return {
        "path": str(candidate),
        "filename": image["original_name"],
        "content_type": image["content_type"],
    }


def delete_document(document_id: int) -> bool:
    """删除文档；外键级联同步删除全部向量块。"""
    ensure_schema()
    source_paths: list[str] = []
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT source_path FROM knowledge_documents WHERE id = %s", (document_id,))
        document = cursor.fetchone()
        if document and document.get("source_path"):
            source_paths.append(document["source_path"])
        cursor.execute("SELECT source_path FROM knowledge_images WHERE document_id = %s", (document_id,))
        source_paths.extend(row["source_path"] for row in cursor.fetchall())
        cursor.execute("DELETE FROM knowledge_documents WHERE id = %s", (document_id,))
        deleted = cursor.rowcount > 0

    uploads_root = Path(settings.UPLOAD_FOLDER).resolve()
    for source_path in set(source_paths if deleted else []):
        candidate = Path(source_path).resolve()
        # 只允许删除本系统上传目录中的原图，数据库路径不能越界影响其他文件。
        if candidate.is_relative_to(uploads_root):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("知识库已删除，但原图清理失败 path=%s error=%s", candidate, exc)
    return deleted
