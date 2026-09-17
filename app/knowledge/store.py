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
                    markdown_content TEXT NOT NULL,
                    regions JSONB NOT NULL DEFAULT '[]'::jsonb,
                    content_hash VARCHAR(64),
                    page_number INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(document_id, image_index),
                    UNIQUE(document_id, content_hash)
                )
                """
            )
            # 为图片保存稳定的页码/顺序元数据，旧表通过 image_index 兼容补齐。
            cursor.execute(
                "ALTER TABLE knowledge_images ADD COLUMN IF NOT EXISTS page_number INTEGER"
            )
            cursor.execute(
                """
                UPDATE knowledge_images
                SET page_number = image_index
                WHERE page_number IS NULL
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
            # 旧字段直接改名，旧后端将无法再读取图片知识内容。
            cursor.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'knowledge_images' AND column_name = 'corrected_text'
                    ) THEN
                        ALTER TABLE knowledge_images RENAME COLUMN corrected_text TO markdown_content;
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'knowledge_images' AND column_name = 'ocr_lines'
                    ) THEN
                        ALTER TABLE knowledge_images RENAME COLUMN ocr_lines TO regions;
                    END IF;
                END $$;
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
                     size_bytes, markdown_content, regions, page_number)
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
                       COALESCE(document.source_metadata->'ocr_lines', '[]'::jsonb),
                       1
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


def append_document(
    document_id: int,
    filename: str,
    size_bytes: int,
    chunks: list[dict],
    vectors: list[list[float]],
) -> dict:
    """把 PDF/TXT 文本块追加到已有知识库，并在同一事务内更新统计信息。"""
    ensure_schema()
    from pgvector import Vector

    with _connect() as connection, connection.cursor() as cursor:
        # 锁定知识库主体，保证并发导入时文本块序号不会重复。
        cursor.execute(
            """
            SELECT id, original_name, content_type, processing_mode, source_type,
                   source_metadata, source_path, size_bytes, chunk_count,
                   created_by, created_at
            FROM knowledge_documents
            WHERE id = %s
            FOR UPDATE
            """,
            (document_id,),
        )
        document = cursor.fetchone()
        if document is None:
            raise AppException(404, "目标知识库不存在")

        cursor.execute(
            """
            SELECT COALESCE(MAX(chunk_index), -1) + 1 AS next_chunk_index
            FROM knowledge_chunks
            WHERE document_id = %s
            """,
            (document_id,),
        )
        next_chunk_index = cursor.fetchone()["next_chunk_index"]
        cursor.executemany(
            """
            INSERT INTO knowledge_chunks
                (document_id, chunk_index, page_number, content, display_content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    next_chunk_index + index,
                    chunk["page_number"],
                    chunk["content"],
                    chunk.get("display_content") or chunk["content"],
                    Vector(vectors[index]),
                )
                for index, chunk in enumerate(chunks)
            ],
        )
        cursor.execute(
            """
            UPDATE knowledge_documents
            SET size_bytes = size_bytes + %s,
                chunk_count = chunk_count + %s
            WHERE id = %s
            RETURNING id, original_name, processing_mode, source_type,
                      size_bytes, chunk_count, created_at
            """,
            (size_bytes, len(chunks), document_id),
        )
        result = cursor.fetchone()
    return {
        **result,
        "added_chunk_count": len(chunks),
        "appended_filename": filename,
    }


def append_image(
    document_id: int | None,
    knowledge_name: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    markdown_content: str,
    regions: list[dict],
    content_hash: str,
    source_path: str,
    chunks: list[dict],
    vectors: list[list[float]],
    username: str,
    page_number: int | None = None,
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

        # 未传真实页码时沿用保存顺序，确保旧调用方仍然可以正常追加图片。
        stored_page_number = max(1, int(page_number or image_index))
        cursor.execute(
            """
            INSERT INTO knowledge_images
                (document_id, image_index, original_name, content_type, source_path,
                 size_bytes, markdown_content, regions, content_hash, page_number)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (
                document_id,
                image_index,
                filename,
                content_type,
                source_path,
                size_bytes,
                markdown_content,
                json.dumps(regions, ensure_ascii=False),
                content_hash,
                stored_page_number,
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
                    stored_page_number,
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
        "page_number": stored_page_number,
        "duplicate": False,
        "added_chunk_count": len(chunks),
    }


def update_image_text(document_id: int, image_id: int, content_hash: str,
                      markdown: str, chunks: list[dict], vectors: list[list[float]]) -> dict:
    """原子更新 OCR 校正文稿和索引；失败时由事务保留原数据。"""
    ensure_schema()
    from pgvector import Vector

    with _connect() as connection, connection.cursor() as cursor:
        # 与续表合并使用相同的文档锁，防止并发更新破坏合并元数据。
        cursor.execute(
            "SELECT source_metadata FROM knowledge_documents WHERE id = %s FOR UPDATE",
            (document_id,),
        )
        document = cursor.fetchone()
        if document is None:
            raise AppException(404, "知识库不存在")
        groups = (document["source_metadata"] or {}).get("table_merges", [])
        if any(s["image_id"] == image_id for g in groups for s in g["selections"]):
            raise AppException(409, "该图片已参与续表合并，请先撤销对应合并，再修改并重新合并")
        cursor.execute(
            """SELECT content_hash, page_number, image_index FROM knowledge_images
               WHERE id = %s AND document_id = %s FOR UPDATE""",
            (image_id, document_id),
        )
        image = cursor.fetchone()
        if image is None:
            raise AppException(404, "原图片不存在")
        if image["content_hash"] != content_hash:
            raise AppException(409, "本地图片已变化，请重新导入，不能覆盖原图内容")
        cursor.execute(
            "DELETE FROM knowledge_chunks WHERE document_id = %s AND source_image_id = %s",
            (document_id, image_id),
        )
        cursor.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) + 1 AS start FROM knowledge_chunks WHERE document_id = %s",
            (document_id,),
        )
        start = cursor.fetchone()["start"]
        cursor.executemany(
            """INSERT INTO knowledge_chunks
               (document_id, source_image_id, chunk_index, page_number, content, display_content, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [(document_id, image_id, start + n, image["page_number"] or image["image_index"],
              chunk["content"], chunk.get("display_content") or chunk["content"], Vector(vectors[n]))
             for n, chunk in enumerate(chunks)],
        )
        cursor.execute(
            "UPDATE knowledge_images SET markdown_content = %s WHERE id = %s",
            (markdown, image_id),
        )
        cursor.execute(
            """UPDATE knowledge_documents SET chunk_count =
               (SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = %s)
               WHERE id = %s RETURNING id, chunk_count""",
            (document_id, document_id),
        )
        return {**cursor.fetchone(), "image_id": image_id, "updated": True, "duplicate": False}


def search_chunks(query_vector: list[float], top_k: int) -> list[dict]:
    """使用余弦距离查询相关文本块及其相邻上下文。"""
    ensure_schema()
    from pgvector import Vector

    database_started_at = time.perf_counter()
    with _connect() as connection, connection.cursor() as cursor:
        # 上下文按同一知识库主体的 chunk_index 拼接，允许相邻图片之间共享检索上下文。
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
                   r.content AS content,
                   COALESCE(
                       (
                           SELECT STRING_AGG(
                               context_chunk.content,
                               E'\n\n' ORDER BY context_chunk.chunk_index
                           )
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
                   size_bytes, page_number, created_at
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


def get_document_name(document_id: int) -> str | None:
    """返回图片知识库主体名称，用于追加图片时补全检索元数据。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT original_name FROM knowledge_documents WHERE id = %s",
            (document_id,),
        )
        document = cursor.fetchone()
    return str(document["original_name"]) if document else None


def delete_image(document_id: int, image_id: int) -> dict | None:
    """删除知识库中的单张图片，并同步删除其文本块、向量和物理文件。"""
    ensure_schema()
    source_path: str | None = None
    with _connect() as connection, connection.cursor() as cursor:
        # 锁定主体，避免删除和追加同时修改汇总统计。
        cursor.execute(
            "SELECT id, source_metadata FROM knowledge_documents WHERE id = %s FOR UPDATE",
            (document_id,),
        )
        parent = cursor.fetchone()
        if parent is None:
            return None
        # 合并表的表头可能来自另一张图，删除前要求撤销以避免失去来源依据。
        if any(s["image_id"] == image_id
               for g in (parent.get("source_metadata") or {}).get("table_merges", [])
               for s in g["selections"]):
            raise AppException(409, "该图片参与了续表合并，请先撤销合并再删除")

        cursor.execute(
            """
            SELECT id, original_name, source_path, size_bytes
            FROM knowledge_images
            WHERE document_id = %s AND id = %s
            FOR UPDATE
            """,
            (document_id, image_id),
        )
        image = cursor.fetchone()
        if image is None:
            return None
        source_path = image["source_path"]

        cursor.execute(
            "SELECT COUNT(*) AS chunk_count FROM knowledge_chunks WHERE source_image_id = %s",
            (image_id,),
        )
        deleted_chunk_count = int(cursor.fetchone()["chunk_count"])
        cursor.execute("DELETE FROM knowledge_images WHERE id = %s", (image_id,))
        cursor.execute(
            """
            UPDATE knowledge_documents AS document
            SET size_bytes = GREATEST(document.size_bytes - %s, 0),
                chunk_count = GREATEST(document.chunk_count - %s, 0),
                source_type = CASE
                    WHEN (SELECT COUNT(*) FROM knowledge_images WHERE document_id = document.id) > 1
                    THEN 'image_set'
                    ELSE 'image'
                END
            WHERE document.id = %s
            RETURNING id, original_name, source_type, size_bytes, chunk_count
            """,
            (image["size_bytes"], deleted_chunk_count, document_id),
        )
        document = cursor.fetchone()

    if source_path:
        uploads_root = Path(settings.UPLOAD_FOLDER).resolve()
        candidate = Path(source_path).resolve()
        # 只清理上传目录内的文件，避免数据库异常路径影响其他文件。
        if candidate.is_relative_to(uploads_root):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("图片记录已删除，但原图清理失败 path=%s error=%s", candidate, exc)
    return {
        "document": document,
        "image_id": image_id,
        "deleted_filename": image["original_name"],
        "deleted_chunk_count": deleted_chunk_count,
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


def table_merge_snapshot(document_id: int) -> dict:
    """读取同一知识库的图片归档及合并元数据，不触碰原图文件。"""
    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT original_name, source_metadata FROM knowledge_documents WHERE id = %s", (document_id,))
        document = cursor.fetchone()
        if document is None:
            raise AppException(404, "知识库不存在")
        cursor.execute(
            """SELECT id, original_name, markdown_content, COALESCE(page_number, image_index) AS page_number
               FROM knowledge_images WHERE document_id = %s ORDER BY image_index""",
            (document_id,),
        )
        return {"id": document_id, "name": document["original_name"],
                "metadata": document["source_metadata"] or {}, "images": cursor.fetchall()}


def search_table_records(query: str, top_k: int) -> list[dict]:
    """从已确认的逻辑表独立召回，历史合并表无需重新生成向量即可使用。"""
    from app.knowledge.table_records import matching_records

    ensure_schema()
    matches = []
    # ponytail: 首期流式扫描合并元数据；大规模知识库再建立字段级检索索引。
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT id, original_name, source_type, source_metadata
               FROM knowledge_documents
               WHERE source_metadata ? 'table_merges' ORDER BY id"""
        )
        for document in cursor:
            for group in document["source_metadata"]["table_merges"]:
                found = matching_records(query, group)
                if not found:
                    continue
                references = list({
                    (item["source"].get("image_id"), item["source"].get("page_number", 1))
                    for item in found
                })
                references.sort(key=lambda ref: (ref[1], ref[0] or 0))
                # 每条证据都附真实页码，不把跨页内容错误归给第一张图片。
                context = f"表名：{group['title']}\n" + "\n\n".join(
                    f"[图片ID：{item['source'].get('image_id')}，第 {item['source'].get('page_number', 1)} 页]\n"
                    + item["text"] for item in found
                )
                matches.append({
                    "document_id": document["id"], "original_name": document["original_name"],
                    "source_type": document["source_type"], "source_image_id": references[0][0],
                    "page_number": references[0][1], "source_image_name": None,
                    "content": context, "context": context, "score": 0.0,
                    "match_type": "table_fields", "table_id": group["id"],
                    "references": [{"image_id": image, "page_number": page} for image, page in references],
                })
    # 关键词匹配不伪造余弦分数；同等匹配维持数据库中的稳定顺序。
    return matches[:top_k]


def commit_table_merge(snapshot: dict, image_ids: list[int], chunks: list[dict],
                       vectors: list[list[float]], metadata: dict) -> dict:
    """锁定文档并校验版本，原子替换选中图片的索引与逻辑表。"""
    from pgvector import Vector
    from app.knowledge.table_merge import revision

    ensure_schema()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT source_metadata FROM knowledge_documents WHERE id = %s FOR UPDATE", (snapshot["id"],))
        document = cursor.fetchone()
        if document is None:
            raise AppException(404, "知识库不存在")
        cursor.execute(
            """SELECT id, markdown_content FROM knowledge_images
               WHERE document_id = %s ORDER BY image_index FOR UPDATE""", (snapshot["id"],),
        )
        if revision(cursor.fetchall(), document["source_metadata"] or {}) != revision(snapshot["images"], snapshot["metadata"]):
            raise AppException(409, "图片已变化，请重新预览")
        cursor.execute("DELETE FROM knowledge_chunks WHERE document_id = %s AND source_image_id = ANY(%s)",
                       (snapshot["id"], image_ids))
        cursor.execute("SELECT COALESCE(MAX(chunk_index), -1) + 1 AS start FROM knowledge_chunks WHERE document_id = %s",
                       (snapshot["id"],))
        start = cursor.fetchone()["start"]
        cursor.executemany(
            """INSERT INTO knowledge_chunks
               (document_id, source_image_id, chunk_index, page_number, content, display_content, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [(snapshot["id"], c["source_image_id"], start + n, c["page_number"],
              c["content"], c.get("display_content", c["content"]), Vector(vectors[n])) for n, c in enumerate(chunks)],
        )
        cursor.execute(
            """UPDATE knowledge_documents SET source_metadata = %s::jsonb,
               chunk_count = (SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = %s)
               WHERE id = %s RETURNING id, chunk_count""",
            (json.dumps(metadata, ensure_ascii=False), snapshot["id"], snapshot["id"]),
        )
        return cursor.fetchone()
