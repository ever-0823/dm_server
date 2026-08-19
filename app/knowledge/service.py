import io
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock

from app.core.config import settings
from app.core.exceptions import AppException


# 本地模型实例可被多个请求复用，推理锁用于避免 CPU 并发时内存瞬间放大。
_inference_lock = Lock()
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


@lru_cache(maxsize=1)
def _embedding_model():
    """延迟加载并缓存 Qwen3 Embedding，首次调用时才下载模型。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise AppException(503, "Qwen3 Embedding 依赖未安装") from exc

    try:
        Path(settings.EMBEDDING_MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        model_path = settings.EMBEDDING_MODEL
        if settings.EMBEDDING_MODEL_SOURCE == "modelscope":
            # 国内网络优先使用 ModelScope 下载，推理仍由 SentenceTransformer 完成。
            from modelscope import snapshot_download

            model_path = snapshot_download(
                settings.EMBEDDING_MODEL,
                cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
            )
        return SentenceTransformer(
            model_path,
            device=settings.EMBEDDING_DEVICE,
        )
    except Exception as exc:
        raise AppException(503, f"Qwen3 Embedding 模型加载失败：{exc}") from exc


def _encode_documents(texts: list[str]) -> list[list[float]]:
    """生成归一化文档向量，便于使用余弦距离检索。"""
    with _inference_lock:
        vectors = _embedding_model().encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def _encode_query(query: str) -> list[float]:
    """使用 Qwen3 的 query 提示生成查询向量。"""
    with _inference_lock:
        vector = _embedding_model().encode([query], prompt_name="query", normalize_embeddings=True)[0]
    return vector.tolist()


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
    """按字符切分正文，并优先在段落或中文句号处结束文本块。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("文本块大小必须大于重叠长度")

    # 只压缩多余空白，保留段落换行作为自然切分边界。
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            # 后半段寻找自然边界，避免为了短句产生过小文本块。
            minimum_end = start + chunk_size // 2
            paragraph_end = normalized.rfind("\n", minimum_end, end)
            sentence_end = normalized.rfind("。", minimum_end, end)
            boundary = max(paragraph_end, sentence_end)
            if boundary >= minimum_end:
                end = boundary + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def import_document(filename: str, content_type: str, content: bytes, username: str) -> dict:
    """提取、切分并向量化文档，最后在一个事务中写入 pgvector。"""
    suffix = Path(filename).suffix.lower()
    pages = extract_pages(content, suffix)
    chunks: list[dict] = []
    for page_number, page_text in pages:
        for chunk_text in split_text(
            page_text,
            settings.KNOWLEDGE_CHUNK_SIZE,
            settings.KNOWLEDGE_CHUNK_OVERLAP,
        ):
            chunks.append({"page_number": page_number, "content": chunk_text})

    if not chunks:
        raise AppException(400, "文档未提取到文字，扫描版 PDF 请先进行 OCR")

    vectors = _encode_documents([chunk["content"] for chunk in chunks])
    if any(len(vector) != settings.EMBEDDING_DIMENSIONS for vector in vectors):
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

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
            (filename, content_type, len(content), len(chunks), username),
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


def search_knowledge(query: str, top_k: int) -> list[dict]:
    """使用 pgvector 余弦距离返回最相关的文本块和来源文档。"""
    ensure_schema()
    query_vector = _encode_query(query)
    if len(query_vector) != settings.EMBEDDING_DIMENSIONS:
        raise AppException(500, "模型输出维度与 EMBEDDING_DIMENSIONS 配置不一致")

    from pgvector import Vector

    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.id AS chunk_id,
                   c.document_id,
                   d.original_name,
                   c.page_number,
                   c.content,
                   1 - (c.embedding <=> %s) AS score
            FROM knowledge_chunks c
            JOIN knowledge_documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (Vector(query_vector), Vector(query_vector), top_k),
        )
        rows = cursor.fetchall()

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
