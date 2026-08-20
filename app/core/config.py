import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    os.getenv(key, default)
    APP_NAME = os.getenv("APP_NAME", "Practice Server")
    属性名 环境变量名 默认值
    """
    APP_NAME = os.getenv("APP_NAME", "Practice Server")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
    DB_NAME = os.getenv("DB_NAME", "practice_db")

    #os.path.join()路径拼接函数
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    #UPLOAD_FOLDER 就是“文件最终保存到哪里
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    # PaddleOCR 模型保存在后端目录中，部署时可通过环境变量修改路径。
    OCR_MODEL_CACHE_DIR = os.getenv(
        "OCR_MODEL_CACHE_DIR",
        os.path.join(BASE_DIR, ".ocr_models"),
    )

    # 知识库使用独立 PostgreSQL，避免影响现有 MySQL 业务数据。
    VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "127.0.0.1")
    VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "5432"))
    VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME", "knowledge_db")
    VECTOR_DB_USER = os.getenv("VECTOR_DB_USER", "postgres")
    VECTOR_DB_PASSWORD = os.getenv("VECTOR_DB_PASSWORD", "1234")

    # Qwen3-Embedding-0.6B 默认输出 1024 维向量，字段维度必须与模型一致。
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    EMBEDDING_MODEL_SOURCE = os.getenv("EMBEDDING_MODEL_SOURCE", "modelscope")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    EMBEDDING_MODEL_CACHE_DIR = os.getenv(
        "EMBEDDING_MODEL_CACHE_DIR",
        os.path.join(BASE_DIR, ".embedding_models"),
    )

    # 文本块保留少量重叠，避免答案恰好落在两个块的边界上。
    KNOWLEDGE_CHUNK_SIZE = int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "600"))
    KNOWLEDGE_CHUNK_OVERLAP = int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "80"))

    # Ollama 负责根据向量检索结果生成最终答案。
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# 创建一个 Settings 实例，方便在项目其他地方导入使用
settings = Settings()
