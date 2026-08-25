"""Qwen3 Embedding 模型加载与向量生成实现。"""

import logging
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock

from app.core.config import settings
from app.core.exceptions import AppException


logger = logging.getLogger(__name__)

# 本地模型实例可被多个请求复用，推理锁用于避免 CPU 并发时内存瞬间放大。
_inference_lock = Lock()


@lru_cache(maxsize=1)
def embedding_model():
    """延迟加载并缓存 Qwen3 Embedding，首次调用时才下载模型。"""
    started_at = time.perf_counter()
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
        # 设备配置支持 CPU 和 CUDA，部署时无需改动调用代码。
        model = SentenceTransformer(model_path, device=settings.EMBEDDING_DEVICE)
        logger.info(
            "知识库模型加载完成 model=%s device=%s elapsed_ms=%.1f",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE,
            (time.perf_counter() - started_at) * 1000,
        )
        return model
    except Exception as exc:
        raise AppException(503, f"Qwen3 Embedding 模型加载失败：{exc}") from exc


def encode_documents(texts: list[str]) -> list[list[float]]:
    """生成归一化文档向量，便于使用余弦距离检索。"""
    started_at = time.perf_counter()
    with _inference_lock:
        vectors = embedding_model().encode(texts, normalize_embeddings=True)
    logger.info(
        "知识库文档向量生成 count=%d elapsed_ms=%.1f",
        len(texts),
        (time.perf_counter() - started_at) * 1000,
    )
    return [vector.tolist() for vector in vectors]


def encode_query(query: str) -> list[float]:
    """使用 Qwen3 的 query 提示生成查询向量。"""
    started_at = time.perf_counter()
    with _inference_lock:
        vector = embedding_model().encode([query], prompt_name="query", normalize_embeddings=True)[0]
    logger.info(
        "知识库查询向量生成 query_length=%d elapsed_ms=%.1f",
        len(query),
        (time.perf_counter() - started_at) * 1000,
    )
    return vector.tolist()
