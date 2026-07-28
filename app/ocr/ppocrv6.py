import os
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings


class OcrUnavailable(Exception):
    """本地 OCR 引擎无法处理请求时抛出的异常。"""


# PaddleOCR 复用同一个模型实例，通过锁避免多个请求同时执行 predict。
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _ocr():
    """只加载一次 PP-OCRv6，避免每次请求都重复初始化模型。"""
    Path(settings.OCR_MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", settings.OCR_MODEL_CACHE_DIR)

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OcrUnavailable("OCR 依赖未安装，请安装 paddleocr 和 paddlepaddle") from exc

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # 当前 Windows CPU 环境的 oneDNN 不支持部分 PP-OCRv6 属性，关闭后使用普通 CPU 推理。
        enable_mkldnn=False,
    )


def recognize_text(content: bytes, suffix: str) -> list[dict]:
    """识别上传的图片，并返回统一格式的文本行。"""
    temp_path: Path | None = None
    try:
        # PaddleOCR 需要文件路径，因此先把上传内容写入临时文件。
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        with _inference_lock:
            results = _ocr().predict(str(temp_path))
        return _to_lines(results)
    except OcrUnavailable:
        raise
    except Exception as exc:
        raise OcrUnavailable(f"OCR 识别失败：{exc}") from exc
    finally:
        # Windows 无法删除正在使用的文件，因此在识别结束后统一清理。
        if temp_path:
            temp_path.unlink(missing_ok=True)


def _to_lines(results: list[Any]) -> list[dict]:
    """把 PaddleOCR 结果转换为文本和置信度字段。"""
    lines: list[dict] = []
    for item in results:
        data = getattr(item, "json", item)
        result = data.get("res", data) if isinstance(data, dict) else {}
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []

        for index, text in enumerate(texts):
            line = {"text": str(text), "score": None}
            if index < len(scores):
                line["score"] = float(scores[index])
            lines.append(line)
    return lines
