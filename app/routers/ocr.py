from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import AppException
from app.core.responses import success_response
from app.dependencies.auth import current_user
from app.ocr.ppocrv6 import OcrUnavailable, recognize_text

router = APIRouter()

# 在上传入口校验文件，避免无效内容进入 PaddleOCR。
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/ocr/ppocrv6")
async def ppocrv6(file: UploadFile = File(...), user=Depends(current_user)):
    """使用本地 PP-OCRv6 模型识别单张图片中的文字。"""
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_IMAGE_TYPES or suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise AppException(400, "仅支持 JPG、PNG、BMP、WEBP 图片")

    # 多读取一个字节，用于准确判断文件是否超过大小限制。
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise AppException(400, "上传图片不能为空")
    if len(content) > MAX_IMAGE_BYTES:
        raise AppException(413, "图片大小不能超过 10 MB")

    try:
        lines = await run_in_threadpool(recognize_text, content, suffix)
    except OcrUnavailable as exc:
        raise AppException(503, str(exc)) from exc

    return success_response(
        message="识别完成",
        data={"text": "\n".join(line["text"] for line in lines), "lines": lines},
        operator=user["username"],
    )
