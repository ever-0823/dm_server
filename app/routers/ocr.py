import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import AppException
from app.core.responses import success_response
from app.dependencies.auth import current_user
from app import knowledge as knowledge_workflow
from app.ocr.ppocrv6 import OcrUnavailable, recognize_text

router = APIRouter()

# 在上传入口校验文件，避免无效内容进入 PaddleOCR。
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def _read_image(file: UploadFile) -> tuple[str, str, bytes]:
    """统一校验并读取 OCR 图片。"""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if file.content_type not in ALLOWED_IMAGE_TYPES or suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise AppException(400, "仅支持 JPG、PNG、BMP、WEBP 图片")

    content = await file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise AppException(400, "上传图片不能为空")
    if len(content) > MAX_IMAGE_BYTES:
        raise AppException(413, "图片大小不能超过 10 MB")
    return filename, suffix, content


@router.post("/ocr/ppocrv6")
async def ppocrv6(file: UploadFile = File(...), user=Depends(current_user)):
    """使用本地 PP-OCRv6 模型识别单张图片中的文字。"""
    _filename, suffix, content = await _read_image(file)

    try:
        lines = await run_in_threadpool(recognize_text, content, suffix)
    except OcrUnavailable as exc:
        raise AppException(503, str(exc)) from exc

    return success_response(
        message="识别完成",
        data={"text": "\n".join(line["text"] for line in lines), "lines": lines},
        operator=user["username"],
    )


@router.post("/knowledge/upload-image")
async def upload_ocr_image_to_knowledge(
    file: UploadFile = File(...),
    knowledge_name: str = Form("", max_length=255),
    document_id: int | None = Form(None),
    corrected_text: str = Form(..., min_length=1, max_length=200000),
    lines_json: str = Form("[]", max_length=1000000),
    user=Depends(current_user),
):
    """保存原图、用户校正文本和 OCR 坐标，并生成知识向量。"""
    filename, _suffix, content = await _read_image(file)
    try:
        lines = json.loads(lines_json)
    except json.JSONDecodeError as exc:
        raise AppException(400, "OCR 坐标数据格式错误") from exc
    if not isinstance(lines, list):
        raise AppException(400, "OCR 坐标数据必须是列表")
    dataset_name = knowledge_name.strip()
    if document_id is None and not dataset_name:
        raise AppException(400, "知识库名称不能为空")

    document = await run_in_threadpool(
        knowledge_workflow.import_image,
        filename,
        dataset_name,
        file.content_type or "application/octet-stream",
        content,
        corrected_text,
        lines,
        user["username"],
        document_id,
    )
    message = "图片已存在，无需重复添加" if document.get("duplicate") else "图片文字已保存到知识库"
    return success_response(data=document, message=message, operator=user["username"])
