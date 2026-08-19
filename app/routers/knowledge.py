from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import AppException
from app.core.responses import success_response
from app.dependencies.auth import current_user
from app.knowledge.service import delete_document, import_document, list_documents, search_knowledge

router = APIRouter()

# 第一阶段仅接收可稳定提取正文的 PDF 和 TXT，并限制内存上传大小。
ALLOWED_SUFFIXES = {".pdf", ".txt"}
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


class KnowledgeSearchRequest(BaseModel):
    """知识库检索参数。"""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/knowledge/upload")
async def upload_knowledge_document(file: UploadFile = File(...), user=Depends(current_user)):
    """上传文档并完成提取、切分、向量化和入库。"""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise AppException(400, "仅支持 PDF、TXT 文档")

    # 多读取一个字节，以便准确区分合法文件和超出限制的文件。
    content = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not content:
        raise AppException(400, "上传文档不能为空")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise AppException(413, "文档大小不能超过 20 MB")

    document = await run_in_threadpool(
        import_document,
        filename,
        file.content_type or "application/octet-stream",
        content,
        user["username"],
    )
    return success_response(data=document, message="知识文档导入成功", operator=user["username"])


@router.post("/knowledge/search")
async def search(payload: KnowledgeSearchRequest, user=Depends(current_user)):
    """向量检索相关文本块，并返回文档名称、页码和相似度。"""
    query = payload.query.strip()
    if not query:
        raise AppException(400, "检索内容不能为空")
    results = await run_in_threadpool(search_knowledge, query, payload.top_k)
    return success_response(data={"items": results}, operator=user["username"])


@router.get("/knowledge/documents")
async def get_documents(user=Depends(current_user)):
    """查询已导入的全部知识文档。"""
    documents = await run_in_threadpool(list_documents)
    return success_response(data={"items": documents}, operator=user["username"])


@router.delete("/knowledge/documents/{document_id}")
async def remove_document(document_id: int, user=Depends(current_user)):
    """删除文档并依靠外键级联删除所属向量。"""
    deleted = await run_in_threadpool(delete_document, document_id)
    if not deleted:
        raise AppException(404, "知识文档不存在")
    return success_response(message="知识文档删除成功", operator=user["username"])
