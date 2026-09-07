import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from app.core.exceptions import AppException
from app.core.responses import success_response
from app.dependencies.auth import current_user
from app import knowledge as knowledge_workflow

router = APIRouter()

# 第一阶段仅接收可稳定提取正文的 PDF 和 TXT，并限制内存上传大小。
ALLOWED_SUFFIXES = {".pdf", ".txt"}
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


class KnowledgeSearchRequest(BaseModel):
    """知识库检索参数。"""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


async def _read_document(file: UploadFile) -> tuple[str, bytes]:
    """统一校验知识文档名称、类型、大小和空内容。"""
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
    return filename, content


@router.post("/knowledge/preview")
async def preview_knowledge_document(file: UploadFile = File(...), user=Depends(current_user)):
    """提取并预览文档文本块，不执行向量化和入库。"""
    filename, content = await _read_document(file)
    preview = await run_in_threadpool(knowledge_workflow.preview_document, filename, content)
    return success_response(data=preview, operator=user["username"])


@router.post("/knowledge/upload")
async def upload_knowledge_document(
    file: UploadFile = File(...),
    qa_split: bool = Form(False),
    user=Depends(current_user),
):
    """上传文档并完成提取、切分、向量化和入库。"""
    filename, content = await _read_document(file)

    document = await run_in_threadpool(
        knowledge_workflow.import_document,
        filename,
        file.content_type or "application/octet-stream",
        content,
        user["username"],
        qa_split,
    )
    return success_response(data=document, message="知识文档导入成功", operator=user["username"])


@router.post("/knowledge/search")
async def search(payload: KnowledgeSearchRequest, user=Depends(current_user)):
    """向量检索相关文本块，并返回文档名称、页码和相似度。"""
    query = payload.query.strip()
    if not query:
        raise AppException(400, "检索内容不能为空")
    results = await run_in_threadpool(knowledge_workflow.search, query, payload.top_k)
    return success_response(data={"items": results}, operator=user["username"])


@router.post("/knowledge/ask")
async def ask(payload: KnowledgeSearchRequest, user=Depends(current_user)):
    """根据检索到的知识上下文调用 Ollama 生成答案。"""
    query = payload.query.strip()
    if not query:
        raise AppException(400, "问题不能为空")
    result = await run_in_threadpool(knowledge_workflow.ask, query, payload.top_k)
    return success_response(data=result, operator=user["username"])


@router.post("/knowledge/ask/stream")
async def ask_stream(payload: KnowledgeSearchRequest, user=Depends(current_user)):
    """以 NDJSON 逐段返回检索元数据和 Ollama 答案。"""
    query = payload.query.strip()
    if not query:
        raise AppException(400, "问题不能为空")
    # 检索和 Embedding 在工作线程完成，避免阻塞 FastAPI 事件循环。
    events = await run_in_threadpool(knowledge_workflow.stream_answer, query, payload.top_k)
    body = (
        (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        for event in events
    )
    return StreamingResponse(body, media_type="application/x-ndjson")


@router.get("/knowledge/documents")
async def get_documents(user=Depends(current_user)):
    """查询已导入的全部知识文档。"""
    documents = await run_in_threadpool(knowledge_workflow.list_documents)
    return success_response(data={"items": documents}, operator=user["username"])


@router.get("/knowledge/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    user=Depends(current_user),
):
    """分页查询指定知识文档的元数据和文本块。"""
    result = await run_in_threadpool(knowledge_workflow.get_document_chunks, document_id, page, page_size)
    if result is None:
        raise AppException(404, "知识文档不存在")
    return success_response(data=result, operator=user["username"])


@router.delete("/knowledge/documents/{document_id}")
async def remove_document(document_id: int, user=Depends(current_user)):
    """删除文档并依靠外键级联删除所属向量。"""
    deleted = await run_in_threadpool(knowledge_workflow.delete_document, document_id)
    if not deleted:
        raise AppException(404, "知识文档不存在")
    return success_response(message="知识文档删除成功", operator=user["username"])
