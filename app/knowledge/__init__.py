"""知识库 module 的公共工作流 interface。"""

# 路由和调用者只从此处访问知识库，内部实现可独立演进。
from app.knowledge.workflow import (
    ask,
    delete_document,
    get_document_chunks,
    get_source_image,
    import_document,
    import_image,
    list_documents,
    preview_document,
    search,
    stream_answer,
)

__all__ = [
    "ask",
    "delete_document",
    "get_document_chunks",
    "get_source_image",
    "import_document",
    "import_image",
    "list_documents",
    "preview_document",
    "search",
    "stream_answer",
]
