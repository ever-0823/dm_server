import json

from fastapi.testclient import TestClient

from app.app_factory import create_app
from app.dependencies.auth import current_user


def test_knowledge_preview_api_does_not_store_document(monkeypatch) -> None:
    """预览接口应返回文本块信息，并且不调用正式导入流程。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {
        "id": 1,
        "username": "test_user",
        "role": "admin",
    }
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.preview_document",
        lambda filename, _content: {
            "filename": filename,
            "page_count": 1,
            "chunk_count": 1,
            "items": [{"page_number": 1, "content": "维护前切断电源。"}],
            "truncated": False,
        },
    )
    try:
        response = TestClient(app).post(
            "/api/knowledge/preview",
            files={"file": ("设备规范.txt", "维护前切断电源。".encode("utf-8"), "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["chunk_count"] == 1


def test_knowledge_preview_rejects_removed_office_formats() -> None:
    """移除 Docling 后，Office 文档必须在上传边界直接拒绝。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "test_user"}
    try:
        response = TestClient(app).post(
            "/api/knowledge/preview",
            files={"file": ("设备规范.docx", b"office-content", "application/octet-stream")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400
    assert "仅支持 PDF、TXT" in response.json()["message"]


def test_knowledge_upload_forwards_qa_split(monkeypatch) -> None:
    """上传接口应把 QA 拆分开关传给知识库工作流。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {
        "id": 1,
        "username": "test_user",
        "role": "admin",
    }
    received: dict = {}

    # 使用内存替身检查参数传递，避免测试加载模型或写入 pgvector。
    def fake_import(filename, content_type, content, username, qa_split=False, document_id=None):
        received.update(
            filename=filename,
            content=content,
            username=username,
            qa_split=qa_split,
            document_id=document_id,
        )
        return {"id": 1, "chunk_count": 1}

    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.import_document",
        fake_import,
    )
    try:
        response = TestClient(app).post(
            "/api/knowledge/upload",
            data={"qa_split": "true", "document_id": "7"},
            files={"file": ("设备规范.txt", "维护前切断电源。".encode("utf-8"), "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert received["filename"] == "设备规范.txt"
    assert received["qa_split"] is True
    assert received["document_id"] == 7


def test_knowledge_search_api(monkeypatch) -> None:
    """验证知识库检索接口能够返回标准结果。"""
    app = create_app()

    # 用测试用户替代真实登录流程，让测试只关注知识库接口本身。
    app.dependency_overrides[current_user] = lambda: {
        "id": 1,
        "username": "test_user",
        "role": "admin",
    }

    # 模拟向量检索结果，避免单元测试加载 1 GB 级 Embedding 模型。
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.search",
        lambda query, top_k: [
            {
                "chunk_id": 1,
                "document_id": 7,
                "original_name": "设备维护规范.txt",
                "page_number": 1,
                "content": "维护设备前必须切断电源。",
                "score": 0.91,
            }
        ][:top_k],
    )

    try:
        response = TestClient(app).post(
            "/api/knowledge/search",
            json={"query": "维护设备前需要做什么？", "top_k": 3},
        )
    finally:
        # 清理依赖覆盖，避免影响同一进程中的其他测试。
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["items"][0]["original_name"] == "设备维护规范.txt"
    assert body["data"]["items"][0]["score"] == 0.91


def test_knowledge_stream_api_returns_ndjson_events(monkeypatch) -> None:
    """流式接口应保持事件顺序，并使用逐行 JSON 响应。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {
        "id": 1,
        "username": "test_user",
        "role": "admin",
    }
    # 路由测试只检查传输协议，真实 Ollama 流由服务层测试覆盖。
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.stream_answer",
        lambda _query, _top_k: iter(
            (
                {"type": "metadata", "sources": [], "items": []},
                {"type": "delta", "content": "切断电源。"},
                {"type": "done"},
            )
        ),
    )

    try:
        response = TestClient(app).post(
            "/api/knowledge/ask/stream",
            json={"query": "维护前做什么？", "top_k": 3},
        )
    finally:
        # 每个测试结束后清理登录替身，避免污染其他接口测试。
        app.dependency_overrides.clear()

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [event["type"] for event in events] == ["metadata", "delta", "done"]


def test_knowledge_document_chunks_api(monkeypatch) -> None:
    """数据集详情接口应返回文档信息和按顺序排列的文本块。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {
        "id": 1,
        "username": "test_user",
        "role": "admin",
    }
    # 使用内存结果验证接口结构，避免测试连接真实 pgvector。
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.get_document_chunks",
        lambda document_id, page, page_size: {
            "document": {"id": document_id, "original_name": "设备规范.txt", "chunk_count": 1},
            "items": [{"id": 9, "chunk_index": 0, "page_number": 1, "content": "维护前切断电源。"}],
            "page": page,
            "page_size": page_size,
            "total": 1,
        },
    )
    try:
        response = TestClient(app).get("/api/knowledge/documents/7/chunks")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["document"]["id"] == 7
    assert response.json()["data"]["items"][0]["content"] == "维护前切断电源。"
    assert response.json()["data"]["page_size"] == 50


def test_knowledge_source_image_api(monkeypatch, tmp_path) -> None:
    """图片来源接口应返回原图文件，而不是暴露服务器路径。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "test_user"}
    image_path = tmp_path / "device.png"
    image_path.write_bytes(b"image-content")
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.get_source_image",
        lambda _document_id: {
            "path": str(image_path),
            "filename": "device.png",
            "content_type": "image/png",
        },
    )
    try:
        response = TestClient(app).get("/api/knowledge/documents/7/source-image")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.content == b"image-content"
    assert response.headers["content-type"] == "image/png"


def test_knowledge_document_images_api(monkeypatch) -> None:
    """图片列表路由应通过知识库公共入口返回图片摘要。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "test_user"}
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.list_document_images",
        lambda _document_id: [
            {
                "id": 11,
                "document_id": 7,
                "image_index": 1,
                "original_name": "device.png",
                "content_type": "image/png",
                "size_bytes": 12,
            }
        ],
    )
    try:
        response = TestClient(app).get("/api/knowledge/documents/7/images")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["items"][0]["original_name"] == "device.png"


def test_knowledge_document_image_delete_api(monkeypatch) -> None:
    """单图删除路由应返回后端更新后的分组统计。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "test_user"}
    monkeypatch.setattr(
        "app.routers.knowledge.knowledge_workflow.delete_image",
        lambda document_id, image_id: {
            "document": {
                "id": document_id,
                "source_type": "image",
                "chunk_count": 0,
            },
            "image_id": image_id,
            "deleted_filename": "device.png",
            "deleted_chunk_count": 2,
        },
    )
    try:
        response = TestClient(app).delete("/api/knowledge/documents/7/images/11")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["image_id"] == 11
    assert response.json()["data"]["deleted_chunk_count"] == 2
