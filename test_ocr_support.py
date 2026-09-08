import json

from fastapi.testclient import TestClient

from app.app_factory import create_app
from app.dependencies.auth import current_user
from app.routers import ocr as ocr_router
from app.ocr.ppocrv6 import _to_lines


def test_ppocrv6_returns_normalized_result(monkeypatch):
    """使用模拟识别结果测试接口，避免测试时下载或加载模型。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    monkeypatch.setattr(
        ocr_router,
        "recognize_text",
        lambda content, suffix: [{"text": "设备编号", "score": 0.98}],
    )

    response = TestClient(app).post(
        "/api/ocr/ppocrv6",
        files={"file": ("device.png", b"image-content", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "text": "设备编号",
        "lines": [{"text": "设备编号", "score": 0.98}],
    }


def test_ocr_lines_keep_bbox() -> None:
    """OCR 矩形坐标应转换为前端和 JSONB 可直接使用的四点格式。"""
    lines = _to_lines(
        [{"res": {"rec_texts": ["设备编号"], "rec_scores": [0.98], "rec_boxes": [[1, 2, 11, 12]]}}]
    )
    assert lines[0]["bbox"] == [[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]]


def test_upload_ocr_image_to_knowledge_forwards_corrected_text(monkeypatch) -> None:
    """图片知识接口应传递原图、校正文本和 OCR 坐标。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    received: dict = {}

    def fake_import(filename, knowledge_name, content_type, content, corrected_text, lines, username, document_id=None):
        received.update(
            filename=filename,
            knowledge_name=knowledge_name,
            content=content,
            corrected_text=corrected_text,
            lines=lines,
            username=username,
            document_id=document_id,
        )
        return {"id": 1, "chunk_count": 1}

    monkeypatch.setattr(ocr_router.knowledge_workflow, "import_image", fake_import)
    try:
        response = TestClient(app).post(
            "/api/knowledge/upload-image",
            data={
                "knowledge_name": "扫码终端故障知识",
                "corrected_text": "设备编号 SCAN-TEST-001",
                "lines_json": json.dumps([{"text": "设备编号", "bbox": [[1, 2], [3, 2]]}]),
            },
            files={"file": ("device.png", b"image-content", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert received["filename"] == "device.png"
    assert received["knowledge_name"] == "扫码终端故障知识"
    assert received["corrected_text"] == "设备编号 SCAN-TEST-001"
    assert received["lines"][0]["bbox"] == [[1, 2], [3, 2]]


def test_upload_ocr_image_can_append_to_existing_knowledge(monkeypatch) -> None:
    """追加图片时应把稳定的知识库 ID 传给工作流。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    received: dict = {}

    def fake_import(*args):
        received["document_id"] = args[-1]
        return {"id": args[-1], "chunk_count": 3, "duplicate": False}

    monkeypatch.setattr(ocr_router.knowledge_workflow, "import_image", fake_import)
    try:
        response = TestClient(app).post(
            "/api/knowledge/upload-image",
            data={
                "document_id": "7",
                "corrected_text": "第二张设备图片",
                "lines_json": "[]",
            },
            files={"file": ("device-2.png", b"image-content-2", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert received["document_id"] == 7
