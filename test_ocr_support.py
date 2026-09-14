import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.app_factory import create_app
from app.dependencies.auth import current_user
from app.routers import ocr as ocr_router
from app.ocr import glm_ocr as ocr


def test_glm_ocr_returns_normalized_result(monkeypatch):
    """使用模拟识别结果测试接口，避免测试时下载或加载模型。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    monkeypatch.setattr(
        ocr_router,
        "parse_image",
        lambda content, suffix: {
            "markdown": "设备编号",
            "regions": [{"text": "设备编号", "label": "text", "score": None}],
            "provider": "selfhosted",
        },
    )

    response = TestClient(app).post(
        "/api/ocr/parse",
        files={"file": ("device.png", b"image-content", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "markdown": "设备编号",
        "regions": [{"text": "设备编号", "label": "text", "score": None}],
        "provider": "selfhosted",
    }


def test_glm_bbox_converts_to_image_pixels() -> None:
    """SDK 归一化坐标应转换为前端可直接使用的四点像素格式。"""
    assert ocr._bbox_points([100, 200, 500, 600], 200, 100) == [
        [20.0, 20.0],
        [100.0, 20.0],
        [100.0, 60.0],
        [20.0, 60.0],
    ]


def test_glm_result_keeps_layout_type_and_bbox(monkeypatch) -> None:
    """真实适配层应保留布局类型、像素坐标，并且不伪造文字置信度。"""
    image = BytesIO()
    Image.new("RGB", (200, 100), "white").save(image, format="PNG")
    monkeypatch.setattr(
        ocr,
        "_parse_result",
        lambda _content: (
            [[{
                "label": "text",
                "native_label": "doc_title",
                "content": "设备规范",
                "bbox_2d": [100, 200, 500, 600],
            }]],
            "设备规范",
        ),
    )
    assert ocr.parse_image(image.getvalue())["regions"] == [{
        "text": "设备规范",
        "label": "doc_title",
        "page_number": 1,
        "score": None,
        "bbox": [[20.0, 20.0], [100.0, 20.0], [100.0, 60.0], [20.0, 60.0]],
    }]


def test_glm_parser_allows_cold_model_connection(monkeypatch, tmp_path) -> None:
    """首次加载本地模型较慢时，连接预检也应使用可配置的等待时间。"""
    import glmocr

    captured: dict = {}

    class FakeGlmOcr:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(glmocr, "GlmOcr", FakeGlmOcr)
    monkeypatch.setattr(ocr.settings, "GLM_OCR_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ocr.settings, "GLM_OCR_CONNECT_TIMEOUT", 300)
    ocr._parser.cache_clear()
    try:
        ocr._parser()
    finally:
        ocr._parser.cache_clear()

    assert captured["_dotted"]["pipeline.ocr_api.connect_timeout"] == 300


def test_glm_parser_uses_official_api_when_configured(monkeypatch) -> None:
    """官方模式只初始化 MaaS 客户端，不加载本地 Ollama 布局管线。"""
    import glmocr

    captured: dict = {}

    class FakeGlmOcr:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(glmocr, "GlmOcr", FakeGlmOcr)
    monkeypatch.setattr(ocr.settings, "GLM_OCR_MODE", "maas")
    monkeypatch.setattr(ocr.settings, "ZHIPU_API_KEY", "sk-test")
    monkeypatch.setattr(ocr.settings, "GLM_OCR_API_URL", "https://example.test/ocr")
    monkeypatch.setattr(ocr.settings, "GLM_OCR_API_MODEL", "glm-ocr")
    ocr._parser.cache_clear()
    try:
        ocr._parser()
    finally:
        ocr._parser.cache_clear()

    assert captured["mode"] == "maas"
    assert captured["api_key"] == "sk-test"
    assert "_dotted" not in captured


def test_old_ppocr_route_is_removed() -> None:
    """旧 PP-OCRv6 路径必须彻底移除，避免继续暴露误导接口。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    try:
        response = TestClient(app).post(
            "/api/ocr/ppocrv6",
            files={"file": ("device.png", b"image-content", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404


def test_old_ocr_glm_route_is_removed() -> None:
    """旧 /ocr/glm 路径必须移除，避免继续暴露模型耦合接口。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    try:
        response = TestClient(app).post(
            "/api/ocr/glm",
            files={"file": ("device.png", b"image-content", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404


def test_upload_ocr_image_to_knowledge_forwards_corrected_text(monkeypatch) -> None:
    """图片知识接口应传递原图、校正 Markdown 和区域坐标。"""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    received: dict = {}

    def fake_import(filename, knowledge_name, content_type, content, markdown_content, regions, username, document_id=None):
        received.update(
            filename=filename,
            knowledge_name=knowledge_name,
            content=content,
            markdown_content=markdown_content,
            regions=regions,
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
                "markdown_content": "设备编号 SCAN-TEST-001",
                "regions_json": json.dumps([{"text": "设备编号", "bbox": [[1, 2], [3, 2]]}]),
            },
            files={"file": ("device.png", b"image-content", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert received["filename"] == "device.png"
    assert received["knowledge_name"] == "扫码终端故障知识"
    assert received["markdown_content"] == "设备编号 SCAN-TEST-001"
    assert received["regions"][0]["bbox"] == [[1, 2], [3, 2]]


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
                "markdown_content": "第二张设备图片",
                "regions_json": "[]",
            },
            files={"file": ("device-2.png", b"image-content-2", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert received["document_id"] == 7
