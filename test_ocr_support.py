from fastapi.testclient import TestClient

from app.app_factory import create_app
from app.dependencies.auth import current_user
from app.routers import ocr as ocr_router


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
