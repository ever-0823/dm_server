from fastapi.testclient import TestClient

from app.app_factory import create_app
from app.dependencies.auth import current_user


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
        "app.routers.knowledge.search_knowledge",
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
