import json

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
