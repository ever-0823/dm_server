import io
import json

from app import knowledge
from app.knowledge import answering, workflow


def test_ask_without_relevant_context_does_not_call_model(monkeypatch) -> None:
    # 低相关问题直接返回知识不足，避免模型脱离资料自由发挥。
    monkeypatch.setattr(workflow, "search", lambda _query, _top_k: [{"score": 0.2}])
    result = knowledge.ask("无关问题", 3)
    assert result["answer"] == "知识库中没有足够信息回答这个问题。"
    assert result["sources"] == []


def test_stream_knowledge_answer_returns_chinese_deltas(monkeypatch) -> None:
    """Ollama 的多行响应应保持中文，并按 delta 事件逐段返回。"""
    result = {
        "original_name": "设备规范.txt",
        "page_number": 1,
        "content": "维护前切断电源。",
        "context": "维护前切断电源。",
        "score": 0.9,
    }
    monkeypatch.setattr(workflow, "search", lambda _query, _top_k: [result])
    response = io.BytesIO(
        b"\n".join(
            json.dumps(item, ensure_ascii=False).encode("utf-8")
            for item in (
                {"message": {"content": "维护前"}, "done": False},
                {"message": {"content": "切断电源。"}, "done": True},
            )
        )
    )
    monkeypatch.setattr(answering, "urlopen", lambda *_args, **_kwargs: response)

    events = list(knowledge.stream_answer("维护前做什么？", 3))

    assert [event["type"] for event in events] == ["metadata", "delta", "delta", "done"]
    assert "".join(event.get("content", "") for event in events) == "维护前切断电源。"
