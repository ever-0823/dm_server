import io
import json

from app import knowledge
from app.core.exceptions import AppException
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
                {"response": "维护前", "done": False},
                {"response": "切断电源。FINAL_ANSWER_END", "done": True},
            )
        )
    )
    monkeypatch.setattr(answering, "urlopen", lambda *_args, **_kwargs: response)

    events = list(knowledge.stream_answer("维护前做什么？", 3))

    assert [event["type"] for event in events] == ["metadata", "delta", "delta", "done"]
    assert "".join(event.get("content", "") for event in events) == "维护前切断电源。"


def test_short_exact_query_uses_matching_knowledge_chunk(monkeypatch) -> None:
    """短关键词命中正文后仍通过模型流式生成完整回答。"""
    result = {
        "original_name": "员工手册.pdf",
        "page_number": 13,
        "content": "一、行为准则\n1. 经营活动\n员工不得从事危害公司利益的行为。",
        "context": "一、行为准则\n1. 经营活动\n员工不得从事危害公司利益的行为。",
        "score": 0.32,
    }
    monkeypatch.setattr(workflow, "search", lambda _query, _top_k: [result])
    response = io.BytesIO(
        ('{"response":"经营活动相关规定共有四项。FINAL_ANSWER_END","done":true}\n').encode("utf-8")
    )
    model_called = False

    def fake_urlopen(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        return response

    monkeypatch.setattr(answering, "urlopen", fake_urlopen)

    events = list(knowledge.stream_answer("经营活动", 5))

    assert events[0]["sources"]
    assert model_called is True
    assert "经营活动相关规定共有四项。" == "".join(
        event.get("content", "") for event in events if event["type"] == "delta"
    )


def test_short_topic_prompt_requires_all_numbered_items(monkeypatch) -> None:
    """短主题词的模型提示必须包含完整章节和明确的条目数量。"""
    result = {
        "original_name": "员工手册.pdf",
        "page_number": 14,
        "content": "1) 第一项。\n2) 第二项。\n3) 第三项。\n4) 第四项。",
        "context": "2. 资源使用\n1) 第一项。\n2) 第二项。\n3) 第三项。\n4) 第四项。\n5) 第五项。\n3. 保密义务\n1) 其它内容。",
        "score": 0.3,
    }
    captured_payload = {}

    def fake_urlopen(request, **_kwargs):
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return io.BytesIO(
            ('{"response":"已完整回答五项。FINAL_ANSWER_END","done":true}\n').encode("utf-8")
        )

    monkeypatch.setattr(workflow, "search", lambda _query, _top_k: [result])
    monkeypatch.setattr(answering, "urlopen", fake_urlopen)

    list(knowledge.stream_answer("资源使用", 5))

    prompt = captured_payload["prompt"]
    assert "共有 5 个编号条目" in prompt
    assert "5) 第五项" in prompt
    assert "3. 保密义务" not in prompt
    assert "FINAL_ANSWER_START" in prompt
    assert "FINAL_ANSWER_END" in prompt
    assert captured_payload["raw"] is True


def test_stream_answer_bypasses_qwen_thinking_template(monkeypatch) -> None:
    """raw 生成应预填最终答案标记，不再进入 qwen3 的强制思考模板。"""
    result = {
        "original_name": "员工手册.pdf",
        "page_number": 17,
        "content": "员工奖励包括通报表扬和通报嘉奖。",
        "context": "员工奖励包括通报表扬和通报嘉奖。",
        "score": 0.8,
    }
    response = io.BytesIO(
        b"\n".join(
            json.dumps(event, ensure_ascii=False).encode("utf-8")
            for event in (
                {"response": "最终", "done": False},
                {"response": "答案FINAL_ANSWER_END", "done": True},
            )
        )
    )
    captured_request = None

    def fake_urlopen(request, **_kwargs):
        nonlocal captured_request
        captured_request = request
        return response

    monkeypatch.setattr(workflow, "search", lambda _query, _top_k: [result])
    monkeypatch.setattr(answering, "urlopen", fake_urlopen)

    events = list(knowledge.stream_answer("员工奖励", 5))
    answer = "".join(event.get("content", "") for event in events if event["type"] == "delta")

    assert answer == "最终答案"
    assert captured_request.full_url.endswith("/api/generate")
    payload = json.loads(captured_request.data.decode("utf-8"))
    assert payload["prompt"].endswith("FINAL_ANSWER_START")


def test_ollama_timeout_returns_stream_error_and_non_stream_exception(monkeypatch) -> None:
    """Ollama 超时应给流式前端明确错误，非流式入口继续返回业务异常。"""
    result = {
        "original_name": "员工手册.pdf",
        "page_number": 1,
        "content": "员工上班时间为上午九点。",
        "context": "员工上班时间为上午九点。",
        "score": 0.9,
    }
    def fail_urlopen(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(answering, "urlopen", fail_urlopen)

    events = list(answering.stream_answer("员工上班时间", [result]))

    assert events[-1] == {"type": "error", "message": "无法连接 Ollama，请确认服务已启动"}
    try:
        answering.collect_answer(iter(events))
    except AppException as exc:
        assert exc.code == 503
        assert exc.message == "无法连接 Ollama，请确认服务已启动"
    else:
        raise AssertionError("非流式入口必须把 error 事件恢复为 AppException")
