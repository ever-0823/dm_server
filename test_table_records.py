"""规则表格检索回归，不连接数据库或模型。"""

import io
import json
import pytest

from app.knowledge import answering, table_records, table_merge, workflow, store


def sample(kind="comparison"):
    """同前缀型号、跨页来源和空值共同覆盖表格问答风险。"""
    return {
        "id": "table-1", "title": "规格参数", "table_type": kind,
        "headers": ["参数", "G1", "G1-EDU"],
        "rows": [["二次开发", "/", "有"], ["重量", "35kg", "35kg+"]],
        "row_sources": [{"image_id": 1, "page_number": 5}, {"image_id": 2, "page_number": 6}],
    }


def test_comparison_query_does_not_mix_models():
    """型号精确匹配，未定义符号保持原值。"""
    rows = table_records.matching_records("G1能否二次开发", sample())
    assert len(rows) == 1
    assert "原表值：/" in rows[0]["text"]
    assert "G1-EDU" not in rows[0]["text"]
    assert rows[0]["source"]["page_number"] == 5
    assert len(table_records.matching_records("G1规格参数", sample())) == 2
    assert not table_records.matching_records("G10能否二次开发", sample())
    assert not table_records.matching_records("G1能否飞行", sample())


def test_record_table_preserves_units_and_blank():
    """订单明细使用列名对应值，不假定首列是参数。"""
    group = {
        "title": "订单", "headers": ["订单号", "金额（元）", "备注"],
        "rows": [["AB-001", "100", ""]], "table_type": "records",
    }
    text = table_records.matching_records("AB-001金额", group)[0]["text"]
    assert "金额（元）：100" in text
    assert "备注：（空白）" in text


def test_chunk_display_and_search_are_separate(monkeypatch):
    """短表整体存储，长表按行分组且展示表头完整。"""
    monkeypatch.setattr(table_merge.settings, "KNOWLEDGE_CHUNK_SIZE", 2000)
    chunks = table_merge._merged_table_chunks(sample())
    assert len(chunks) == 1
    assert "| 参数 | G1 | G1-EDU |" in chunks[0]["display_content"]
    assert "对象：G1\n参数：二次开发\n原表值：/" in chunks[0]["content"]
    assert "页码：6" in chunks[0]["content"]


def test_table_evidence_reaches_streaming_model(monkeypatch):
    """明确字段命中不受向量阈值拦截，最终仍调用模型流式回答。"""
    captured = []
    def generate(request, **_kwargs):
        """捕获提示词验证来源与符号约束，而非依赖模型猜测。"""
        captured.append(json.loads(request.data)["prompt"])
        return io.BytesIO((json.dumps({"response": "原表标记为/，含义未明确。", "done": True}) + "\n").encode())
    monkeypatch.setattr(answering, "urlopen", generate)
    result = {
        "content": table_records.search_text(sample()), "score": 0.0,
        "original_name": "规格", "page_number": 5, "match_type": "table_fields",
    }
    answer = answering.collect_answer(answering.stream_answer("G1能否二次开发", [result]))
    assert captured and "不得混用型号" in captured[0]
    assert "原表值：/" in captured[0]
    assert answer["answer"] == "原表标记为/，含义未明确。"


def test_field_recall_is_independent_of_vector_top_k(monkeypatch):
    """向量排名未命中时，独立字段召回仍然进入问答候选。"""
    evidence = {"document_id": 7, "match_type": "table_fields", "content": "订单号：AB-001"}
    monkeypatch.setattr(workflow.embedding, "encode_query",
                        lambda _q: [0.0] * workflow.settings.EMBEDDING_DIMENSIONS)
    monkeypatch.setattr(workflow.store, "search_chunks", lambda *_a: [])
    monkeypatch.setattr(workflow.store, "search_table_records", lambda *_a: [evidence])
    assert workflow.search("AB-001金额", 1) == [evidence]


def test_stored_table_recall_has_all_page_references(monkeypatch):
    """查询归档结构可召回所有相关页，且关键词命中不冒充余弦相似度。"""
    class Cursor:
        """用只读游标替代 PostgreSQL，测试真实检索函数的映射逻辑。"""
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def execute(self, sql):
            assert "source_metadata" in sql
        def __iter__(self):
            return iter([{"id": 7, "original_name": "参数", "source_type": "image_set",
                          "source_metadata": {"table_merges": [sample()]}}])
        def cursor(self):
            return self
    monkeypatch.setattr(store, "ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_connect", Cursor)
    result = store.search_table_records("G1规格参数", 5)[0]
    assert result["score"] == 0.0
    assert result["match_type"] == "table_fields"
    assert [ref["page_number"] for ref in result["references"]] == [5, 6]
    assert "35kg+" not in result["content"]


def test_reindex_failure_does_not_replace_old_chunks(monkeypatch):
    """所有向量生成成功前，不调用替换事务。"""
    group = sample()
    group["selections"] = [{"image_id": 1, "table_index": 0}, {"image_id": 2, "table_index": 0}]
    snapshot = {
        "id": 7, "name": "参数", "metadata": {"table_merges": [group]},
        "images": [{"id": i, "page_number": i, "markdown_content": "| 参数 | G1 | G1-EDU |\n| --- | --- | --- |\n| 重量 | 35kg | 35kg+ |"}
                   for i in (1, 2)],
    }
    committed = []
    monkeypatch.setattr(store, "table_merge_snapshot", lambda _id: snapshot)
    monkeypatch.setattr(store, "commit_table_merge", lambda *_args: committed.append(True))
    def fail(_texts):
        """模拟模型不可用，不允许先删除旧向量。"""
        raise RuntimeError("embedding unavailable")
    monkeypatch.setattr(table_merge.embedding, "encode_documents", fail)
    with pytest.raises(RuntimeError):
        table_merge.reindex(7)
    assert committed == []


def test_table_merge_type_and_reindex_api(monkeypatch):
    """接口限制表格类型，重建入口只委托后台操作。"""
    from fastapi.testclient import TestClient
    from app.app_factory import create_app
    from app.dependencies.auth import current_user
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"username": "tester"}
    monkeypatch.setattr(table_merge, "reindex", lambda doc: {"id": doc, "chunk_count": 2})
    with TestClient(app) as client:
        response = client.post("/api/knowledge/documents/7/table-merge/reindex")
        assert response.status_code == 200
        assert response.json()["data"]["chunk_count"] == 2
        response = client.post("/api/knowledge/documents/7/table-merge/preview", json={
            "table_type": "guess", "selections": [
                {"image_id": 1, "table_index": 0}, {"image_id": 2, "table_index": 0}
            ],
        })
        assert response.status_code == 422
