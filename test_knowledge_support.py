from app.knowledge.documents import extract_pages, split_text
from app.knowledge.qa import parse_qa_response
from app.knowledge import workflow
from app.knowledge.workflow import preview_document
from app.core.config import settings


def test_split_text_keeps_overlap_and_content() -> None:
    # 小尺寸参数用于快速验证切分不会丢失正文首尾。
    text = "第一段设备说明。\n第二段维护步骤。\n第三段故障处理。"
    chunks = split_text(text, chunk_size=12, overlap=3)
    assert len(chunks) > 1
    assert chunks[0].startswith("第一段")
    assert chunks[-1].endswith("故障处理。")


def test_extract_txt_supports_utf8_bom() -> None:
    # 系统导出的带 BOM 中文 TXT 应保持原文内容。
    pages = extract_pages("设备知识".encode("utf-8-sig"), ".txt")
    assert pages == [(1, "设备知识")]


def test_split_text_keeps_numbered_answer_with_question() -> None:
    # 编号问题和其后回答应优先落在同一文本块，避免只检索到问题标题。
    text = "1. 维护设备前要做什么？\n必须先切断电源并悬挂警示牌。\n2. 故障如何处理？\n先检查电源。"
    chunks = split_text(text, chunk_size=80, overlap=10)
    assert any("维护设备前要做什么" in chunk and "切断电源" in chunk for chunk in chunks)


def test_split_text_keeps_table_of_contents_in_one_chunk() -> None:
    # 目录需要作为一个完整文本块，不能因超过普通块大小被拆成多段。
    text = "目录\n" + "\n".join(f"第{i}章 设备管理制度 ........ {i}" for i in range(1, 31))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert chunks == [text]


def test_parse_qa_response_returns_searchable_content() -> None:
    # QA 内容统一落成问题和答案两行，便于后续直接向量化检索。
    result = parse_qa_response("问题：维护前需要做什么？\n答案：必须先切断电源。")
    assert result == {
        "question": "维护前需要做什么？",
        "answer": "必须先切断电源。",
        "content": "问题：维护前需要做什么？\n答案：必须先切断电源。",
    }


def test_preview_document_uses_existing_chunk_rules() -> None:
    # 预览只提取和切分内容，不依赖 Embedding、Ollama 或 pgvector。
    result = preview_document("设备规范.txt", "维护前切断电源。".encode("utf-8"))
    assert result["page_count"] == 1
    assert result["chunk_count"] == 1
    assert result["items"][0]["content"] == "维护前切断电源。"


def test_search_prioritizes_body_match_over_table_of_contents(monkeypatch) -> None:
    """正文和目录同时命中时，应优先返回正文文本块。"""
    toc = {
        "chunk_id": 1,
        "content": "目录\n二、资源使用 ........................................ 14",
        "context": "目录\n二、资源使用 ........................................ 14",
        "score": 0.31,
    }
    body = {
        "chunk_id": 2,
        "content": "1) 员工未经批准，不得将公司资产赠予、转让、出租、出借。",
        "context": "二、资源使用\n" + "1) 员工未经批准，不得将公司资产赠予、转让、出租、出借。",
        "score": 0.29,
    }
    monkeypatch.setattr(
        workflow.embedding,
        "encode_query",
        lambda _query: [0.0] * settings.EMBEDDING_DIMENSIONS,
    )
    monkeypatch.setattr(
        workflow.store,
        "search_chunks",
        lambda _vector, _limit: [toc, body],
    )

    result = workflow.search("资源使用", 1)

    assert result == [body]


def test_image_query_prioritizes_ocr_source(monkeypatch) -> None:
    """明确提到图片的问题应去除指代词并优先返回 OCR 图片来源。"""
    encoded_queries: list[str] = []
    document = {"chunk_id": 1, "content": "电源指示灯", "context": "电源指示灯", "score": 0.9, "source_type": "document"}
    image = {"chunk_id": 2, "content": "电源指示灯", "context": "电源指示灯", "score": 0.8, "source_type": "image"}
    monkeypatch.setattr(
        workflow.embedding,
        "encode_query",
        lambda query: encoded_queries.append(query) or [0.0] * settings.EMBEDDING_DIMENSIONS,
    )
    monkeypatch.setattr(workflow.store, "search_chunks", lambda _vector, _limit: [document, image])

    result = workflow.search("图片里的电源指示灯", 1)

    assert encoded_queries == ["电源指示灯"]
    assert result == [image]


def test_import_image_separates_display_and_search_text(monkeypatch, tmp_path) -> None:
    """图片入库应向量化带元数据文本，但归档展示用户校正后的原文。"""
    saved: dict = {}
    monkeypatch.setattr(settings, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        workflow.embedding,
        "encode_documents",
        lambda texts: saved.update(vector_texts=texts) or [[0.0] * settings.EMBEDDING_DIMENSIONS for _ in texts],
    )
    monkeypatch.setattr(
        workflow.store,
        "append_image",
        lambda *args: saved.update(args=args) or {"id": 1, "chunk_count": len(args[9])},
    )

    result = workflow.import_image(
        "SCAN-TEST-001.jpg",
        "扫码终端故障知识",
        "image/jpeg",
        b"image",
        "电源指示灯不亮",
        [{"text": "电源指示灯不亮", "bbox": [[1, 2], [3, 2], [3, 4], [1, 4]]}],
        "tester",
    )

    chunks = saved["args"][9]
    assert result["chunk_count"] == 1
    assert chunks[0]["display_content"] == "电源指示灯不亮"
    assert "知识库名称：扫码终端故障知识" in saved["vector_texts"][0]
    assert saved["args"][0] is None
    assert saved["args"][1] == "扫码终端故障知识"
    assert saved["args"][2] == "SCAN-TEST-001.jpg"


if __name__ == "__main__":
    test_split_text_keeps_overlap_and_content()
    test_extract_txt_supports_utf8_bom()
    print("ok")
