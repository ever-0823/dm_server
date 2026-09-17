"""校正后文本更新及 HTML 可见正文提取回归。"""

from app.knowledge import workflow
from app.ocr.glm_ocr import markdown_to_text


def test_rich_html_excludes_styles():
    """样式头不得进入检索内容，实体字符应还原。"""
    text = markdown_to_text(
        '<html><head><style>p {color:red}</style></head>'
        '<body><p>研发 &amp; 测试</p></body></html>'
    )
    assert "color" not in text
    assert "研发 & 测试" in text


def test_update_uses_existing_image_without_appending(monkeypatch):
    """更新路径复用原图片 ID，不再次创建物理原图或追加数据。"""
    monkeypatch.setattr(workflow.embedding, "encode_documents",
                        lambda texts: [[0.0] * workflow.settings.EMBEDDING_DIMENSIONS for _ in texts])
    calls = []
    def update(*args):
        calls.append(args)
        return {"id": 3, "image_id": 9, "updated": True}
    monkeypatch.setattr(workflow.store, "update_image_text", update)
    result = workflow.import_image("a.png", "测试", "image/png", b"image",
                                   "修正后的正文", [], "tester", 3, 9)
    assert result["updated"]
    assert calls[0][:2] == (3, 9)
    assert calls[0][3] == "修正后的正文"
