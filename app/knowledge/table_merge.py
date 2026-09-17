"""用户确认的跨图片续表合并，不根据知识库归属自动推断表格关系。"""

import hashlib
import json
import re
from uuid import uuid4

from app.core.config import settings
from app.core.exceptions import AppException
from app.knowledge import documents, embedding, store, table_records
from app.ocr.glm_ocr import _html_cells, markdown_to_text


def extract_tables(markdown: str) -> list[dict]:
    """定位 HTML 或管道表格，保留原文区间以便只替换选中的区域。"""
    tables = []
    pattern = r"<table\b.*?</table>|(?:^[^\n]*\|[^\n]*(?:\n|$))+"
    for match in re.finditer(pattern, markdown, re.I | re.S | re.M):
        raw = match.group()
        error = ""
        header_rowspan = False
        if raw.lower().startswith("<table"):
            cells = _html_cells(raw)
            # 只展开纵向合并，不猜测横向合并字段的语义；归档 HTML 不作改动。
            header_rowspan = any(c["row"] == 0 and c["row_span"] > 1 for c in cells)
            if any(c["column_span"] != 1 for c in cells):
                error = "含横向合并单元格，请先校正为单层表头和规则列"
            if len(re.findall(r"<table\b", raw, re.I)) > 1:
                error = "嵌套表格暂不支持续表合并，请先校正结构"
            row_count = max(len(re.findall(r"<tr\b", raw, re.I)),
                            max((c["row"] + 1 for c in cells), default=0))
            column_count = max((c["column"] + c["column_span"] for c in cells), default=0)
            rows = []
            if not error and cells:
                # 用坐标补齐 rowspan 覆盖的格子，不能简单按每行已有 td 顺序拼接。
                grid = {}
                for cell in cells:
                    if cell["row"] + cell["row_span"] > row_count:
                        error = "纵向合并范围超出表格实际行数，请先校正"
                        break
                    for row in range(cell["row"], cell["row"] + cell["row_span"]):
                        grid[row, cell["column"]] = cell["text"]
                if not error:
                    if len(grid) != row_count * column_count:
                        error = "表格存在缺失单元格或列数不一致，请先校正"
                    else:
                        rows = [[grid[row, col] for col in range(column_count)] for row in range(row_count)]
            # 有问题的表格仍占据原索引，避免前端选表序号漂移到另一张表。
            if error:
                rows = [[c["text"] for c in cells]]
        else:
            rows = []
            for line in raw.splitlines():
                # 仅去掉 Markdown 外框；保留内部空单元格以及 / 的原始含义。
                cells = [v.strip() for v in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
                if all(re.fullmatch(r":?-{3,}:?", v or "") for v in cells):
                    continue
                rows.append([v.replace(r"\|", "|") for v in cells])
        if len(rows) < 1:
            continue
        tables.append({
            "index": len(tables), "start": match.start(), "end": match.end(),
            "rows": rows, "error": error,
            "header_rowspan": header_rowspan,
        })
    return tables


def table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """输出一个具有明确列标题的 Markdown 表格，转义单元格管道符。"""
    def line(values):
        """避免单元格中的换行和竖线被误识别成新行或新列。"""
        return "| " + " | ".join(v.replace("|", r"\|").replace("\n", "<br>") for v in values) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(r) for r in rows)])


def revision(images: list[dict], metadata: dict) -> str:
    """绑定预览时的数据版本，防止用户确认期间图片或合并状态发生变化。"""
    value = [(i["id"], i["markdown_content"]) for i in images]
    return hashlib.sha256(json.dumps([value, metadata], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def options(document_id: int) -> dict:
    """读取可合并表格及已保存的逻辑表，不返回本地文件路径。"""
    snapshot = store.table_merge_snapshot(document_id)
    return {
        "images": [
            {"id": i["id"], "name": i["original_name"], "page": i["page_number"],
             "tables": extract_tables(i["markdown_content"])}
            for i in snapshot["images"]
        ],
        "merges": snapshot["metadata"].get("table_merges", []),
    }


def build_merge(snapshot: dict, payload: dict) -> dict:
    """按用户指定的图片和表格顺序拼接，列不匹配时拒绝猜测。"""
    selections = payload["selections"]
    ids = [s["image_id"] for s in selections]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise AppException(400, "至少选择两张不同图片，每张图片选择一个表格")
    occupied = {s["image_id"] for g in snapshot["metadata"].get("table_merges", []) for s in g["selections"]}
    if occupied.intersection(ids):
        raise AppException(409, "选中图片已参与合并，请先撤销原合并")
    images = {i["id"]: i for i in snapshot["images"]}
    parts = []
    headers = [h.strip() for h in payload.get("headers", [])]
    all_rows, sources = [], []
    for position, selection in enumerate(selections):
        image = images.get(selection["image_id"])
        if image is None:
            raise AppException(404, "选中的图片不属于当前知识库")
        tables = extract_tables(image["markdown_content"])
        index = selection["table_index"]
        if not 0 <= index < len(tables):
            raise AppException(400, "选中的表格不存在")
        table = tables[index]
        if table["error"]:
            raise AppException(400, table["error"])
        # 续页首行可以是纵向合并的数据，但明确作为表头时不允许跨行。
        if selection.get("skip_header") and table["header_rowspan"]:
            raise AppException(400, "所选表头含纵向合并，请校正为单层表头；续页数据不要勾选首行为表头")
        rows = list(table["rows"])
        if not position and not headers:
            if not selection.get("skip_header"):
                raise AppException(400, "请指定表头，或勾选首图的首行为表头")
            headers = [v or ("参数" if n == 0 else f"列{n + 1}") for n, v in enumerate(rows[0])]
        if len(headers) < 2 or not all(headers) or len(set(headers)) != len(headers):
            raise AppException(400, "表头至少两列，列名不能为空或重复")
        # 表头是否跳过必须由每张图片明确指定，避免跨图续表时把表头写成数据。
        if rows[0] == headers and not selection.get("skip_header"):
            raise AppException(400, f"{image['original_name']}：检测到首行与表头相同，请勾选跳过首行表头")
        if selection.get("skip_header"):
            # 续页表头列序或单位发生变化时必须重新确认，不能仅按列数强行拼接。
            if position and rows[0] != headers:
                raise AppException(400, f"{image['original_name']}：表头或列数与表头不同，请校正列名、顺序和单位后重试")
            rows = rows[1:]
        if not rows or any(len(row) != len(headers) for row in rows):
            raise AppException(400, f"{image['original_name']}：数据为空或列数与表头不同，请校正后重试")
        all_rows.extend(rows)
        sources.extend([{"image_id": image["id"], "page_number": image["page_number"]}] * len(rows))
        parts.append((image, table, rows))
    group = {
        "id": uuid4().hex, "title": payload["title"].strip() or "合并表格",
        "table_type": payload.get("table_type", "records"),
        "headers": headers, "rows": all_rows, "row_sources": sources,
        "selections": selections, "markdown": table_markdown(headers, all_rows),
    }
    return {"group": group, "parts": parts, "revision": revision(snapshot["images"], snapshot["metadata"])}


def preview(document_id: int, payload: dict) -> dict:
    """生成无副作用的完整预览，尚不调用 Embedding 或写数据库。"""
    built = build_merge(store.table_merge_snapshot(document_id), payload)
    return {"group": built["group"], "revision": built["revision"]}


def _original_chunks(image: dict) -> list[dict]:
    """撤销时从归档原文重新生成该图片原始文本块。"""
    return [{"source_image_id": image["id"], "page_number": image["page_number"], "content": text}
             for text in documents.split_text(markdown_to_text(image["markdown_content"]),
                                             settings.KNOWLEDGE_CHUNK_SIZE, settings.KNOWLEDGE_CHUNK_OVERLAP)]


def _merged_table_chunks(group: dict) -> list[dict]:
    """把完整续表按长度切分，并为每个文本块重复写入表头。"""
    headers = list(group["headers"])
    rows = list(group["rows"])
    sources = list(group.get("row_sources") or [])
    chunks: list[dict] = []
    current_rows: list[list[str]] = []
    current_sources: list[dict] = []

    def make_chunk(batch: list[list[str]], refs: list[dict]) -> dict:
        """同一批数据生成展示与检索文本，跨页来源随正文保留。"""
        subset = {**group, "rows": batch, "row_sources": refs}
        source = refs[0] if refs else {}
        provenance = "\n".join(
            f"图片ID：{image_id}，页码：{page}"
            for image_id, page in dict.fromkeys(
                (ref.get("image_id"), ref.get("page_number", 1)) for ref in refs
            )
        )
        return {
            "source_image_id": source.get("image_id"), "page_number": source.get("page_number", 1),
            "content": table_records.search_text(subset) + "\n来源：\n" + provenance,
            "display_content": group["title"] + "\n" + table_markdown(headers, batch),
        }

    for index, row in enumerate(rows):
        candidate_rows = current_rows + [row]
        candidate = make_chunk(candidate_rows, current_sources + [sources[index]])
        # 表头和已有数据超过限制时，从下一行开始新的文本块，避免拆断单行数据。
        if current_rows and max(len(candidate["content"]), len(candidate["display_content"])) > settings.KNOWLEDGE_CHUNK_SIZE:
            chunks.append(make_chunk(current_rows, current_sources))
            current_rows = []
            current_sources = []
        current_rows.append(row)
        current_sources.append(sources[index] if index < len(sources) else {})

    if current_rows:
        chunks.append(make_chunk(current_rows, current_sources))
    return chunks


def apply(document_id: int, payload: dict) -> dict:
    """确认后重建所选图片的索引，完整逻辑表保存在文档元数据中。"""
    snapshot = store.table_merge_snapshot(document_id)
    built = build_merge(snapshot, payload)
    if payload.get("revision") != built["revision"]:
        raise AppException(409, "图片或合并状态已变化，请重新预览")
    chunks = []
    group = built["group"]
    # 先保留每张图片中表格以外的正文，合并表格本身统一生成文本块。
    for image, table, rows in built["parts"]:
        original = image["markdown_content"]
        remaining = original[:table["start"]] + "\n\n" + original[table["end"]:]
        chunks.extend(_original_chunks({**image, "markdown_content": remaining}))
    # 表格较短时一张表一个文本块；过长时按行切分，每个块都会保留完整表头。
    chunks.extend(_merged_table_chunks(group))
    metadata = {**snapshot["metadata"], "table_merges": snapshot["metadata"].get("table_merges", []) + [group]}
    return _commit(snapshot, [p[0]["id"] for p in built["parts"]], chunks, metadata)


def undo(document_id: int, group_id: str) -> dict:
    """撤销指定合并并恢复所有参与图片的原始索引。"""
    snapshot = store.table_merge_snapshot(document_id)
    groups = snapshot["metadata"].get("table_merges", [])
    group = next((g for g in groups if g["id"] == group_id), None)
    if group is None:
        raise AppException(404, "合并表格不存在")
    ids = [s["image_id"] for s in group["selections"]]
    chunks = [chunk for i in snapshot["images"] if i["id"] in ids for chunk in _original_chunks(i)]
    return _commit(snapshot, ids, chunks, {**snapshot["metadata"], "table_merges": [g for g in groups if g["id"] != group_id]})


def reindex(document_id: int) -> dict:
    """从归档表格重建该知识库索引；向量失败或版本变化时保留原索引。"""
    snapshot = store.table_merge_snapshot(document_id)
    groups = snapshot["metadata"].get("table_merges", [])
    if not groups:
        raise AppException(400, "当前知识库没有已确认的合并表格")
    selections = {s["image_id"]: s for group in groups for s in group["selections"]}
    chunks = []
    for image in snapshot["images"]:
        if image["id"] not in selections:
            continue
        tables = extract_tables(image["markdown_content"])
        index = selections[image["id"]]["table_index"]
        if index >= len(tables):
            raise AppException(409, "原始表格发生变化，请重新确认合并")
        table = tables[index]
        remaining = image["markdown_content"][:table["start"]] + "\n\n" + image["markdown_content"][table["end"]:]
        chunks.extend(_original_chunks({**image, "markdown_content": remaining}))
    for group in groups:
        chunks.extend(_merged_table_chunks(group))
    return _commit(snapshot, list(selections), chunks, snapshot["metadata"])


def _commit(snapshot: dict, ids: list[int], chunks: list[dict], metadata: dict) -> dict:
    """先生成全部向量，再在事务中替换；失败时保留旧数据。"""
    vectors = embedding.encode_documents([snapshot["name"] + "\n" + c["content"] for c in chunks])
    if len(vectors) != len(chunks) or any(len(v) != settings.EMBEDDING_DIMENSIONS for v in vectors):
        raise AppException(500, "向量数量或维度不正确")
    return store.commit_table_merge(snapshot, ids, chunks, vectors, metadata)
