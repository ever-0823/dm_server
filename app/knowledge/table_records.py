"""规则表格的字段关联与精确匹配，不解释空值和业务符号。"""

import re


def records(group: dict) -> list[dict]:
    """保留每行来源；对比表按列对象展开，其他表保留完整记录。"""
    headers = group["headers"]
    result = []
    for index, row in enumerate(group["rows"]):
        if len(row) != len(headers):
            continue
        source = (group.get("row_sources") or [{}] * len(group["rows"]))[index]
        if group.get("table_type") == "comparison":
            for column in range(1, len(headers)):
                result.append({
                    "labels": [headers[column], row[0]],
                    "text": f"对象：{headers[column]}\n{headers[0]}：{row[0]}\n原表值：{row[column] or '（空白）'}",
                    "source": source, "row_index": index,
                })
        else:
            # 未指定类型的历史表不猜测列含义，同时保留行标识和全部列名。
            result.append({
                "labels": list(headers) + list(row),
                "text": "\n".join(f"{key}：{value or '（空白）'}" for key, value in zip(headers, row)),
                "source": source, "row_index": index,
            })
    return result


def search_text(group: dict) -> str:
    """字段名随每条记录保存，避免模型只看到孤立数值。"""
    return f"表名：{group['title']}\n\n" + "\n\n".join(item["text"] for item in records(group))


def query_terms(query: str) -> list[str]:
    """去除有限的问句外壳；保留型号、编号和中文业务词，不使用模糊前缀。"""
    text = query.casefold().strip()
    text = re.sub(r"^(?:请问|请介绍|请列出|请说明|查询|搜索)", "", text)
    text = re.sub(r"是否可以|是否支持|能否|是否|可以|支持|是多少|有哪些|是什么", " ", text)
    text = re.sub(r"[吗呢的？?，,。:：]", " ", text)
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u4e00-\u9fff]+", text)


def _matches(term: str, text: str) -> bool:
    """型号使用 ASCII 边界，G1 不会误命中 G1-EDU。"""
    if term.isascii():
        return re.search(r"(?<![a-z0-9_.-])" + re.escape(term) + r"(?![a-z0-9_.-])",
                         text.casefold()) is not None
    return term in text.casefold()


def matching_records(query: str, group: dict) -> list[dict]:
    """所有有效查询词须同时落在同一记录，泛问只在有对象约束时放宽。"""
    terms = query_terms(query)
    if not terms:
        return []
    identifiers = [term for term in terms if term.isascii()]
    if identifiers:
        terms = [term for term in terms if term not in {"规格参数", "参数", "规格", "全部参数"}]
    return [
        item for item in records(group)
        if all(_matches(term, "\n".join(item["labels"])) for term in terms)
    ]
