from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any, Dict, List


class HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[Dict[str, Any]] = []
        self._in_table = False
        self._in_caption = False
        self._in_row = False
        self._in_cell = False
        self._current_table: Dict[str, Any] | None = None
        self._current_row: List[str] = []
        self._current_cell: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._current_table = {"caption": "", "rows": []}
        elif self._in_table and tag == "caption":
            self._in_caption = True
            self._current_cell = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_table and tag in {"td", "th"}:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "caption" and self._in_caption and self._current_table is not None:
            self._current_table["caption"] = _clean_text("".join(self._current_cell))
            self._in_caption = False
            self._current_cell = []
        elif tag in {"td", "th"} and self._in_cell:
            self._current_row.append(_clean_text("".join(self._current_cell)))
            self._in_cell = False
            self._current_cell = []
        elif tag == "tr" and self._in_row and self._current_table is not None:
            if any(cell.strip() for cell in self._current_row):
                self._current_table["rows"].append(self._current_row)
            self._in_row = False
            self._current_row = []
        elif tag == "table" and self._in_table:
            if self._current_table and self._current_table.get("rows"):
                self.tables.append(self._current_table)
            self._current_table = None
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_caption or self._in_cell:
            self._current_cell.append(data)


def contains_html_table(text: str) -> bool:
    return bool(re.search(r"<\s*(table|tr|td|th|caption)\b", text or "", flags=re.I))


def extract_html_tables(text: str) -> List[Dict[str, Any]]:
    parser = HtmlTableParser()
    parser.feed(text or "")
    captions = [_clean_text(item) for item in re.findall(r"<\s*caption[^>]*>(.*?)<\s*/\s*caption\s*>", text or "", flags=re.I | re.S)]
    tables = []
    for idx, table in enumerate(parser.tables):
        if not table.get("rows"):
            continue
        normalized = _normalize_table(table)
        if not normalized.get("caption") and idx < len(captions):
            normalized["caption"] = captions[idx]
        tables.append(normalized)
    return tables


def html_tables_to_structured_text(text: str) -> str:
    tables = extract_html_tables(text)
    if not tables:
        return _strip_html(text)

    parts: List[str] = []
    prefix = _strip_html(re.split(r"<\s*table\b", text, maxsplit=1, flags=re.I)[0]).strip()
    if prefix:
        parts.append(prefix)
    for idx, table in enumerate(tables, start=1):
        parts.append(_table_to_text(table, idx))
    return "\n\n".join(part for part in parts if part.strip())


def _normalize_table(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = _pad_rows(table.get("rows") or [])
    if not rows:
        return {"caption": table.get("caption", ""), "headers": [], "records": []}
    headers = rows[0]
    records: List[Dict[str, str]] = []
    last_values: Dict[str, str] = {}
    for row in rows[1:]:
        record: Dict[str, str] = {}
        for idx, header in enumerate(headers):
            key = header or f"列{idx + 1}"
            value = row[idx] if idx < len(row) else ""
            if value:
                last_values[key] = value
            elif key in last_values and _should_forward_fill(key):
                value = last_values[key]
            record[key] = value
        if any(record.values()):
            records.append(record)
    return {
        "caption": table.get("caption", ""),
        "headers": headers,
        "records": records,
    }


def _table_to_text(table: Dict[str, Any], idx: int) -> str:
    caption = table.get("caption") or f"表格{idx}"
    lines = [f"[结构化表格] {caption}"]
    headers = table.get("headers") or []
    if headers:
        lines.append("表头：" + " | ".join(headers))
    for record in table.get("records") or []:
        fields = [f"{key}={value}" for key, value in record.items() if value]
        if fields:
            lines.append("- " + "；".join(fields))
    return "\n".join(lines)


def _pad_rows(rows: List[List[str]]) -> List[List[str]]:
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def _should_forward_fill(key: str) -> bool:
    return key in {"缺陷部位", "类别", "分类", "部位", "对象", "设备类", "章节"}


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return _clean_text(text)


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
