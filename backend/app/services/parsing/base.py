"""Normalized parse output shared by all document parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

ElementType = Literal["title", "heading", "paragraph", "list_item", "table", "figure", "caption", "formula", "code", "footnote", "ocr_text"]


@dataclass
class ParsedElement:
    element_id: str
    type: ElementType
    text: str
    page_no: int  # 1-based original PDF page
    bbox: list[float] | None = None  # [x0, y0, x1, y1], PDF points, top-left origin
    level: int | None = None  # heading level
    caption: str | None = None  # tables / figures
    extraction_method: str = "docling"
    page_end: int | None = None


@dataclass
class PageInfo:
    page_no: int
    width: float
    height: float
    text_layer_chars: int = 0
    parsed_chars: int = 0
    garbled_ratio: float = 0.0


@dataclass
class ParsedDocument:
    parser: str
    page_count: int
    title: str | None
    metadata: dict
    pages: dict[int, PageInfo]
    elements: list[ParsedElement] = field(default_factory=list)


class DocumentParser(Protocol):
    name: str

    def parse(self, path: str) -> ParsedDocument: ...


_HEADING_NUM = re.compile(r"^\s*((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+\S")


def heading_level(text: str, default: int = 1) -> int:
    """Infer heading depth from numbering such as '3', '3.2', '3.2.1' or 'A.1'."""
    m = _HEADING_NUM.match(text)
    if not m:
        return default
    return min(6, m.group(1).count(".") + 1)


def garbled_ratio(text: str) -> float:
    """Fraction of characters that indicate a broken text layer (replacement chars, private-use glyphs, controls)."""
    if not text:
        return 0.0
    bad = sum(1 for ch in text if ch == "�" or 0xE000 <= ord(ch) <= 0xF8FF or (ord(ch) < 32 and ch not in "\n\r\t"))
    return bad / len(text)
