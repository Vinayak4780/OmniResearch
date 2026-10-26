"""Structure-aware chunking.

Rules
 * Headings start a new section; the heading path is kept on every chunk (``section_path``).
 * Paragraph-like elements of one section are packed together up to ``max_chars``; a chunk never
   spans sections. Oversized paragraphs are split on sentence boundaries with overlap.
 * Tables become their own chunks (caption + markdown); large tables are split by rows and every
   part repeats the header row.
 * Figures become their own chunks (caption), keeping page and bounding box so the page can be
   rendered for visual inspection later.
 * OCR text is chunked like paragraphs but keeps ``extraction_method=ocr:<engine>``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from app.services.parsing.base import ParsedElement

CHUNK_VERSION = "struct-v1"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    page_no: int
    page_end: int
    section_path: str
    element_type: str
    extraction_method: str
    text: str
    bbox: list[float] | None = None
    source_element_id: str | None = None
    chunk_version: str = CHUNK_VERSION
    content_hash: str = ""
    element_ids: list[str] = field(default_factory=list)

    def embedding_text(self, title: str | None = None) -> str:
        """Text sent to embedders/BM25: adds document title and section context."""
        ctx = " > ".join(x for x in [title or "", self.section_path] if x)
        return f"{ctx}\n{self.text}" if ctx else self.text


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sents = _SENT.split(text)
    parts: list[str] = []
    cur = ""
    for s in sents:
        while len(s) > max_chars:  # pathological sentence
            head, s = s[:max_chars], s[max_chars:]
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(head)
        if cur and len(cur) + len(s) + 1 > max_chars:
            parts.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            cur = (tail + " " + s).strip()
        else:
            cur = (cur + " " + s).strip()
    if cur:
        parts.append(cur)
    return parts


def _split_table(md: str, max_chars: int) -> list[str]:
    lines = md.splitlines()
    if len(md) <= max_chars or len(lines) < 4:
        return [md]
    header, rows = lines[:2], lines[2:]
    out, cur = [], list(header)
    for r in rows:
        if len("\n".join(cur + [r])) > max_chars and len(cur) > 2:
            out.append("\n".join(cur))
            cur = list(header)
        cur.append(r)
    out.append("\n".join(cur))
    return out


def chunk_elements(document_id: str, elements: list[ParsedElement], *, max_chars: int = 1800, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []
    buf: list[ParsedElement] = []

    def section() -> str:
        return " > ".join(t for _, t in path)

    def emit(text: str, etype: str, method: str, page: int, page_end: int, bbox, source: str | None, ids: list[str]) -> None:
        text = text.strip()
        if not text:
            return
        idx = len(chunks)
        h = hashlib.sha256(f"{document_id}|{section()}|{text}".encode()).hexdigest()
        chunks.append(
            Chunk(
                id=f"{document_id[:12]}-{idx:05d}-{h[:8]}",
                document_id=document_id,
                chunk_index=idx,
                page_no=page,
                page_end=max(page, page_end),
                section_path=section(),
                element_type=etype,
                extraction_method=method,
                text=text,
                bbox=bbox,
                source_element_id=source,
                content_hash=h,
                element_ids=ids,
            )
        )

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        # group consecutive elements with the same extraction method
        groups: list[list[ParsedElement]] = []
        for el in buf:
            if groups and groups[-1][-1].extraction_method == el.extraction_method:
                groups[-1].append(el)
            else:
                groups.append([el])
        for g in groups:
            cur: list[ParsedElement] = []
            size = 0
            for el in g + [None]:  # type: ignore[list-item]
                if el is not None and (not cur or size + len(el.text) + 2 <= max_chars):
                    cur.append(el)
                    size += len(el.text) + 2
                    continue
                if cur:
                    text = "\n\n".join(e.text for e in cur)
                    etype = "ocr_text" if cur[0].type == "ocr_text" else ("list" if all(e.type == "list_item" for e in cur) else "text")
                    bbox = cur[0].bbox if len(cur) == 1 else None
                    for part in _split_long(text, max_chars, overlap):
                        emit(part, etype, cur[0].extraction_method, cur[0].page_no, max((e.page_end or e.page_no) for e in cur), bbox, cur[0].element_id, [e.element_id for e in cur])
                cur = [el] if el is not None else []
                size = len(el.text) + 2 if el is not None else 0
        buf = []

    for el in elements:
        if el.type in ("heading", "title"):
            flush()
            level = el.level or 1
            path = [(lv, t) for lv, t in path if lv < level] + [(level, el.text.strip()[:200])]
            continue
        if el.type == "table":
            flush()
            cap = (el.caption or "").strip()
            for i, part in enumerate(_split_table(el.text, max_chars)):
                label = cap + (f" (part {i + 1})" if i else "")
                emit(f"{label}\n{part}" if label else part, "table", el.extraction_method, el.page_no, el.page_end or el.page_no, el.bbox, el.element_id, [el.element_id])
            continue
        if el.type == "figure":
            flush()
            cap = (el.caption or el.text or "").strip()
            emit(cap or "[Figure without caption]", "figure", el.extraction_method, el.page_no, el.page_no, el.bbox, el.element_id, [el.element_id])
            continue
        if el.type == "caption":
            flush()
            emit(el.text, "caption", el.extraction_method, el.page_no, el.page_no, el.bbox, el.element_id, [el.element_id])
            continue
        if buf and (buf[-1].page_no != el.page_no and sum(len(b.text) for b in buf) > max_chars // 2):
            flush()
        buf.append(el)
    flush()
    return chunks
