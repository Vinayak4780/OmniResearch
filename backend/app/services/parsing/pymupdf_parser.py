"""Fallback parser using PyMuPDF text blocks. Used only when Docling is unavailable or fails;
documents parsed this way are labelled ``parser=pymupdf`` and have no table structure."""

from __future__ import annotations

import re
import statistics

from app.services.parsing.base import ParsedDocument, ParsedElement, PageInfo, garbled_ratio, heading_level

_CAPTION = re.compile(r"^\s*(Table|Figure|Fig\.)\s+\d+[:.]", re.I)


class PyMuPDFParser:
    name = "pymupdf"

    def parse(self, path: str) -> ParsedDocument:
        import pymupdf

        elements: list[ParsedElement] = []
        pages: dict[int, PageInfo] = {}
        with pymupdf.open(path) as pdf:
            meta = pdf.metadata or {}
            sizes: list[float] = []
            raw: list[tuple[int, list]] = []
            for i, page in enumerate(pdf):
                layer = page.get_text("text")
                pages[i + 1] = PageInfo(i + 1, page.rect.width, page.rect.height, len(layer.strip()), len(layer), garbled_ratio(layer))
                blocks = page.get_text("dict")["blocks"]
                raw.append((i + 1, blocks))
                for b in blocks:
                    for line in b.get("lines", []):
                        for span in line["spans"]:
                            if span["text"].strip():
                                sizes.append(span["size"])
            body = statistics.median(sizes) if sizes else 10.0
            n = 0
            for page_no, blocks in raw:
                for b in blocks:
                    if b.get("type") != 0:
                        continue
                    spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join(" ".join(s["text"] for s in line["spans"]).strip() for line in b["lines"]).strip()
                    text = re.sub(r"\s+", " ", text)
                    size = max(s["size"] for s in spans)
                    bold = all("bold" in s["font"].lower() or (s["flags"] & 16) for s in spans)
                    if _CAPTION.match(text):
                        etype = "caption"
                    elif (size > body * 1.15 or bold) and len(text) < 150:
                        etype = "heading"
                    else:
                        etype = "paragraph"
                    n += 1
                    elements.append(
                        ParsedElement(
                            element_id=f"#/blocks/{n}",
                            type=etype,  # type: ignore[arg-type]
                            text=text,
                            page_no=page_no,
                            bbox=[round(x, 1) for x in b["bbox"]],
                            level=heading_level(text, 1 if size > body * 1.4 else 2) if etype == "heading" else None,
                            extraction_method="pymupdf",
                        )
                    )
        title = meta.get("title") or next((e.text for e in elements if e.type == "heading"), None)
        return ParsedDocument(parser="pymupdf", page_count=len(pages), title=title, metadata={k: v for k, v in meta.items() if v}, pages=pages, elements=elements)
