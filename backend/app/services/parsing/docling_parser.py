"""Docling-based parser (primary). OCR is disabled inside Docling: pages that need OCR are routed
through the configured OCRProvider by the ingestion pipeline, so OCR stays provider-independent."""

from __future__ import annotations

import threading
from typing import Any

from app.services.parsing.base import ParsedDocument, ParsedElement, PageInfo, garbled_ratio, heading_level

_SKIP_LABELS = {"page_header", "page_footer", "document_index", "marker", "empty_value"}
_LABEL_MAP = {
    "title": "title",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "reference": "paragraph",
    "list_item": "list_item",
    "table": "table",
    "picture": "figure",
    "chart": "figure",
    "caption": "caption",
    "formula": "formula",
    "code": "code",
    "footnote": "footnote",
    "handwritten_text": "paragraph",
    "key_value_region": "paragraph",
    "form": "paragraph",
}


class DoclingParser:
    name = "docling"
    _converter: Any = None
    _lock = threading.Lock()

    def __init__(self, do_table_structure: bool = True, document_timeout: float | None = 600):
        self.do_table_structure = do_table_structure
        self.document_timeout = document_timeout

    @staticmethod
    def available() -> bool:
        try:
            import docling  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_converter(self):
        with DoclingParser._lock:
            if DoclingParser._converter is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                opts = PdfPipelineOptions(do_ocr=False, do_table_structure=self.do_table_structure, document_timeout=self.document_timeout)
                DoclingParser._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
            return DoclingParser._converter

    def parse(self, path: str) -> ParsedDocument:
        import pymupdf

        result = self._get_converter().convert(path, raises_on_error=True)
        doc = result.document

        pages: dict[int, PageInfo] = {}
        with pymupdf.open(path) as pdf:
            meta = pdf.metadata or {}
            for i, p in enumerate(pdf):
                layer = p.get_text("text")
                pages[i + 1] = PageInfo(i + 1, p.rect.width, p.rect.height, len(layer.strip()), 0, garbled_ratio(layer))

        caption_refs: set[str] = set()
        for item, _ in doc.iterate_items():
            for cref in getattr(item, "captions", None) or []:
                caption_refs.add(getattr(cref, "cref", str(cref)))

        elements: list[ParsedElement] = []
        title: str | None = None
        for item, _depth in doc.iterate_items():
            label = item.label.value if hasattr(item.label, "value") else str(item.label)
            if label in _SKIP_LABELS or item.self_ref in caption_refs:
                continue
            etype = _LABEL_MAP.get(label)
            if etype is None:
                continue
            prov = item.prov[0] if getattr(item, "prov", None) else None
            page_no = prov.page_no if prov else 1
            page_end = item.prov[-1].page_no if getattr(item, "prov", None) else page_no
            bbox = None
            if prov is not None and page_no in pages:
                b = prov.bbox
                try:
                    tl = b.to_top_left_origin(page_height=pages[page_no].height)
                    bbox = [round(tl.l, 1), round(tl.t, 1), round(tl.r, 1), round(tl.b, 1)]
                except Exception:
                    bbox = None
            caption = None
            if etype == "table":
                text = item.export_to_markdown(doc=doc)
                caption = item.caption_text(doc) or None
                if caption and text.startswith(caption):
                    text = text[len(caption):].lstrip()
            elif etype == "figure":
                caption = item.caption_text(doc) or None
                text = caption or ""
            else:
                text = getattr(item, "text", "") or ""
            if not text.strip() and etype != "figure":
                continue
            level = None
            if etype in ("heading", "title"):
                level = getattr(item, "level", None) or 1
                level = max(level, heading_level(text, level))
                if title is None and (etype == "title" or not elements):
                    title = text.strip()
            elements.append(
                ParsedElement(
                    element_id=item.self_ref,
                    type=etype,  # type: ignore[arg-type]
                    text=text.strip(),
                    page_no=page_no,
                    page_end=page_end,
                    bbox=bbox,
                    level=level,
                    caption=caption,
                    extraction_method="docling",
                )
            )
            if page_no in pages:
                pages[page_no].parsed_chars += len(text)

        return ParsedDocument(
            parser="docling",
            page_count=len(pages),
            title=title or (meta.get("title") or None),
            metadata={k: v for k, v in meta.items() if v},
            pages=pages,
            elements=elements,
        )
