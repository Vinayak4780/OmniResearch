"""Unified evidence records. Provenance is carried unchanged from retrieval to the final report."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

_FENCE_TAG = re.compile(r"<\s*(/?)\s*(source|visual_observation|question)\b", re.I)
_ATTR_UNSAFE = re.compile(r'[<>"\r\n]')
_ATTR_REPLACEMENT = {"<": "‹", ">": "›", '"': "'"}


def _attr(v: object) -> str:
    """Attribute values from documents/webpages cannot close the tag or inject attributes."""
    return _ATTR_UNSAFE.sub(lambda m: _ATTR_REPLACEMENT.get(m.group(0), " "), str(v))


def neutralize(text: str) -> str:
    """Stop source text from opening or closing the untrusted-data fences."""
    return _FENCE_TAG.sub(lambda m: f"‹{m.group(1)}{m.group(2)}", text)


class Evidence(BaseModel):
    key: str  # stable dedupe key: chunk id or url#passage-hash
    id: str | None = None  # E1, E2, ... assigned when selected
    source_type: Literal["local", "web"]
    text: str  # verbatim excerpt from the source
    subquery: str | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)  # scores, ranks, retrieval mode
    relevance: float = 0.0

    # Local PDF provenance
    document_id: str | None = None
    filename: str | None = None
    doc_title: str | None = None
    page_no: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    element_type: str | None = None
    extraction_method: str | None = None
    chunk_id: str | None = None
    bbox: list[float] | None = None
    source_element_id: str | None = None

    # Web provenance
    url: str | None = None
    final_url: str | None = None
    canonical_url: str | None = None
    domain: str | None = None
    web_title: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    search_engine: str | None = None
    snippet_only: bool = False  # page could not be fetched; text is a search-engine snippet

    # Visual inspection (vision-language model output; never presented as source text)
    visual_observation: str | None = None
    visual_model: str | None = None
    inspected_visually: bool = False

    def citation_label(self) -> str:
        if self.source_type == "local":
            pages = f"p. {self.page_no}" if self.page_end in (None, self.page_no) else f"pp. {self.page_no}–{self.page_end}"
            return f"{self.doc_title or self.filename}, {pages}"
        return f"{self.web_title or self.domain} ({self.domain})"

    def link(self) -> str:
        if self.source_type == "local":
            return f"/documents/{self.document_id}/view?page={self.page_no}" + (f"&evidence={self.id}" if self.id else "")
        return self.canonical_url or self.final_url or self.url or ""

    def prompt_block(self, max_chars: int = 1500) -> str:
        """Evidence rendered for an LLM, inside explicit untrusted-data delimiters."""
        meta = [f"id={self.id}", f"type={self.source_type}"]
        if self.source_type == "local":
            meta += [
                f'doc="{_attr(self.doc_title or self.filename)}"',
                f"page={self.page_no}",
                f'section="{_attr(self.section_path or "")}"',
                f"element={_attr(self.element_type)}",
                f"extraction={_attr(self.extraction_method)}",
            ]
        else:
            meta += [f"domain={_attr(self.domain)}", f'title="{_attr((self.web_title or "")[:120])}"', f"published={_attr(self.published_at or 'unknown')}"]
            if self.snippet_only:
                meta.append("snippet_only=true")
        out = f"<source {' '.join(meta)}>\n{neutralize(self.text[:max_chars])}\n</source>"
        if self.visual_observation:
            out += f"\n<visual_observation of={self.id} model={_attr(self.visual_model)}>\n{neutralize(self.visual_observation[:800])}\n</visual_observation>"
        return out


def web_key(url: str, passage: str) -> str:
    return f"{url}#{hashlib.sha1(passage.encode()).hexdigest()[:10]}"


def domain_of(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host[4:] if host.startswith("www.") else host
