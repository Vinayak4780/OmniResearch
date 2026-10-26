"""Citation validation.

Two clearly separated layers:
 * **Mechanical** (deterministic): cited evidence exists, its source exists (document + page range /
   fetched URL), the evidence excerpt occurs verbatim in the source, and numbers in the claim appear
   in the cited evidence.
 * **Semantic** (model-based, VERIFICATION route): does the cited evidence support the claim?
   This is a model judgment, reported as such — not proof.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.text import extract_numbers, match_key
from app.providers.base import NoProviderAvailableError, ProviderError, Task
from app.providers.router import ProviderRouter
from app.repositories.db import Database
from app.services.research.evidence import Evidence
from app.services.research.llm import StructuredOutputError, chat_json

_CITE = re.compile(r"\[(E\d+)\]")
# numbers that are rarely factual claims: years and single digits used as list/section numbers
_IGNORABLE_NUMBER = re.compile(r"^(19|20)\d\d$|^\d$")


class MechanicalCheck(BaseModel):
    ok: bool
    citations_exist: bool
    sources_exist: bool
    excerpts_match: bool
    numbers_supported: bool
    unsupported_numbers: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class SemanticVerdict(BaseModel):
    statement_id: str
    verdict: Literal["supported", "partially_supported", "not_supported", "cannot_judge"]
    rationale: str = ""


class SemanticBatch(BaseModel):
    verdicts: list[SemanticVerdict]


class EvidenceCheck(BaseModel):
    evidence_id: str
    source_exists: bool
    excerpt_match: bool
    detail: str = ""


class CitationValidator:
    def __init__(self, db: Database, router: ProviderRouter):
        self.db = db
        self.router = router
        self._page_cache: dict[tuple[str, int], str | None] = {}

    def _page_text(self, doc_id: str, page_no: int) -> str | None:
        key = (doc_id, page_no)
        if key not in self._page_cache:
            row = self.db.one("SELECT text FROM pages WHERE document_id=? AND page_no=?", (doc_id, page_no))
            self._page_cache[key] = row["text"] if row else None
        return self._page_cache[key]

    def check_evidence(self, e: Evidence, web_pages: dict[str, Any]) -> EvidenceCheck:
        if e.source_type == "local":
            doc = self.db.one("SELECT page_count FROM documents WHERE id=?", (e.document_id,))
            if not doc:
                return EvidenceCheck(evidence_id=e.id or e.key, source_exists=False, excerpt_match=False, detail="document no longer exists")
            if not e.page_no or e.page_no < 1 or (doc["page_count"] and e.page_no > doc["page_count"]):
                return EvidenceCheck(evidence_id=e.id or e.key, source_exists=False, excerpt_match=False, detail=f"page {e.page_no} out of range")
            pages = range(e.page_no, (e.page_end or e.page_no) + 1)
            text = " ".join(t for p in pages if (t := self._page_text(e.document_id, p)))  # type: ignore[arg-type]
            ok = bool(text) and match_key(e.text)[:400] in match_key(text)
            return EvidenceCheck(evidence_id=e.id or e.key, source_exists=True, excerpt_match=ok, detail="" if ok else "excerpt not found on cited page(s)")
        page = web_pages.get(e.url or "")
        if e.snippet_only:
            return EvidenceCheck(evidence_id=e.id or e.key, source_exists=True, excerpt_match=False, detail="search snippet only; page not fetched")
        if page is None:
            return EvidenceCheck(evidence_id=e.id or e.key, source_exists=False, excerpt_match=False, detail="page not in fetch log")
        page_text = page.text if hasattr(page, "text") else str(page)
        ok = match_key(e.text) in match_key(page_text)
        return EvidenceCheck(evidence_id=e.id or e.key, source_exists=True, excerpt_match=ok, detail="" if ok else "excerpt not found in fetched page")

    def mechanical(self, text: str, citations: list[str], evidence: dict[str, Evidence], ev_checks: dict[str, EvidenceCheck]) -> MechanicalCheck:
        problems: list[str] = []
        cited = [evidence[c] for c in citations if c in evidence]
        exist = len(cited) == len(citations) and bool(citations)
        if not citations:
            problems.append("no citation")
        missing = [c for c in citations if c not in evidence]
        if missing:
            problems.append(f"unknown evidence ids {missing}")
        src_ok = all(ev_checks[e.id].source_exists for e in cited if e.id in ev_checks) and bool(cited)
        exc_ok = all(ev_checks[e.id].excerpt_match for e in cited if e.id in ev_checks) and bool(cited)
        for e in cited:
            chk = ev_checks.get(e.id or "")
            if chk and not (chk.source_exists and chk.excerpt_match):
                problems.append(f"{e.id}: {chk.detail}")
        claim_numbers = {n for n in extract_numbers(_CITE.sub("", text)) if not _IGNORABLE_NUMBER.match(n)}
        ev_numbers: set[str] = set()
        for e in cited:
            ev_numbers |= set(extract_numbers(e.text))
            if e.visual_observation:
                pass  # numbers read off a chart by a VLM are not counted as source-text support
        unsupported = sorted(n for n in claim_numbers if n not in ev_numbers and n.lstrip("-") not in ev_numbers)
        if unsupported:
            problems.append(f"numbers not found in cited evidence: {unsupported}")
        ok = exist and src_ok and exc_ok and not unsupported
        return MechanicalCheck(
            ok=ok,
            citations_exist=exist,
            sources_exist=src_ok,
            excerpts_match=exc_ok,
            numbers_supported=not unsupported,
            unsupported_numbers=unsupported,
            problems=problems,
        )

    async def semantic(self, items: list[dict[str, Any]], evidence: dict[str, Evidence], batch_size: int = 8) -> tuple[dict[str, SemanticVerdict], list[str]]:
        """items: [{id, text, citations}] → verdicts by statement id. Returns (verdicts, warnings)."""
        verdicts: dict[str, SemanticVerdict] = {}
        warnings: list[str] = []
        if not items:
            return verdicts, warnings
        system = (
            "You are a strict fact-checking assistant. For each statement, decide whether the cited sources support it. "
            "'supported' = every factual element (including numbers, units, conditions) is stated in the cited sources. "
            "'partially_supported' = some elements are supported, others are missing or overstated. "
            "'not_supported' = the sources do not state it or contradict it. 'cannot_judge' = sources are irrelevant/insufficient to judge. "
            "Visual observations are model interpretations of images, not source text; do not treat them as source statements. "
            "Judge only against the provided sources, not your own knowledge."
        )
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            used: dict[str, Evidence] = {}
            for it in batch:
                for c in it["citations"]:
                    if c in evidence:
                        used[c] = evidence[c]
            sources = "\n\n".join(e.prompt_block(1200) for e in used.values())
            stmts = "\n".join(f'- statement_id={it["id"]} cites={it["citations"]}: {it["text"]}' for it in batch)
            try:
                out = await chat_json(self.router, Task.VERIFICATION, SemanticBatch, system=system, user=f"SOURCES:\n{sources}\n\nSTATEMENTS:\n{stmts}", max_tokens=1500, temperature=0.0)
                for v in out.verdicts:
                    verdicts[v.statement_id] = v
            except (ProviderError, NoProviderAvailableError, StructuredOutputError) as exc:
                warnings.append(f"semantic verification unavailable for {len(batch)} statements: {exc}")
        return verdicts, warnings
