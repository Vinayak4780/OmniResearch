"""Follow-up questions about a finished investigation.

Answers are grounded in the investigation's evidence plus a fresh hybrid search over the same
document scope. New passages become new evidence ids on the job, so their citations open the PDF
page like any other. Cited numbers are checked mechanically against the cited evidence.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.text import extract_numbers
from app.observability.metrics import current_job_id
from app.providers.base import Task
from app.repositories.db import Database, utcnow
from app.services.research.evidence import Evidence
from app.services.citations.validator import _IGNORABLE_NUMBER
from app.services.research.llm import chat_json
from app.services.research.report import coerce_citations
from app.services.retrieval.hybrid import Retriever
from app.agents.tools import chunk_to_evidence

MAX_HISTORY = 6
FRESH_PASSAGES = 6


class FollowUpAnswer(BaseModel):
    answer: str = Field(description="Markdown answer. Cite evidence inline as [E3]. Say clearly if the sources do not answer the question.")
    citations: list[str] = Field(default_factory=list)
    answered_from_sources: bool = Field(default=True, description="False if the evidence does not contain the answer")

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> list[str]:
        return coerce_citations(v)


class FollowUpService:
    def __init__(self, db: Database, router, retriever: Retriever):
        self.db = db
        self.router = router
        self.retriever = retriever

    def history(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM research_chat WHERE job_id=? ORDER BY id", (job_id,))
        evidence = self._evidence(job_id) if rows else {}
        for r in rows:
            r["citations"] = json.loads(r.pop("citations_json"))
            r["check"] = json.loads(r.pop("check_json"))
            ids = list(dict.fromkeys(r["citations"] + coerce_citations(r["content"])))
            r["links"] = {c: evidence[c].link() for c in ids if c in evidence}
        return rows

    def _evidence(self, job_id: str) -> dict[str, Evidence]:
        rows = self.db.all("SELECT data_json FROM research_evidence WHERE job_id=?", (job_id,))
        evs = [Evidence.model_validate(json.loads(r["data_json"])) for r in rows]
        return {e.id: e for e in evs if e.id}

    async def ask(self, job: dict[str, Any], message: str) -> dict[str, Any]:
        job_id = job["id"]
        current_job_id.set(job_id)  # attribute follow-up calls to the job's usage
        if job["status"] != "completed":
            raise ValueError("Follow-up questions are available once the investigation has completed")
        message = message.strip()
        if not 2 <= len(message) <= 2000:
            raise ValueError("Message must be 2–2000 characters")
        evidence = self._evidence(job_id)

        # Fresh retrieval in the original scope; new passages get new evidence ids.
        opts = job.get("options") or {}
        res = await self.retriever.search(message, k=FRESH_PASSAGES, document_ids=opts.get("document_ids") or None, collections=opts.get("collections") or None)
        known_chunks = {e.chunk_id for e in evidence.values() if e.chunk_id}
        next_n = max([int(i[1:]) for i in evidence] or [0])
        added: list[str] = []
        for ch in res.chunks:
            if ch.chunk_id in known_chunks:
                continue
            next_n += 1
            ev = chunk_to_evidence(ch, "followup", message)
            ev.id = f"E{next_n}"
            evidence[ev.id] = ev
            added.append(ev.id)
            self.db.execute("INSERT INTO research_evidence(job_id, evidence_id, data_json) VALUES (?,?,?)", (job_id, ev.id, ev.model_dump_json()))

        history = self.history(job_id)[-MAX_HISTORY:]
        convo = "\n".join(f"{h['role'].upper()}: {h['content'][:1500]}" for h in history)
        report = (job.get("report_md") or "")[:6000]
        blocks = "\n\n".join(e.prompt_block(1000) for e in evidence.values())
        out = await chat_json(
            self.router,
            Task.SYNTHESIS,
            FollowUpAnswer,
            system=(
                "You answer follow-up questions about a research report using ONLY the evidence provided. "
                "Cite evidence ids inline like [E3] for every factual statement and list them in 'citations'. "
                "Copy numbers exactly with units and conditions. If the evidence does not answer the question, "
                "say so and set answered_from_sources=false; never fill gaps from memory."
            ),
            user=f"REPORT (for context):\n{report}\n\nCONVERSATION SO FAR:\n{convo}\n\nEVIDENCE:\n{blocks}\n\n<question>\n{message}\n</question>",
            max_tokens=3000,
            temperature=0.1,
        )
        mentioned = list(dict.fromkeys(coerce_citations(out.citations) + coerce_citations(out.answer)))
        cites = [c for c in mentioned if c in evidence]
        ev_numbers = {n for c in cites for n in extract_numbers(evidence[c].text)}
        claim_numbers = {n for n in extract_numbers(out.answer) if not _IGNORABLE_NUMBER.match(n)}  # "E3" ids are not matched
        unsupported = sorted(n for n in claim_numbers if n not in ev_numbers)
        check = {
            "answered_from_sources": out.answered_from_sources,
            "unknown_citations": [c for c in mentioned if c not in evidence],
            "numbers_not_in_cited_evidence": unsupported,
            "new_evidence": added,
        }
        insert = "INSERT INTO research_chat(job_id, role, content, citations_json, check_json, created_at) VALUES (?,?,?,?,?,?)"
        q_id = self.db.execute(insert, (job_id, "user", message, "[]", "{}", utcnow())).lastrowid
        a_at = utcnow()
        a_id = self.db.execute(insert, (job_id, "assistant", out.answer, json.dumps(cites), json.dumps(check), a_at)).lastrowid
        links = {c: evidence[c].link() for c in cites}
        return {
            "id": a_id,
            "question_id": q_id,
            "created_at": a_at,
            "answer": out.answer,
            "citations": cites,
            "links": links,
            "check": check,
            "evidence": [evidence[c].model_dump() for c in added],
        }
