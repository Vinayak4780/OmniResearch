"""Notebooks: a set of sources plus a source-grounded chat, a guide, notes and deep-research reports.

Chat answers are quick single-pass RAG over the selected sources (hybrid retrieval + reranking +
one model call), with the same citation discipline as reports: every factual sentence cites a
passage, citations are checked to exist, and numbers that do not appear in the cited passages are
flagged. Deep research uses the full multi-step agent and is listed in the notebook's Studio.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from app.agents.tools import chunk_to_evidence
from app.core.text import extract_numbers
from app.observability.metrics import current_job_id
from app.providers.base import Task
from app.repositories.db import utcnow
from app.services.citations.validator import _IGNORABLE_NUMBER
from app.services.research.llm import chat_json
from app.services.research.report import coerce_citations

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

PASSAGES = 8
WEB_PASSAGES = 5
HISTORY = 6


def grounding_check(text: str, mentioned: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    """Unknown citation ids, and numbers in the text that occur in none of the cited passages."""
    cites = [c for c in mentioned if c in evidence]
    ev_numbers = {n for c in cites for n in extract_numbers(evidence[c].text)}
    unsupported = sorted(n for n in extract_numbers(re.sub(r"\[E\d+\]", "", text)) if not _IGNORABLE_NUMBER.match(n) and n not in ev_numbers)
    return {"unknown_citations": [c for c in mentioned if c not in evidence], "numbers_not_in_cited_evidence": unsupported}


class NotebookAnswer(BaseModel):
    answer: str = Field(description="Markdown answer. Cite passages inline as [E2]. Say plainly if the sources do not answer the question.")
    citations: list[str] = Field(default_factory=list)
    answered_from_sources: bool = True

    @field_validator("citations", mode="before")
    @classmethod
    def _c(cls, v: Any) -> list[str]:
        return coerce_citations(v)


class NotebookGuide(BaseModel):
    summary: str = Field(description="3–5 sentence overview of what the sources cover, in plain language")
    questions: list[str] = Field(description="Exactly 3 useful questions a reader could ask about these sources")


class NotebookService:
    def __init__(self, c: "Container"):
        self.c = c
        self.db = c.db

    # ---------------------------------------------------------------- notebooks
    def _row(self, nb_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM notebooks WHERE id=?", (nb_id,))
        if not row:
            raise KeyError(nb_id)
        return row

    def list(self) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM notebooks ORDER BY updated_at DESC")
        for r in rows:
            r["source_count"] = self.db.one("SELECT COUNT(*) n FROM documents WHERE collection=?", (r["id"],))["n"]
            r["report_count"] = self.db.one("SELECT COUNT(*) n FROM research_jobs WHERE notebook_id=?", (r["id"],))["n"]
        return rows

    def get(self, nb_id: str) -> dict[str, Any]:
        r = self._row(nb_id)
        r["sources"] = self.sources(nb_id)
        return r

    def create(self, title: str) -> dict[str, Any]:
        title = (title or "").strip()[:120] or "Untitled notebook"
        nb_id = "nb-" + uuid.uuid4().hex[:10]
        now = utcnow()
        self.db.execute("INSERT INTO notebooks(id, title, created_at, updated_at) VALUES (?,?,?,?)", (nb_id, title, now, now))
        return self.get(nb_id)

    def rename(self, nb_id: str, title: str) -> dict[str, Any]:
        self._row(nb_id)
        self.db.execute("UPDATE notebooks SET title=?, updated_at=? WHERE id=?", ((title or "").strip()[:120] or "Untitled notebook", utcnow(), nb_id))
        return self.get(nb_id)

    def touch(self, nb_id: str) -> None:
        self.db.execute("UPDATE notebooks SET updated_at=? WHERE id=?", (utcnow(), nb_id))

    def delete(self, nb_id: str) -> None:
        self._row(nb_id)
        if self.db.one("SELECT 1 FROM research_jobs WHERE notebook_id=? AND status IN ('queued','running')", (nb_id,)):
            raise ValueError("A deep research job is still running in this notebook; cancel it first")
        for d in self.db.all("SELECT id FROM documents WHERE collection=?", (nb_id,)):
            self.c.ingestion.delete(d["id"])
        for j in self.db.all("SELECT id FROM research_jobs WHERE notebook_id=?", (nb_id,)):
            self.c.research.delete(j["id"])
        self.db.execute("DELETE FROM notebooks WHERE id=?", (nb_id,))  # messages and notes cascade

    def sources(self, nb_id: str) -> list[dict[str, Any]]:
        rows = self.db.all(
            "SELECT id, filename, title, status, error, page_count, size_bytes, parser, source_url, created_at, updated_at FROM documents WHERE collection=? ORDER BY created_at",
            (nb_id,),
        )
        for r in rows:
            job = self.db.one("SELECT status, stage, progress FROM ingestion_jobs WHERE document_id=? ORDER BY created_at DESC LIMIT 1", (r["id"],))
            r["job"] = job
        return rows

    def _ready_ids(self, nb_id: str, selected: list[str] | None) -> list[str]:
        ready = [s["id"] for s in self.sources(nb_id) if s["status"].startswith("ready")]
        if selected is not None:
            ready = [d for d in ready if d in set(selected)]
        return ready

    # ---------------------------------------------------------------- guide
    async def guide(self, nb_id: str) -> dict[str, Any]:
        self._row(nb_id)
        ids = self._ready_ids(nb_id, None)
        if not ids:
            return {"summary": None, "questions": [], "sources_hash": None, "reason": "Add a source to get a summary and suggested questions."}
        key_hash = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]
        cached = self.db.kv_get(f"nb_guide:{nb_id}")
        if cached and cached.get("sources_hash") == key_hash:
            return cached
        # the opening passages of each source (title, abstract, introduction) are enough for an overview
        blocks = []
        for d in ids[:12]:
            doc = self.db.one("SELECT filename, title FROM documents WHERE id=?", (d,))
            chunks = self.db.all("SELECT text FROM chunks WHERE document_id=? ORDER BY chunk_index LIMIT 3", (d,))
            text = "\n".join(c["text"] for c in chunks)[:2500]
            blocks.append(f'<source title="{(doc["title"] or doc["filename"]).replace(chr(34), "")}">\n{text}\n</source>')
        g = await chat_json(
            self.c.router,
            Task.CHAT,
            NotebookGuide,
            system="You write a short orientation for a notebook of research sources: what they cover, and three good questions to ask.",
            user="\n\n".join(blocks),
            max_tokens=1500,
            temperature=0.2,
        )
        out = {"summary": g.summary.strip(), "questions": [q.strip() for q in g.questions if q.strip()][:3], "sources_hash": key_hash}
        self.db.kv_set(f"nb_guide:{nb_id}", out)
        return out

    # ---------------------------------------------------------------- chat
    def messages(self, nb_id: str) -> list[dict[str, Any]]:
        self._row(nb_id)
        rows = self.db.all("SELECT * FROM notebook_messages WHERE notebook_id=? ORDER BY id", (nb_id,))
        for r in rows:
            r["citations"] = json.loads(r.pop("citations_json"))
            r["check"] = json.loads(r.pop("check_json"))
            r["source_ids"] = json.loads(r.pop("source_ids_json"))
        return rows

    def clear_chat(self, nb_id: str) -> None:
        self._row(nb_id)
        self.db.execute("DELETE FROM notebook_messages WHERE notebook_id=?", (nb_id,))

    async def ask(self, nb_id: str, message: str, source_ids: list[str] | None = None, web: bool = False) -> dict[str, Any]:
        self._row(nb_id)
        message = (message or "").strip()
        if not 2 <= len(message) <= 4000:
            raise ValueError("Message must be 2–4000 characters")
        scope = self._ready_ids(nb_id, source_ids)
        if not scope and not web:
            raise ValueError("Select at least one processed source to chat with")
        if web and not self.c.web.available()[0]:
            from app.services.notebook_web import NO_SEARCH

            raise ValueError(NO_SEARCH)
        current_job_id.set(f"notebook:{nb_id}")
        history = self.messages(nb_id)[-HISTORY:]
        # Follow-ups ("what about its memory use?") need the previous question to retrieve the right passages.
        prev_q = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        query = f"{message}\n{prev_q}" if prev_q and len(message) < 80 else message
        chunks, warnings = [], []
        if scope:
            res = await self.c.retriever.search(query, k=PASSAGES, document_ids=scope)
            chunks, warnings = res.chunks, list(res.warnings)
        found = [chunk_to_evidence(ch, "chat", query) for ch in chunks]
        if web:
            # Fast answer: one web search, the top pages fetched once, the most relevant passages kept (full pages first).
            web_ev, web_warn = await self.c.web.research_many([(message, None)])
            found += sorted(web_ev, key=lambda e: e.snippet_only)[:WEB_PASSAGES]
            warnings += web_warn
        evidence = {}
        for i, ev in enumerate(found, 1):
            ev.id = f"E{i}"
            evidence[ev.id] = ev
        convo = "\n".join(f"{h['role'].upper()}: {h['content'][:1200]}" for h in history)
        if evidence:
            blocks = "\n\n".join(e.prompt_block(1400) for e in evidence.values())
            out = await chat_json(
                self.c.router,
                Task.CHAT,
                NotebookAnswer,
                system=(
                    "You are a research assistant answering questions about the user's sources, using ONLY the passages provided "
                    "(type=local passages come from the user's documents, type=web from a web search; prefer the user's documents and say "
                    "when a point comes only from the web). "
                    "Write a clear, well-structured answer (short paragraphs or bullet points). Cite every factual sentence inline "
                    "with the passage ids it comes from, like [E2] or [E1][E3], and list them in 'citations'. Copy numbers exactly with "
                    "units and conditions. If the passages do not contain the answer, say so plainly and set answered_from_sources=false; "
                    "never fill gaps from your own knowledge."
                ),
                user=f"CONVERSATION SO FAR:\n{convo or '(none)'}\n\nPASSAGES:\n{blocks}\n\n<question>\n{message}\n</question>",
                max_tokens=3000,
                temperature=0.1,
            )
        else:
            out = NotebookAnswer(answer="I couldn't find anything about that in the selected sources.", citations=[], answered_from_sources=False)
        mentioned = list(dict.fromkeys(coerce_citations(out.citations) + coerce_citations(out.answer)))
        cites = [c for c in mentioned if c in evidence]
        check = {"answered_from_sources": out.answered_from_sources, **grounding_check(out.answer, mentioned, evidence), "retrieval_warnings": warnings, "web": web}
        cited_ev = {c: evidence[c].model_dump() | {"link": evidence[c].link()} for c in cites}
        insert = "INSERT INTO notebook_messages(notebook_id, role, content, citations_json, check_json, source_ids_json, created_at) VALUES (?,?,?,?,?,?,?)"
        self.db.execute(insert, (nb_id, "user", message, "{}", "{}", json.dumps(scope), utcnow()))
        a_id = self.db.execute(insert, (nb_id, "assistant", out.answer, json.dumps(cited_ev, default=str), json.dumps(check), json.dumps(scope), utcnow())).lastrowid
        self.touch(nb_id)
        return next(m for m in self.messages(nb_id) if m["id"] == a_id)

    # ---------------------------------------------------------------- notes
    def notes(self, nb_id: str) -> list[dict[str, Any]]:
        self._row(nb_id)
        rows = self.db.all("SELECT * FROM notebook_notes WHERE notebook_id=? ORDER BY id DESC", (nb_id,))
        for r in rows:
            r["citations"] = json.loads(r.pop("citations_json"))
        return rows

    def add_note(self, nb_id: str, content: str, title: str | None = None, message_id: int | None = None) -> dict[str, Any]:
        self._row(nb_id)
        citations: dict[str, Any] = {}
        if message_id is not None:
            msg = next((m for m in self.messages(nb_id) if m["id"] == message_id and m["role"] == "assistant"), None)
            if msg is None:
                raise KeyError(f"message {message_id}")
            content, citations = msg["content"], msg["citations"]
            q = self.db.one("SELECT content FROM notebook_messages WHERE notebook_id=? AND id<? AND role='user' ORDER BY id DESC LIMIT 1", (nb_id, message_id))
            title = title or (q["content"][:80] if q else None)
        content = (content or "").strip()
        if not content:
            raise ValueError("A note needs some text")
        title = (title or content.split("\n", 1)[0])[:120]
        nid = self.db.execute(
            "INSERT INTO notebook_notes(notebook_id, title, content, citations_json, created_at) VALUES (?,?,?,?,?)",
            (nb_id, title, content[:20000], json.dumps(citations, default=str), utcnow()),
        ).lastrowid
        self.touch(nb_id)
        return next(n for n in self.notes(nb_id) if n["id"] == nid)

    def delete_note(self, nb_id: str, note_id: int) -> None:
        self.db.execute("DELETE FROM notebook_notes WHERE notebook_id=? AND id=?", (nb_id, note_id))

    # ---------------------------------------------------------------- deep research (Studio)
    def start_research(self, nb_id: str, question: str, source_ids: list[str] | None, web_mode: str | None) -> dict[str, Any]:
        self._row(nb_id)
        opts: dict[str, Any] = {"notebook_id": nb_id, "web_mode": web_mode or "auto"}
        scope = self._ready_ids(nb_id, source_ids)
        if source_ids is not None:
            opts["document_ids"] = scope or ["__none__"]
        else:
            opts["collections"] = [nb_id]
        job_id = self.c.research.create(question, opts)
        self.touch(nb_id)
        return self.c.research.get(job_id)

    def reports(self, nb_id: str) -> list[dict[str, Any]]:
        self._row(nb_id)
        return self.db.all(
            "SELECT id, question, status, stage, error, created_at, updated_at FROM research_jobs WHERE notebook_id=? ORDER BY created_at DESC", (nb_id,)
        )
