"""Studio outputs: summary, briefing doc, study guide, FAQ, timeline and mind map for a notebook.

Each output is written from passages sampled across every selected source (not one search), so it
covers the whole notebook. The same citation discipline as chat applies: every statement cites
passage ids, unknown ids and numbers missing from the cited passages are flagged, and statements
without any citation are counted so the reader can see how much of the output is grounded.
Outputs run in the background and survive page reloads; a server restart marks running ones failed.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from app.agents.tools import chunk_to_evidence
from app.observability.logging import get_logger
from app.observability.metrics import current_job_id
from app.providers.base import Task
from app.repositories.db import utcnow
from app.services.notebooks import grounding_check
from app.services.research.llm import chat_json
from app.services.research.report import coerce_citations

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger("omni.studio")

PASSAGE_BUDGET = 28
PASSAGE_CHARS = 1200
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")

KINDS: dict[str, tuple[str, str]] = {
    "summary": (
        "Summary",
        "A detailed summary of the sources: the main topics, the key findings (with exact numbers), and how the sources relate "
        "to or disagree with each other. Use short headings and bullet points.",
    ),
    "briefing": (
        "Briefing doc",
        "A briefing document for a busy reader with these sections: '## Executive summary' (3–5 bullets), '## Key findings', "
        "'## Important numbers' (a bullet per number with its context), and '## Disagreements and open questions'.",
    ),
    "study_guide": (
        "Study guide",
        "A study guide with sections: '## Key concepts' (term — short explanation), '## Quiz' (8–10 short-answer questions, each "
        "followed by its answer), and '## Glossary'.",
    ),
    "faq": ("FAQ", "8–12 frequently asked questions about the sources, each as '### question' followed by a concise answer."),
    "timeline": (
        "Timeline",
        "A chronological timeline of dated events and developments in the sources, as bullets '**date** — event', oldest first; "
        "then '## Key people, organisations and systems' with one line each. If the sources contain few dates, say so plainly and "
        "list only what exists — never invent dates.",
    ),
    "mindmap": ("Mind map", ""),
}


class StudioDoc(BaseModel):
    content: str = Field(description="Markdown. Cite passages inline like [E2] or [E1][E4] after every factual statement.")


class MindNode(BaseModel):
    label: str = Field(description="Short phrase, at most 8 words")
    citations: list[str] = Field(default_factory=list, description="Passage ids supporting this node, e.g. ['E3']")
    children: list["MindNode"] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _c(cls, v: Any) -> list[str]:
        return coerce_citations(v)


class MindMapOut(BaseModel):
    central: str = Field(description="The notebook's central topic, at most 6 words")
    branches: list[MindNode] = Field(description="4–7 main themes, each with 2–5 children; at most 3 levels deep")


def _walk(nodes: list[MindNode], depth: int = 1) -> list[dict[str, Any]]:
    return [{"label": n.label.strip()[:120], "citations": n.citations, "children": _walk(n.children, depth + 1) if depth < 3 else []} for n in nodes[:8]]


def _labels(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n in nodes:
        out.append(n)
        out += _labels(n["children"])
    return out


def uncited_statements(md: str) -> int:
    """Lines that state something (bullets/paragraphs of some length, not headings or questions) without a citation."""
    n = 0
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.endswith("?") or s.startswith("|---") or len(re.sub(r"[*_`>\-|\d.\s]", "", s)) < 30:
            continue
        if not re.search(r"\[E\d+\]", s):
            n += 1
    return n


class NotebookStudioService:
    def __init__(self, c: "Container"):
        self.c = c
        self.db = c.db
        self._tasks: dict[int, asyncio.Task] = {}
        self.db.execute("UPDATE notebook_outputs SET status='failed', error='Interrupted by a server restart', updated_at=? WHERE status='running'", (utcnow(),))

    async def stop(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # ---------------------------------------------------------------- CRUD
    @staticmethod
    def _decode(r: dict[str, Any]) -> dict[str, Any]:
        for k in ("data", "citations", "check", "source_ids"):
            r[k] = json.loads(r.pop(f"{k}_json"))
        return r

    def list(self, nb_id: str) -> list[dict[str, Any]]:
        self.c.notebooks._row(nb_id)
        rows = self.db.all("SELECT * FROM notebook_outputs WHERE notebook_id=? ORDER BY id DESC", (nb_id,))
        return [self._decode(r) for r in rows]

    def get(self, nb_id: str, out_id: int) -> dict[str, Any]:
        r = self.db.one("SELECT * FROM notebook_outputs WHERE notebook_id=? AND id=?", (nb_id, out_id))
        if not r:
            raise KeyError(f"output {out_id}")
        return self._decode(r)

    def delete(self, nb_id: str, out_id: int) -> None:
        self.get(nb_id, out_id)
        if out_id in self._tasks:
            self._tasks[out_id].cancel()
        self.db.execute("DELETE FROM notebook_outputs WHERE notebook_id=? AND id=?", (nb_id, out_id))

    def create(self, nb_id: str, kind: str, source_ids: list[str] | None = None) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"Unknown output type '{kind}'")
        scope = self.c.notebooks._ready_ids(nb_id, source_ids)
        if not scope:
            raise ValueError("Select at least one processed source")
        if not self.c.router.is_configured(Task.CHAT):
            raise ValueError("This needs an AI model. Connect one in Settings → Connect AI.")
        now = utcnow()
        out_id = self.db.execute(
            "INSERT INTO notebook_outputs(notebook_id, kind, title, status, source_ids_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (nb_id, kind, KINDS[kind][0], "running", json.dumps(scope), now, now),
        ).lastrowid
        task = asyncio.create_task(self._run(nb_id, out_id, kind, scope))
        self._tasks[out_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(out_id, None))
        self.c.notebooks.touch(nb_id)
        return self.get(nb_id, out_id)

    async def wait(self, out_id: int) -> None:
        t = self._tasks.get(out_id)
        if t:
            await asyncio.gather(t, return_exceptions=True)

    # ---------------------------------------------------------------- generation
    def _passages(self, scope: list[str], kind: str) -> dict[str, Any]:
        """Passages spread evenly across each source (dated passages first for timelines)."""
        per_doc = max(3, PASSAGE_BUDGET // len(scope))
        ids: list[str] = []
        for d in scope:
            rows = self.db.all("SELECT id, text FROM chunks WHERE document_id=? AND length(text) > 80 ORDER BY chunk_index", (d,))
            if kind == "timeline":
                rows = sorted(rows, key=lambda r: not _YEAR.search(r["text"]))
                picked = rows[:per_doc]
            else:
                step = max(1, len(rows) / per_doc)
                picked = [rows[int(i * step)] for i in range(min(per_doc, len(rows)))]
            ids += [r["id"] for r in picked]
        ids = ids[: max(PASSAGE_BUDGET, 3 * len(scope))]
        loaded = self.c.retriever._load(ids)
        evidence = {}
        for i, cid in enumerate([x for x in ids if x in loaded], 1):
            ev = chunk_to_evidence(loaded[cid], "studio", kind)
            ev.id = f"E{i}"
            evidence[ev.id] = ev
        return evidence

    async def _run(self, nb_id: str, out_id: int, kind: str, scope: list[str]) -> None:
        current_job_id.set(f"notebook:{nb_id}")
        try:
            evidence = self._passages(scope, kind)
            if not evidence:
                raise ValueError("The selected sources have no readable text")
            blocks = "\n\n".join(e.prompt_block(PASSAGE_CHARS) for e in evidence.values())
            rules = (
                "Use ONLY the passages provided; never add facts from your own knowledge. Copy numbers exactly with units and "
                "conditions. Cite the passage ids that support each statement."
            )
            data: dict[str, Any] = {}
            if kind == "mindmap":
                mm = await chat_json(
                    self.c.router,
                    Task.CHAT,
                    MindMapOut,
                    system=f"You build a mind map of a notebook of research sources: the central topic, its main themes and their sub-points. {rules}",
                    user=f"PASSAGES:\n{blocks}",
                    max_tokens=5000,
                    temperature=0.2,
                )
                branches = _walk(mm.branches)
                data = {"central": mm.central.strip()[:120], "branches": branches}
                nodes = _labels(branches)
                content = "\n".join(n["label"] for n in nodes)
                mentioned = list(dict.fromkeys(c for n in nodes for c in n["citations"]))
                uncited = sum(1 for n in nodes if not n["citations"] and not n["children"])
            else:
                doc = await chat_json(
                    self.c.router,
                    Task.CHAT,
                    StudioDoc,
                    system=f"You write study and briefing material from the user's sources. Write: {KINDS[kind][1]} {rules}",
                    user=f"PASSAGES:\n{blocks}",
                    max_tokens=6000,
                    temperature=0.2,
                )
                content = doc.content.strip()
                mentioned = coerce_citations(content)
                uncited = uncited_statements(content)
            check = grounding_check(content, mentioned, evidence)
            check["uncited_statements"] = uncited
            check["passages_used"] = len(evidence)
            cites = {c: evidence[c].model_dump() | {"link": evidence[c].link()} for c in mentioned if c in evidence}
            self.db.execute(
                "UPDATE notebook_outputs SET status='completed', content=?, data_json=?, citations_json=?, check_json=?, updated_at=? WHERE id=?",
                (content, json.dumps(data), json.dumps(cites, default=str), json.dumps(check), utcnow(), out_id),
            )
        except asyncio.CancelledError:
            self.db.execute("UPDATE notebook_outputs SET status='failed', error='Cancelled', updated_at=? WHERE id=?", (utcnow(), out_id))
            raise
        except Exception as exc:
            log.exception("studio output failed")
            self.db.execute("UPDATE notebook_outputs SET status='failed', error=?, updated_at=? WHERE id=?", (f"{exc}"[:1000], utcnow(), out_id))
