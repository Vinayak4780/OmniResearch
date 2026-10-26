"""Research job orchestration: start/cancel/resume investigations, persist events, evidence and reports."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from app.agents.graph import build_graph
from app.agents.nodes import JobCancelled
from app.agents.tools import ResearchDeps
from app.observability.logging import get_logger, log_event, trace_id_var
from app.observability.metrics import current_job_id
from app.providers.base import TASK_CAPABILITY, Capability, Task
from app.providers.router import ProviderRouter
from app.repositories.db import Database, utcnow
from app.services.citations.validator import CitationValidator
from app.services.retrieval.hybrid import Retriever
from app.services.web.research import WebResearchService

log = get_logger("omni.research")


class ResearchService:
    def __init__(
        self,
        settings,
        db: Database,
        router: ProviderRouter,
        retriever: Retriever,
        web: WebResearchService,
        render_page,
        pdf_path,
        checkpoint_path: Path | None = None,
    ):
        self.settings = settings
        self.db = db
        self.router = router
        self.retriever = retriever
        self.web = web
        self._render_page = render_page
        self._pdf_path = pdf_path
        self.checkpoint_path = checkpoint_path
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        self._saver_cm = None
        self._saver = None

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self.checkpoint_path is not None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self._saver_cm = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path))
            self._saver = await self._saver_cm.__aenter__()
        for r in self.db.all("SELECT id FROM research_jobs WHERE status IN ('running','queued')"):
            if self._saver is not None:
                self._event(r["id"], "system", "resuming after restart", {})
                self._launch(r["id"], resume=True)
            else:
                self._update(r["id"], status="interrupted", error="Server restarted during the investigation")

    async def stop(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(None, None, None)

    def deps(self) -> ResearchDeps:
        return ResearchDeps(
            router=self.router,
            retriever=self.retriever,
            web=self.web,
            validator=CitationValidator(self.db, self.router),
            render_page=self._render_page,
            pdf_path=self._pdf_path,
            emit=self._event,
            is_cancelled=lambda j: j in self._cancelled,
            settings=self.settings,
            usage=self.usage,
        )

    # ---------------------------------------------------------------- persistence helpers
    def _update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = utcnow()
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE research_jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def _event(self, job_id: str, stage: str, message: str, data: dict) -> None:
        self.db.execute(
            "INSERT INTO research_events(job_id, ts, stage, message, data_json) VALUES (?,?,?,?,?)",
            (job_id, utcnow(), stage, message[:1000], json.dumps(data, default=str)[:20000]),
        )
        if stage != "system":
            self.db.execute("UPDATE research_jobs SET stage=?, updated_at=? WHERE id=?", (stage, utcnow(), job_id))

    def usage(self, job_id: str) -> dict[str, Any]:
        """Token/cost accounting for a job, from the persisted call log (survives restarts and resumes)."""
        calls = self.db.all("SELECT * FROM provider_calls WHERE job_id=? ORDER BY id", (job_id,))
        ok = [c for c in calls if c["ok"]]
        costs = [c["cost_usd"] for c in ok if c["cost_usd"] is not None]
        providers: dict[str, list[str]] = {}
        for c in ok:
            label = c["provider_id"] + (f" ({c['model']})" if c["model"] else "")
            bucket = providers.setdefault(c["task"] or c["capability"], [])
            if label not in bucket:
                bucket.append(label)
        return {
            "usage": {
                "calls": len(calls),
                "failed_calls": len(calls) - len(ok),
                "fallback_calls": sum(1 for c in ok if c["fallback_from"]),
                "input_tokens": sum(c["input_tokens"] or 0 for c in ok) or None,
                "output_tokens": sum(c["output_tokens"] or 0 for c in ok) or None,
                "cost_usd": sum(costs) if costs else None,
            },
            "providers_by_task": providers,
        }

    # ---------------------------------------------------------------- API
    def preflight(self) -> list[str]:
        """Problems that prevent an investigation from running."""
        problems = []
        if not self.router.is_configured(Task.SYNTHESIS):
            problems.append("No chat provider is assigned to 'synthesis' (the default LLM). Configure one in Settings → AI Providers.")
        return problems

    def external_destinations(self, options: dict[str, Any]) -> list[dict[str, Any]]:
        """Which configured providers would receive document content (for the privacy warning)."""
        out = []
        for task in Task:
            p = self.router.primary(task)
            # web search/extraction receive queries and URLs, not document content
            if p is not None and p.requires_network and TASK_CAPABILITY[task] not in (Capability.WEB_SEARCH, Capability.WEB_EXTRACTION):
                out.append({"task": task.value, "provider_id": p.provider_id, "adapter": p.adapter})
        return out

    def create(self, question: str, options: dict[str, Any] | None = None) -> str:
        question = question.strip()
        if not 5 <= len(question) <= 4000:
            raise ValueError("Question must be 5–4000 characters")
        problems = self.preflight()
        if problems:
            raise ValueError(" ".join(problems))
        opts = {k: v for k, v in (options or {}).items() if k in ("collections", "document_ids", "web_mode", "notebook_id")}
        if opts.get("web_mode") not in (None, "auto", "always", "never"):
            raise ValueError("web_mode must be auto, always or never")
        job_id = uuid.uuid4().hex[:16]
        now = utcnow()
        self.db.execute(
            "INSERT INTO research_jobs(id, question, status, options_json, stage, created_at, updated_at, notebook_id) VALUES (?,?,?,?,?,?,?,?)",
            (job_id, question, "queued", json.dumps(opts), "queued", now, now, opts.get("notebook_id")),
        )
        self._launch(job_id)
        return job_id

    def _launch(self, job_id: str, resume: bool = False) -> None:
        task = asyncio.create_task(self._run(job_id, resume))
        self._tasks[job_id] = task

        def done(t: asyncio.Task) -> None:
            # A task cancelled before its first step never reaches _run's handlers.
            if t.cancelled():
                self._tasks.pop(job_id, None)
                if job_id in self._cancelled:
                    self._cancelled.discard(job_id)
                    self._update(job_id, status="cancelled", stage="cancelled")
                    self._event(job_id, "system", "cancelled by user", {})

        task.add_done_callback(done)

    async def _run(self, job_id: str, resume: bool) -> None:
        job = self.db.one("SELECT * FROM research_jobs WHERE id=?", (job_id,))
        if not job:
            return
        trace_id_var.set(job_id)
        current_job_id.set(job_id)
        self._update(job_id, status="running", error=None)
        graph = build_graph(self.deps(), checkpointer=self._saver)
        config = {"configurable": {"thread_id": job_id}, "recursion_limit": 80}
        init = None if resume else {"job_id": job_id, "question": job["question"], "options": json.loads(job["options_json"]), "warnings": []}
        try:
            final = await graph.ainvoke(init, config=config)
            self._save_evidence(job_id, final.get("evidence") or {})
            report = final.get("report") or {}
            report.update(self.usage(job_id))
            self._update(job_id, status="completed", stage="done", report_md=final.get("report_md"), report_json=json.dumps(report, default=str))
            log_event(log, "research completed", job_id=job_id)
        except (JobCancelled, asyncio.CancelledError):
            if job_id in self._cancelled:
                self._update(job_id, status="cancelled", stage="cancelled")
                self._event(job_id, "system", "cancelled by user", {})
            else:
                raise
        except Exception as exc:
            log.exception("research failed")
            self._update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:2000])
            self._event(job_id, "system", f"failed: {type(exc).__name__}: {str(exc)[:300]}", {})
        finally:
            self._tasks.pop(job_id, None)
            self._cancelled.discard(job_id)

    def _save_evidence(self, job_id: str, evidence: dict[str, Any]) -> None:
        self.db.execute("DELETE FROM research_evidence WHERE job_id=?", (job_id,))
        self.db.executemany(
            "INSERT INTO research_evidence(job_id, evidence_id, data_json) VALUES (?,?,?)",
            [(job_id, k, json.dumps(v, default=str)) for k, v in evidence.items()],
        )

    def cancel(self, job_id: str) -> bool:
        job = self.db.one("SELECT status FROM research_jobs WHERE id=?", (job_id,))
        if not job:
            raise KeyError(job_id)
        if job["status"] not in ("queued", "running"):
            return False
        self._cancelled.add(job_id)
        task = self._tasks.get(job_id)
        if task:
            task.cancel()
        else:
            self._update(job_id, status="cancelled")
        return True

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.db.one("SELECT * FROM research_jobs WHERE id=?", (job_id,))
        if not job:
            raise KeyError(job_id)
        job["options"] = json.loads(job.pop("options_json"))
        raw = job.pop("report_json", None)
        job["report"] = json.loads(raw) if raw else None
        return job

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.db.all("SELECT id, question, status, stage, error, created_at, updated_at FROM research_jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM research_events WHERE job_id=? AND id>? ORDER BY id", (job_id, after))
        for r in rows:
            r["data"] = json.loads(r.pop("data_json"))
        return rows

    def evidence(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT data_json FROM research_evidence WHERE job_id=?", (job_id,))
        items = [json.loads(r["data_json"]) for r in rows]
        return sorted(items, key=lambda e: int((e.get("id") or "E0")[1:]))

    def delete(self, job_id: str) -> None:
        if job_id in self._tasks:
            raise ValueError("Cancel the job before deleting it")
        self.db.execute("DELETE FROM research_jobs WHERE id=?", (job_id,))

    async def wait(self, job_id: str, timeout: float = 120) -> dict[str, Any]:
        task = self._tasks.get(job_id)
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        return self.get(job_id)
