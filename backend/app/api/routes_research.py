"""Research workspace and web settings endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.api.deps import container, require_admin
from app.core.container import Container
from app.services.web.research import WebSettings

router = APIRouter(prefix="/api", tags=["research"])


class ResearchIn(BaseModel):
    question: str = Field(min_length=5, max_length=4000)
    collections: list[str] | None = None
    document_ids: list[str] | None = None
    web_mode: Literal["auto", "always", "never"] | None = None


@router.post("/research", status_code=201)
async def start_research(body: ResearchIn, c: Container = Depends(container)):
    job_id = c.research.create(body.question, body.model_dump(exclude={"question"}, exclude_none=True))
    return c.research.get(job_id)


@router.get("/research")
def list_research(c: Container = Depends(container)):
    return c.research.list()


@router.get("/research/preflight")
def preflight(c: Container = Depends(container)):
    web_ok, web_reason = c.web.available()
    return {"problems": c.research.preflight(), "external_destinations": c.research.external_destinations({}), "web_available": web_ok, "web_reason": web_reason}


@router.get("/research/{job_id}")
def get_research(job_id: str, c: Container = Depends(container)):
    return c.research.get(job_id)


@router.get("/research/{job_id}/events")
def research_events(job_id: str, after: int = 0, c: Container = Depends(container)):
    c.research.get(job_id)
    return c.research.events(job_id, after)


@router.get("/research/{job_id}/evidence")
def research_evidence(job_id: str, c: Container = Depends(container)):
    return c.research.evidence(job_id)


@router.post("/research/{job_id}/cancel")
async def cancel_research(job_id: str, c: Container = Depends(container)):
    return {"cancelled": c.research.cancel(job_id)}


@router.delete("/research/{job_id}", status_code=204)
def delete_research(job_id: str, c: Container = Depends(container)):
    c.research.delete(job_id)


@router.get("/research/{job_id}/report.md", response_class=PlainTextResponse)
def report_markdown(job_id: str, c: Container = Depends(container)):
    job = c.research.get(job_id)
    if not job.get("report_md"):
        raise KeyError(f"report for {job_id}")
    return PlainTextResponse(job["report_md"], media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="omniresearch-{job_id}.md"'})


class ChatIn(BaseModel):
    message: str = Field(min_length=2, max_length=2000)


@router.get("/research/{job_id}/chat")
def research_chat_history(job_id: str, c: Container = Depends(container)):
    c.research.get(job_id)
    return c.followup.history(job_id)


@router.post("/research/{job_id}/chat")
async def research_chat(job_id: str, body: ChatIn, c: Container = Depends(container)):
    return await c.followup.ask(c.research.get(job_id), body.message)


@router.get("/research/{job_id}/mindmap")
def research_mindmap(job_id: str, c: Container = Depends(container)):
    from app.services.research.mindmap import build_mindmap

    job = c.research.get(job_id)
    if job["status"] != "completed" or not job.get("report"):
        raise ValueError("The mind map is available once the investigation has completed")
    return build_mindmap(job["question"], job["report"], c.research.evidence(job_id))


# ---------------------------------------------------------------- web settings
@router.get("/settings/web")
def get_web(c: Container = Depends(container)):
    ok, reason = c.web.available()
    return {"settings": c.web.settings().model_dump(), "available": ok, "reason": reason}


@router.put("/settings/web", dependencies=[Depends(require_admin)])
def put_web(body: WebSettings, c: Container = Depends(container)):
    return c.web.save_settings(body).model_dump()


@router.get("/web/errors")
def web_errors(c: Container = Depends(container)):
    return c.web.recent_errors()
