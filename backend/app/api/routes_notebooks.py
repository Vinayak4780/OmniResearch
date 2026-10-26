"""Notebooks: sources, grounded chat, guide, notes and deep research."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import container
from app.core.container import Container

router = APIRouter(prefix="/api/notebooks", tags=["notebooks"])


class NotebookIn(BaseModel):
    title: str = Field(default="Untitled notebook", max_length=120)


class AskIn(BaseModel):
    message: str = Field(min_length=2, max_length=4000)
    source_ids: list[str] | None = Field(default=None, description="Selected sources; omitted = all processed sources")
    web: bool = Field(default=False, description="Fast answer: also search the web once and use the best passages")


class NoteIn(BaseModel):
    content: str = Field(default="", max_length=20000)
    title: str | None = Field(default=None, max_length=120)
    message_id: int | None = Field(default=None, description="Save a chat answer (with its citations) as a note")


class LinksIn(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=10)


class DiscoverIn(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    mode: Literal["fast", "deep"] = "fast"


class OutputIn(BaseModel):
    kind: Literal["summary", "briefing", "study_guide", "faq", "timeline", "mindmap"]
    source_ids: list[str] | None = None


class ResearchIn(BaseModel):
    question: str = Field(min_length=5, max_length=4000)
    source_ids: list[str] | None = None
    web_mode: Literal["auto", "always", "never"] | None = None


@router.get("")
def list_notebooks(c: Container = Depends(container)):
    return c.notebooks.list()


@router.post("", status_code=201)
def create_notebook(body: NotebookIn, c: Container = Depends(container)):
    return c.notebooks.create(body.title)


@router.get("/{nb_id}")
def get_notebook(nb_id: str, c: Container = Depends(container)):
    return c.notebooks.get(nb_id)


@router.patch("/{nb_id}")
def rename_notebook(nb_id: str, body: NotebookIn, c: Container = Depends(container)):
    return c.notebooks.rename(nb_id, body.title)


@router.delete("/{nb_id}", status_code=204)
def delete_notebook(nb_id: str, c: Container = Depends(container)):
    c.notebooks.delete(nb_id)


@router.get("/{nb_id}/sources")
def notebook_sources(nb_id: str, c: Container = Depends(container)):
    c.notebooks.get(nb_id)
    return c.notebooks.sources(nb_id)


@router.post("/{nb_id}/sources/links")
async def notebook_add_links(nb_id: str, body: LinksIn, c: Container = Depends(container)):
    return {"results": await c.notebook_web.add_links(nb_id, body.urls)}


@router.post("/{nb_id}/discover")
async def notebook_discover(nb_id: str, body: DiscoverIn, c: Container = Depends(container)):
    return await c.notebook_web.discover(nb_id, body.query, body.mode)


@router.get("/{nb_id}/guide")
async def notebook_guide(nb_id: str, c: Container = Depends(container)):
    return await c.notebooks.guide(nb_id)


@router.get("/{nb_id}/messages")
def notebook_messages(nb_id: str, c: Container = Depends(container)):
    return c.notebooks.messages(nb_id)


@router.post("/{nb_id}/messages")
async def notebook_ask(nb_id: str, body: AskIn, c: Container = Depends(container)):
    return await c.notebooks.ask(nb_id, body.message, body.source_ids, body.web)


@router.delete("/{nb_id}/messages", status_code=204)
def notebook_clear_chat(nb_id: str, c: Container = Depends(container)):
    c.notebooks.clear_chat(nb_id)


@router.get("/{nb_id}/notes")
def notebook_notes(nb_id: str, c: Container = Depends(container)):
    return c.notebooks.notes(nb_id)


@router.post("/{nb_id}/notes", status_code=201)
def notebook_add_note(nb_id: str, body: NoteIn, c: Container = Depends(container)):
    return c.notebooks.add_note(nb_id, body.content, body.title, body.message_id)


@router.delete("/{nb_id}/notes/{note_id}", status_code=204)
def notebook_delete_note(nb_id: str, note_id: int, c: Container = Depends(container)):
    c.notebooks.delete_note(nb_id, note_id)


@router.get("/{nb_id}/reports")
def notebook_reports(nb_id: str, c: Container = Depends(container)):
    return c.notebooks.reports(nb_id)


@router.post("/{nb_id}/reports", status_code=201)
async def notebook_start_research(nb_id: str, body: ResearchIn, c: Container = Depends(container)):
    return c.notebooks.start_research(nb_id, body.question, body.source_ids, body.web_mode)


@router.get("/{nb_id}/outputs")
def notebook_outputs(nb_id: str, c: Container = Depends(container)):
    return c.studio.list(nb_id)


@router.post("/{nb_id}/outputs", status_code=201)
async def notebook_create_output(nb_id: str, body: OutputIn, c: Container = Depends(container)):
    return c.studio.create(nb_id, body.kind, body.source_ids)


@router.get("/{nb_id}/outputs/{out_id}")
def notebook_output(nb_id: str, out_id: int, c: Container = Depends(container)):
    return c.studio.get(nb_id, out_id)


@router.delete("/{nb_id}/outputs/{out_id}", status_code=204)
def notebook_delete_output(nb_id: str, out_id: int, c: Container = Depends(container)):
    c.studio.delete(nb_id, out_id)
