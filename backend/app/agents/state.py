"""Typed LangGraph state for a research investigation. Values are JSON-serializable so the
SQLite checkpointer can persist them and interrupted jobs can resume."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


def merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            # keep the higher-relevance copy of the same evidence, but never lose an assigned id
            keep = v if v.get("relevance", 0) >= out[k].get("relevance", 0) else out[k]
            keep = {**keep, "id": out[k].get("id") or v.get("id")}
            out[k] = keep
        else:
            out[k] = v
    return out


def replace_dict(a: dict | None, b: dict | None) -> dict:
    return b if b is not None else (a or {})


class ResearchState(TypedDict, total=False):
    job_id: str
    question: str
    options: dict[str, Any]
    analysis: dict[str, Any]
    plan: dict[str, Any]
    subqueries: list[dict[str, Any]]
    query_history: Annotated[list[dict[str, Any]], operator.add]
    candidates: Annotated[dict[str, dict[str, Any]], merge_dict]  # raw retrieved evidence by key
    evidence: Annotated[dict[str, dict[str, Any]], replace_dict]  # selected evidence by E-id
    eid_counter: int
    web_pages: Annotated[dict[str, dict[str, Any]], merge_dict]
    use_web: bool
    web_reason: str
    iteration: int
    last_round_new: int  # evidence items added by the most recent retrieval round
    sufficiency: dict[str, Any]
    comparison: dict[str, Any]
    draft: dict[str, Any]
    verification: dict[str, Any]
    report_md: str
    report: dict[str, Any]
    warnings: Annotated[list[str], operator.add]


# ---------------------------------------------------------------- structured LLM outputs


class QuestionAnalysis(BaseModel):
    intent: str = Field(description="What the user wants to learn, in one sentence")
    key_concepts: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list, description="Axes to compare, e.g. memory, latency, accuracy")
    needs_current_information: bool = Field(default=False, description="True if recent/current developments matter")
    needs_tables_or_figures: bool = False
    scope_notes: str = ""


class ResearchPlan(BaseModel):
    steps: list[str]
    evidence_required: list[str] = Field(default_factory=list)
    use_web: Literal["yes", "no", "if_local_insufficient"] = "if_local_insufficient"
    rationale: str = ""


class SubQuery(BaseModel):
    query: str = Field(description="A focused search query")
    focus: Literal["text", "table", "figure"] = "text"
    source: Literal["local", "web", "both"] = "local"
    purpose: str = ""


class Decomposition(BaseModel):
    subqueries: list[SubQuery]


class Sufficiency(BaseModel):
    sufficient: bool
    gaps: list[str] = Field(default_factory=list)
    followups: list[SubQuery] = Field(default_factory=list)
    needs_web: bool = False


class RewrittenQueries(BaseModel):
    queries: list[str]


class Observation(BaseModel):
    evidence_id: str
    claim: str


class ComparisonDimension(BaseModel):
    dimension: str
    observations: list[Observation] = Field(default_factory=list)


class ContradictionItem(BaseModel):
    topic: str
    description: str
    evidence_ids: list[str]
    possible_explanation: str = ""


class Comparison(BaseModel):
    comparisons: list[ComparisonDimension] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)
    condition_differences: list[str] = Field(default_factory=list, description="Differences in hardware/model/dataset/settings that limit comparability")


class RewrittenClaim(BaseModel):
    supported_rewrite: str | None = Field(description="Claim rewritten to state only what the cited evidence says, or null if impossible")
