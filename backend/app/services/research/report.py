"""Report schema (what the synthesis model must produce) and Markdown rendering."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.services.research.evidence import Evidence


def coerce_citations(v: Any) -> list[str]:
    """Models sometimes emit null, '[E3]' or 'E3, E4'; keep only well-formed evidence ids."""
    if v is None:
        return []
    out: list[str] = []
    for it in v if isinstance(v, list) else [v]:
        if isinstance(it, str):
            out += re.findall(r"\bE\d+\b", it)
    return list(dict.fromkeys(out))


class _Cited(BaseModel):
    citations: list[str] = Field(default_factory=list, description="Evidence ids like E3 that directly support the claim")

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> list[str]:
        return coerce_citations(v)


class Statement(_Cited):
    text: str = Field(description="One claim. Put evidence ids in 'citations', not in the text.")
    kind: Literal["source", "interpretation"] = Field(default="source", description="'source' = stated by the cited sources; 'interpretation' = your analysis")


class Finding(BaseModel):
    heading: str
    statements: list[Statement]


class TableRow(_Cited):
    cells: list[str]

    @field_validator("cells", mode="before")
    @classmethod
    def _cells(cls, v: Any) -> list[str]:
        return ["" if c is None else str(c) for c in (v or [])]


class ComparisonTable(BaseModel):
    title: str
    columns: list[str]
    rows: list[TableRow]
    conditions_note: str = Field(default="", description="Hardware, model, dataset and settings behind the numbers")


class Contradiction(_Cited):
    topic: str
    description: str
    possible_explanation: str = ""


class ReportDraft(BaseModel):
    title: str
    executive_summary: list[Statement]
    scope: str
    findings: list[Finding]
    comparison_tables: list[ComparisonTable] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    conclusions: list[Statement] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


def iter_statements(draft: ReportDraft):
    """Yield (statement_id, Statement) in a stable order."""
    for i, s in enumerate(draft.executive_summary):
        yield f"S{i + 1}", s
    for fi, f in enumerate(draft.findings):
        for si, s in enumerate(f.statements):
            yield f"F{fi + 1}.{si + 1}", s
    for i, s in enumerate(draft.conclusions):
        yield f"C{i + 1}", s


_STATUS_MARK = {"verified": "", "partially_verified": " ◐", "unverified": " ⚠ *unverified*", "interpretation": "", "uncited": " ⚠ *no citation*"}


def _cites(ids: list[str], evidence: dict[str, Evidence]) -> str:
    parts = []
    for c in ids:
        e = evidence.get(c)
        parts.append(f"[[{c}]]({e.link()})" if e else f"[{c}?]")
    return " " + " ".join(parts) if parts else ""


def _clean(text: str) -> str:
    return re.sub(r"\s*\[(E\d+)\]", "", text).strip()


def render_markdown(
    question: str,
    draft: ReportDraft,
    evidence: dict[str, Evidence],
    verification: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    L: list[str] = [f"# {draft.title}", ""]
    L += [f"> **Research question:** {question}", ""]
    vs = meta.get("verification_summary", {})
    if vs:
        L += [
            f"*Claims checked: {vs.get('total', 0)} — verified {vs.get('verified', 0)}, partially verified {vs.get('partially_verified', 0)}, "
            f"unverified {vs.get('unverified', 0)}, interpretation {vs.get('interpretation', 0)}. "
            "Verification combines deterministic checks (source exists, excerpt matches, numbers present) with a model-based support judgment, which is not proof.*",
            "",
        ]

    def stmt_line(sid: str, s: Statement) -> str:
        v = verification.get(sid, {})
        status = v.get("status", "")
        prefix = "*Interpretation:* " if s.kind == "interpretation" else ""
        return f"- {prefix}{_clean(s.text)}{_cites(s.citations, evidence)}{_STATUS_MARK.get(status, '')}"

    ids = dict((id(s), sid) for sid, s in iter_statements(draft))
    L += ["## Executive summary", ""] + [stmt_line(ids[id(s)], s) for s in draft.executive_summary] + [""]
    L += ["## Research question and scope", "", draft.scope, ""]
    L += ["## Findings", ""]
    for f in draft.findings:
        L += [f"### {f.heading}", ""] + [stmt_line(ids[id(s)], s) for s in f.statements] + [""]
    if draft.comparison_tables:
        L += ["## Comparison", ""]
        for t in draft.comparison_tables:
            L += [f"**{t.title}**", ""]
            cols = t.columns + ["Sources"]
            L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for i, r in enumerate(t.rows):
                cells = (r.cells + [""] * len(t.columns))[: len(t.columns)]
                flag = verification.get(f"T:{t.title}:{i}", {}).get("status")
                mark = " ⚠" if flag == "unverified" else ""
                L.append("| " + " | ".join(c.replace("|", "/") for c in cells) + f" |{_cites(r.citations, evidence)}{mark} |")
            if t.conditions_note:
                L += ["", f"*Conditions:* {t.conditions_note}"]
            L.append("")
    L += ["## Contradictory findings", ""]
    if draft.contradictions:
        for c in draft.contradictions:
            L.append(f"- **{c.topic}:** {_clean(c.description)}{_cites(c.citations, evidence)}" + (f" — *possible explanation:* {c.possible_explanation}" if c.possible_explanation else ""))
    else:
        L.append("- No contradictions were identified among the retrieved evidence.")
    L.append("")
    L += ["## Limitations", ""] + [f"- {x}" for x in draft.limitations or ["None stated."]] + [""]
    L += ["## Evidence gaps and unresolved questions", ""] + [f"- {x}" for x in draft.evidence_gaps or ["None identified."]] + [""]
    L += ["## Conclusions", ""] + [stmt_line(ids[id(s)], s) for s in draft.conclusions] + [""]
    L += ["## Suggested follow-up questions", ""] + [f"- {q}" for q in draft.follow_up_questions] + [""]

    cited = {c for _, s in iter_statements(draft) for c in s.citations} | {c for t in draft.comparison_tables for r in t.rows for c in r.citations} | {
        c for x in draft.contradictions for c in x.citations
    }
    local = [e for e in evidence.values() if e.id in cited and e.source_type == "local"]
    web = [e for e in evidence.values() if e.id in cited and e.source_type == "web"]
    L += ["## Sources", "", "### Uploaded documents", ""]
    for e in sorted(local, key=lambda x: int((x.id or "E0")[1:])):
        vis = f" Visual inspection by `{e.visual_model}` (model interpretation)." if e.inspected_visually else ""
        L.append(
            f"- **[{e.id}]** [{e.citation_label()}]({e.link()}) — section: {e.section_path or '—'}; element: {e.element_type}; "
            f"extraction: {e.extraction_method}.{vis}\n  > {e.text[:300].replace(chr(10), ' ')}{'…' if len(e.text) > 300 else ''}"
        )
    if not local:
        L.append("- None cited.")
    L += ["", "### Web sources", ""]
    for e in sorted(web, key=lambda x: int((x.id or "E0")[1:])):
        L.append(
            f"- **[{e.id}]** [{e.web_title or e.domain}]({e.link()}) — {e.domain}; published: {e.published_at or 'unknown'}; retrieved: {e.retrieved_at}"
            + ("; **search snippet only (page not fetched)**" if e.snippet_only else "")
            + f"\n  > {e.text[:300].replace(chr(10), ' ')}{'…' if len(e.text) > 300 else ''}"
        )
    if not web:
        L.append("- None cited.")
    L.append("")
    usage = meta.get("usage")
    if usage:
        L += ["## Run details", ""]
        L.append(f"- Provider calls: {usage.get('calls', 0)}; input tokens: {usage.get('input_tokens') or 'n/a'}; output tokens: {usage.get('output_tokens') or 'n/a'}; "
                 f"estimated cost: {('$%.4f' % usage['cost_usd']) if usage.get('cost_usd') is not None else 'n/a (no pricing configured)'}")
        for t, pid in (meta.get("providers_by_task") or {}).items():
            L.append(f"- {t}: {', '.join(pid)}")
        for w in meta.get("warnings", [])[:20]:
            L.append(f"- ⚠ {w}")
        L.append("")
    return "\n".join(L)
