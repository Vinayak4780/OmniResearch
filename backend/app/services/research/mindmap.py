"""Mind-map graph built deterministically from a finished investigation (no model calls).

Tree edges (``child``) form the map; cross-links connect claims to the evidence they cite
(``cites``) and conflicting evidence (``contradicts``).
"""

from __future__ import annotations

from typing import Any

from app.services.research.report import ReportDraft, iter_statements


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _round(query_id: str) -> int:
    """Q3 -> round 0 (initial decomposition); Q2.1 -> follow-up round 2."""
    head = query_id[1:].split(".")[0]
    return int(head) if "." in query_id and head.isdigit() else 0


def build_mindmap(question: str, report: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(nid: str, label: str, ntype: str, parent: str | None = None, *, full: str | None = None, **data: Any) -> str:
        nodes.append({"id": nid, "label": _short(label, 120) if full is None else label, "type": ntype, "data": {"full_text": full or label, **data}})
        if parent:
            edges.append({"source": parent, "target": nid, "type": "child"})
        return nid

    root = node("root", _short(question, 120), "question", full=question)
    draft = ReportDraft.model_validate(report.get("draft") or {"title": "", "executive_summary": [], "scope": "", "findings": []})
    statuses = ((report.get("verification") or {}).get("statements")) or {}
    ev_by_id = {e["id"]: e for e in evidence if e.get("id")}

    # Sub-questions the agent searched
    queries = report.get("queries") or []
    if queries:
        qroot = node("queries", f"Searches ({len(queries)})", "group", root)
        for q in queries:
            node(f"q:{q['id']}", _short(q["query"], 70), "query", qroot, full=q["query"], focus=q.get("focus"), source=q.get("source"), purpose=q.get("purpose"), round=_round(q["id"]))

    # Findings → claims → cited evidence
    ids = {id(s): sid for sid, s in iter_statements(draft)}
    froot = node("findings", "Findings", "group", root)
    sections = [("Summary", draft.executive_summary)] + [(f.heading, f.statements) for f in draft.findings] + [("Conclusions", draft.conclusions)]
    cited: set[str] = set()
    for i, (heading, stmts) in enumerate(sections):
        if not stmts:
            continue
        hid = node(f"h:{i}", _short(heading, 70), "finding", froot, full=heading)
        for s in stmts:
            sid = ids[id(s)]
            st = statuses.get(sid, {}).get("status")
            nid = node(f"s:{sid}", _short(s.text), "claim", hid, full=s.text, statement_id=sid, text=s.text, kind=s.kind, status=st, citations=s.citations)
            for c in s.citations:
                if c in ev_by_id:
                    edges.append({"source": nid, "target": f"e:{c}", "type": "cites"})
                    cited.add(c)

    # Sources grouped by document / web domain
    sroot = node("sources", "Sources", "group", root)
    groups: dict[str, str] = {}
    for e in sorted(ev_by_id.values(), key=lambda x: int(x["id"][1:])):
        if e["source_type"] == "local":
            gkey, glabel = f"d:{e['document_id']}", e.get("doc_title") or e.get("filename") or "document"
        else:
            gkey, glabel = f"w:{e.get('domain')}", e.get("domain") or "web"
        if gkey not in groups:
            groups[gkey] = node(
                gkey, _short(glabel, 70), "document" if e["source_type"] == "local" else "website", sroot,
                full=glabel, document_id=e.get("document_id"), domain=e.get("domain"),
            )
        label = f"{e['id']} · p.{e['page_no']}" if e["source_type"] == "local" else f"{e['id']} · {_short(e.get('web_title') or '', 40)}"
        node(
            f"e:{e['id']}",
            label,
            "evidence",
            groups[gkey],
            evidence_id=e["id"],
            source_type=e["source_type"],
            full=e["text"],
            page_no=e.get("page_no"),
            page_end=e.get("page_end"),
            document_id=e.get("document_id"),
            url=e.get("url"),
            element_type=e.get("element_type"),
            extraction_method=e.get("extraction_method"),
            excerpt=_short(e.get("text") or "", 400),
            cited=e["id"] in cited,
            inspected_visually=e.get("inspected_visually", False),
        )

    # Contradictions with cross-links to the conflicting evidence
    if draft.contradictions:
        croot = node("contradictions", f"Contradictions ({len(draft.contradictions)})", "group", root)
        for i, c in enumerate(draft.contradictions):
            cid = node(f"c:{i}", _short(c.topic, 70), "contradiction", croot, full=c.topic, description=c.description, explanation=c.possible_explanation, citations=c.citations)
            for ev in c.citations:
                if ev in ev_by_id:
                    edges.append({"source": cid, "target": f"e:{ev}", "type": "contradicts"})

    if draft.evidence_gaps:
        groot = node("gaps", f"Open questions ({len(draft.evidence_gaps)})", "group", root)
        for i, g in enumerate(draft.evidence_gaps):
            node(f"g:{i}", _short(g, 90), "gap", groot, full=g, text=g)

    return {"nodes": nodes, "edges": edges, "title": draft.title}
