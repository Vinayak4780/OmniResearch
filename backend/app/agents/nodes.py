"""LangGraph node implementations. Nodes depend only on injected services and capability routes —
no vendor SDKs, model names or endpoints appear here."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable

from app.agents.state import (
    Comparison,
    Decomposition,
    QuestionAnalysis,
    ResearchPlan,
    ResearchState,
    RewrittenClaim,
    RewrittenQueries,
    SubQuery,
    Sufficiency,
)
from app.agents.tools import ResearchDeps, ResearchTools, chunk_to_evidence
from app.core.text import tokenize
from app.observability.logging import get_logger
from app.providers.base import NoProviderAvailableError, ProviderError, RerankCandidate, Task
from app.services.research.evidence import Evidence
from app.services.research.llm import StructuredOutputError, chat_json
from app.services.research.report import ReportDraft, Statement, iter_statements, render_markdown
from app.providers.base import WebPage

log = get_logger("omni.agent")
LLM_ERRORS = (ProviderError, NoProviderAvailableError, StructuredOutputError)


class JobCancelled(Exception):
    pass


Node = Callable[[ResearchState], Awaitable[dict[str, Any]]]


def _q(question: str) -> str:
    return f"<question>\n{question}\n</question>"


def _ev(state: ResearchState) -> dict[str, Evidence]:
    return {k: Evidence.model_validate(v) for k, v in (state.get("evidence") or {}).items()}


class ResearchNodes:
    def __init__(self, deps: ResearchDeps):
        self.d = deps
        self.tools = ResearchTools(deps)
        s = deps.settings
        self.max_iterations = s.agent_max_iterations
        self.max_subqueries = s.agent_max_subqueries
        self.per_query = s.agent_evidence_per_query
        self.max_evidence = s.agent_max_evidence

    # ------------------------------------------------------------ instrumentation
    def node(self, stage: str) -> Callable[[Node], Node]:
        def deco(fn: Node) -> Node:
            @functools.wraps(fn)
            async def wrapper(state: ResearchState) -> dict[str, Any]:
                job = state["job_id"]
                if self.d.is_cancelled(job):
                    raise JobCancelled(job)
                self.d.emit(job, stage, "started", {})
                t0 = time.perf_counter()
                out = await fn(state)
                if self.d.is_cancelled(job):
                    raise JobCancelled(job)
                summary = out.pop("_event", {}) if isinstance(out, dict) else {}
                self.d.emit(job, stage, summary.pop("message", "completed"), {**summary, "seconds": round(time.perf_counter() - t0, 2)})
                return out

            return wrapper

        return deco

    def _opts(self, state: ResearchState) -> dict[str, Any]:
        return state.get("options") or {}

    # ------------------------------------------------------------ 1. question analysis
    async def analyze_question(self, state: ResearchState) -> dict[str, Any]:
        try:
            a = await chat_json(
                self.d.router,
                Task.PLANNING,
                QuestionAnalysis,
                system="You analyze research questions for a technical literature-research assistant.",
                user=_q(state["question"]),
                max_tokens=800,
            )
            return {"analysis": a.model_dump(), "_event": {"message": a.intent, "analysis": a.model_dump()}}
        except LLM_ERRORS as exc:
            a = QuestionAnalysis(intent=state["question"], key_concepts=tokenize(state["question"])[:8])
            return {"analysis": a.model_dump(), "warnings": [f"question analysis fell back to defaults: {exc}"], "_event": {"message": "fallback analysis"}}

    # ------------------------------------------------------------ 2. planning
    async def plan_research(self, state: ResearchState) -> dict[str, Any]:
        web_ok, why = self.d.web.available()
        try:
            p = await chat_json(
                self.d.router,
                Task.PLANNING,
                ResearchPlan,
                system=(
                    "Plan a multi-step literature investigation. The assistant can search the user's uploaded documents "
                    f"(text, tables, figure captions) {'and the live web' if web_ok else '(web search is NOT available)'}. "
                    "Decide what evidence is required to answer rigorously and whether web research is needed."
                ),
                user=f"{_q(state['question'])}\nAnalysis: {state.get('analysis')}",
                max_tokens=900,
            )
        except LLM_ERRORS as exc:
            p = ResearchPlan(steps=["Search uploaded documents", "Compare evidence", "Write report"])
            return {"plan": p.model_dump(), "warnings": [f"planning fell back to defaults: {exc}"], "_event": {"message": "fallback plan"}}
        return {"plan": p.model_dump(), "_event": {"message": f"{len(p.steps)} steps; web: {p.use_web}", "plan": p.model_dump()}}

    # ------------------------------------------------------------ 3. decomposition
    async def decompose(self, state: ResearchState) -> dict[str, Any]:
        warnings: list[str] = []
        try:
            d = await chat_json(
                self.d.router,
                Task.DECOMPOSITION,
                Decomposition,
                system=(
                    f"Break the research question into at most {self.max_subqueries} focused search queries covering every concept and "
                    "comparison dimension. Use focus='table' for numeric results, focus='figure' for charts/diagrams. "
                    "Use source='web' only for information that is likely current or absent from papers; 'both' when corroboration helps."
                ),
                user=f"{_q(state['question'])}\nAnalysis: {state.get('analysis')}\nPlan: {state.get('plan')}",
                max_tokens=900,
            )
            subs = d.subqueries[: self.max_subqueries]
        except LLM_ERRORS as exc:
            warnings.append(f"decomposition fell back to the original question: {exc}")
            subs = []
        queries = [SubQuery(query=state["question"], focus="text", source="both", purpose="original question")] + subs
        seen: set[str] = set()
        out = []
        for i, s in enumerate(queries):
            key = s.query.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append({"id": f"Q{i + 1}", **s.model_dump()})
        return {"subqueries": out, "query_history": out, "iteration": 0, "warnings": warnings, "_event": {"message": f"{len(out)} sub-queries", "subqueries": out}}

    # ------------------------------------------------------------ 4/5. web decision + fan-out
    async def route_sources(self, state: ResearchState) -> dict[str, Any]:
        opts = self._opts(state)
        web_ok, why = self.d.web.available()
        mode = opts.get("web_mode") or self.d.web.settings().mode
        plan = state.get("plan") or {}
        analysis = state.get("analysis") or {}
        has_local = self.d.retriever.db.one("SELECT COUNT(*) AS n FROM chunks")["n"] > 0
        if not web_ok:
            use, reason = False, why
        elif mode == "never":
            use, reason = False, "web research disabled for this investigation"
        elif mode == "always":
            use, reason = True, "web research requested"
        elif not has_local:
            use, reason = True, "no indexed local documents"
        elif plan.get("use_web") == "yes" or analysis.get("needs_current_information"):
            use, reason = True, "question needs current or external information"
        else:
            use, reason = False, "deferred until local evidence is assessed"
        return {"use_web": use, "web_reason": reason, "_event": {"message": f"web research: {'yes' if use else 'no'} ({reason})"}}

    def fan_out(self, state: ResearchState) -> list[str]:
        targets = ["retrieve_local"]
        if state.get("use_web"):
            targets.append("web_research")
        return targets

    # ------------------------------------------------------------ 6a. local retrieval
    async def retrieve_local(self, state: ResearchState) -> dict[str, Any]:
        opts = self._opts(state)
        all_subs = state.get("subqueries", [])
        # Web-only sub-queries are searched locally too when web research is not running this round.
        subs = all_subs if not state.get("use_web") else ([s for s in all_subs if s.get("source") in ("local", "both")] or all_subs)
        candidates: dict[str, dict] = {}
        warnings: list[str] = []
        for s in subs:
            chunks, w = await self.tools.search_documents(
                s["query"], k=self.per_query, document_ids=opts.get("document_ids") or None, collections=opts.get("collections") or None, focus=s.get("focus", "text")
            )
            warnings += w
            for c in chunks:
                ev = chunk_to_evidence(c, s["id"], s["query"])
                prev = candidates.get(ev.key)
                if prev is None or ev.relevance > prev["relevance"]:
                    candidates[ev.key] = ev.model_dump()
        return {"candidates": candidates, "warnings": sorted(set(warnings)), "_event": {"message": f"{len(candidates)} local passages from {len(subs)} queries"}}

    # ------------------------------------------------------------ 6b. web retrieval
    async def web_research(self, state: ResearchState) -> dict[str, Any]:
        ws = self.d.web.settings()
        subs = [s for s in state.get("subqueries", []) if s.get("source") in ("web", "both")] or state.get("subqueries", [])[:1]
        subs = subs[: ws.max_queries]
        pages: dict[str, WebPage] = {u: WebPage.model_validate(p) for u, p in (state.get("web_pages") or {}).items()}
        evs, warnings = await self.d.web.research_many([(s["query"], s["id"]) for s in subs], pages)
        candidates: dict[str, dict] = {e.key: e.model_dump() for e in evs}
        fetched = {u: p.model_dump() for u, p in pages.items()}
        n_pages = len({e["url"] for e in candidates.values() if not e["snippet_only"]})
        return {
            "candidates": candidates,
            "web_pages": fetched,
            "warnings": warnings,
            "_event": {"message": f"{len(candidates)} web passages from {n_pages} fetched pages ({len(subs)} searches)"},
        }

    # ------------------------------------------------------------ 7. reranking + selection
    async def rerank_evidence(self, state: ResearchState) -> dict[str, Any]:
        cands = [Evidence.model_validate(v) for v in (state.get("candidates") or {}).values()]
        existing = _ev(state)
        by_key = {e.key: e for e in existing.values()}
        warnings: list[str] = []
        scores: dict[str, float] = {}
        if cands and self.d.router.is_configured(Task.RERANKING):
            try:
                rr = await self.d.router.rerank(state["question"], [RerankCandidate(id=c.key, text=c.text[:2000]) for c in cands[:120]], top_n=min(len(cands), 120))
                scores = {r.id: r.score for r in rr.results}
            except LLM_ERRORS as exc:
                warnings.append(f"evidence reranking failed; using retrieval order: {exc}")
        ordered = sorted(cands, key=lambda c: (scores.get(c.key, float("-inf")) if scores else c.relevance), reverse=True)
        if scores:
            for c in ordered:
                c.retrieval["question_rerank_score"] = scores.get(c.key)
        # Two passes: (1) fill quotas — up to a third of slots for web evidence when present, and at most
        # max(6, cap/3) items per document/domain for diversity; (2) fill any remaining slots in rank order.
        cap = max(0, self.max_evidence - len(existing))
        fresh = [c for c in ordered if c.key not in by_key]
        web_cap = self.max_evidence // 3 if any(c.source_type == "web" for c in fresh) else 0
        per_src_cap = max(6, self.max_evidence // 3)
        selected: list[Evidence] = []
        per_src: dict[str, int] = {}
        for c in fresh:
            if len(selected) >= cap:
                break
            src = c.document_id or c.domain or "?"
            n_web = sum(1 for x in selected if x.source_type == "web")
            quota_ok = n_web < web_cap if c.source_type == "web" else (len(selected) - n_web) < cap - web_cap
            if quota_ok and per_src.get(src, 0) < per_src_cap:
                selected.append(c)
                per_src[src] = per_src.get(src, 0) + 1
        chosen = {c.key for c in selected}
        for c in fresh:
            if len(selected) >= cap:
                break
            if c.key not in chosen:
                selected.append(c)
                chosen.add(c.key)
        counter = state.get("eid_counter", 0)
        evidence = {k: v.model_dump() for k, v in existing.items()}
        for c in selected:
            counter += 1
            c.id = f"E{counter}"
            evidence[c.id] = c.model_dump()
        return {
            "evidence": evidence,
            "eid_counter": counter,
            "last_round_new": len(selected),
            "warnings": warnings,
            "_event": {"message": f"{len(selected)} new evidence items selected ({len(evidence)} total)", "reranked": bool(scores)},
        }

    # ------------------------------------------------------------ 8. table & figure inspection
    async def inspect_visuals(self, state: ResearchState) -> dict[str, Any]:
        evidence = _ev(state)
        visual = [e for e in evidence.values() if e.source_type == "local" and not e.inspected_visually and (e.element_type in ("figure",) or (e.element_type == "caption" and e.text.lower().startswith(("figure", "fig"))))]
        if not visual:
            return {"_event": {"message": "no figures among evidence"}}
        if not self.tools.vision_available():
            notes = [f"{len(visual)} figure(s) were located by caption but not visually inspected (no vision provider configured)."]
            return {"warnings": notes, "_event": {"message": notes[0]}}
        inspected = 0
        warnings: list[str] = []
        for e in visual[:3]:
            try:
                text, model = await self.tools.inspect_figure(e, state["question"])
                e.visual_observation, e.visual_model, e.inspected_visually = text, model, True
                inspected += 1
            except LLM_ERRORS as exc:
                warnings.append(f"visual inspection of {e.id} failed: {exc}")
            except Exception as exc:  # rendering failure
                warnings.append(f"could not render {e.id} for inspection: {type(exc).__name__}")
        return {"evidence": {k: v.model_dump() for k, v in evidence.items()}, "warnings": warnings, "_event": {"message": f"{inspected} figure(s) inspected by vision model"}}

    # ------------------------------------------------------------ 9. sufficiency
    async def assess_sufficiency(self, state: ResearchState) -> dict[str, Any]:
        evidence = _ev(state)
        it = state.get("iteration", 0)
        web_ok, _ = self.d.web.available()
        if it >= self.max_iterations:
            s = Sufficiency(sufficient=True, gaps=["Maximum research iterations reached; remaining gaps were not investigated further."])
            return {"sufficiency": s.model_dump(), "_event": {"message": "iteration limit reached"}}
        if it > 0 and state.get("last_round_new", 1) == 0:
            # Diminishing returns: the last follow-up round found nothing new in the available sources.
            prev = (state.get("sufficiency") or {}).get("gaps", [])
            note = "Follow-up searches found no further relevant evidence in the available sources."
            s = Sufficiency(sufficient=True, gaps=prev + [note])
            return {"sufficiency": s.model_dump(), "_event": {"message": "stopping: last follow-up round found no new evidence"}}
        if not evidence:
            follow = [SubQuery(query=state["question"], source="web" if web_ok else "local")]
            s = Sufficiency(sufficient=False, gaps=["No relevant evidence found."], followups=follow if web_ok and not state.get("use_web") else [], needs_web=web_ok)
        else:
            blocks = "\n\n".join(e.prompt_block(500) for e in evidence.values())
            asked = [q["query"] for q in state.get("query_history", [])]
            try:
                s = await chat_json(
                    self.d.router,
                    Task.PLANNING,
                    Sufficiency,
                    system=(
                        "Assess whether the collected evidence is sufficient to answer the research question rigorously, covering every "
                        "concept and comparison dimension with source-backed facts. If not, list concrete gaps and up to 3 follow-up search "
                        "queries that differ from the queries already tried. "
                        + ("Set needs_web=true if the gaps likely require web sources." if web_ok else "Web search is unavailable.")
                    ),
                    user=f"{_q(state['question'])}\nAlready tried: {asked}\n\nEVIDENCE:\n{blocks}",
                    max_tokens=900,
                )
            except LLM_ERRORS as exc:
                return {"sufficiency": {"sufficient": True, "gaps": [], "followups": []}, "warnings": [f"sufficiency assessment failed; proceeding: {exc}"], "_event": {"message": "assessment unavailable"}}
        tried = {q["query"].strip().lower() for q in state.get("query_history", [])}
        s.followups = [f for f in s.followups if f.query.strip().lower() not in tried][:3]
        if not s.followups and not (s.needs_web and web_ok and not state.get("use_web")):
            s.sufficient = True
        return {"sufficiency": s.model_dump(), "_event": {"message": "sufficient" if s.sufficient else f"insufficient: {len(s.gaps)} gap(s)", "gaps": s.gaps}}

    def after_sufficiency(self, state: ResearchState) -> str:
        return "compare_sources" if (state.get("sufficiency") or {}).get("sufficient", True) else "follow_up"

    # ------------------------------------------------------------ 10. follow-up searches
    async def follow_up(self, state: ResearchState) -> dict[str, Any]:
        suff = Sufficiency.model_validate(state["sufficiency"])
        web_ok, _ = self.d.web.available()
        use_web = state.get("use_web", False) or (suff.needs_web and web_ok and (self._opts(state).get("web_mode") or self.d.web.settings().mode) != "never")
        follow = suff.followups or [SubQuery(query=g, source="both") for g in suff.gaps[:2]]
        warnings: list[str] = []
        try:
            rq = await chat_json(
                self.d.router,
                Task.REWRITING,
                RewrittenQueries,
                system="Rewrite each information need into one effective keyword search query (no quotes, no operators). Return the same number of queries in order.",
                user="\n".join(f"- {f.query}" for f in follow),
                max_tokens=400,
            )
            if len(rq.queries) == len(follow):
                for f, q in zip(follow, rq.queries):
                    if q.strip():
                        f.query = q.strip()
        except LLM_ERRORS as exc:
            warnings.append(f"query rewriting unavailable: {exc}")
        it = state.get("iteration", 0) + 1
        subs = [{"id": f"Q{it}.{i + 1}", **f.model_dump()} for i, f in enumerate(follow)]
        if use_web and not state.get("use_web"):
            for s in subs:
                s["source"] = "both"
        return {
            "subqueries": subs,
            "query_history": subs,
            "iteration": it,
            "use_web": use_web,
            "web_reason": state.get("web_reason") if state.get("use_web") else ("local evidence insufficient" if use_web else state.get("web_reason")),
            "warnings": warnings,
            "_event": {"message": f"round {it}: {len(subs)} follow-up queries", "subqueries": subs, "round": it},
        }

    # ------------------------------------------------------------ 11/12. comparison & contradictions
    async def compare_sources(self, state: ResearchState) -> dict[str, Any]:
        evidence = _ev(state)
        if len(evidence) < 2:
            return {"comparison": Comparison().model_dump(), "_event": {"message": "too little evidence to compare"}}
        web_overview = self.tools.compare_web_sources(list(evidence.values()))
        blocks = "\n\n".join(e.prompt_block(900) for e in evidence.values())
        try:
            c = await chat_json(
                self.d.router,
                Task.PLANNING,
                Comparison,
                system=(
                    "Compare the evidence across sources along the question's dimensions. Record what each source reports with its evidence id. "
                    "Identify contradictions: sources reporting different values or conclusions for the same quantity. Note when differences "
                    "may be explained by different hardware, models, datasets, workloads or settings. Only use the evidence provided."
                ),
                user=f"{_q(state['question'])}\nDimensions: {(state.get('analysis') or {}).get('comparison_dimensions')}\nWeb source overview: {web_overview}\n\nEVIDENCE:\n{blocks}",
                max_tokens=2000,
            )
        except LLM_ERRORS as exc:
            return {"comparison": Comparison().model_dump(), "warnings": [f"cross-source comparison failed: {exc}"], "_event": {"message": "comparison unavailable"}}
        valid = set(evidence)
        for x in c.contradictions:
            x.evidence_ids = [i for i in x.evidence_ids if i in valid]
        c.contradictions = [x for x in c.contradictions if len(x.evidence_ids) >= 2]
        return {"comparison": c.model_dump(), "_event": {"message": f"{len(c.comparisons)} dimensions compared, {len(c.contradictions)} contradiction(s)"}}

    # ------------------------------------------------------------ 13. drafting
    async def draft_report(self, state: ResearchState) -> dict[str, Any]:
        evidence = _ev(state)
        gaps = (state.get("sufficiency") or {}).get("gaps", [])
        if not evidence:
            draft = ReportDraft(
                title="Insufficient evidence",
                executive_summary=[Statement(text="No relevant evidence was found in the selected documents or available web sources, so the question cannot be answered from sources.", kind="interpretation")],
                scope=state["question"],
                findings=[],
                limitations=["No source evidence was retrieved."],
                evidence_gaps=gaps or ["All aspects of the question remain unresolved."],
                follow_up_questions=["Upload documents that address the question, or enable a web search provider."],
            )
            return {"draft": draft.model_dump(), "_event": {"message": "no evidence; produced an insufficient-evidence report"}}
        blocks = "\n\n".join(e.prompt_block(1400) for e in evidence.values())
        system = (
            "Write a rigorous research report using ONLY the evidence provided. Rules:\n"
            "- Every factual statement of kind 'source' must cite the evidence ids that state it, in 'citations'.\n"
            "- Never invent numbers, citations, URLs, page numbers, or results. Copy numbers exactly with their units.\n"
            "- Keep experimental conditions (model size, hardware, dataset, batch size, sequence length) next to reported numbers.\n"
            "- Use kind='interpretation' for your own analysis, synthesis or recommendations.\n"
            "- Report contradictions explicitly and do not silently pick one side.\n"
            "- Distinguish uploaded-document evidence (type=local) from web evidence (type=web) where it matters.\n"
            "- Visual observations are model interpretations of images; say so when you rely on them.\n"
            "- List limitations, evidence gaps and unresolved questions honestly.\n"
            "- Comparison tables: one row per technique/source with citations for that row."
        )
        user = (
            f"{_q(state['question'])}\nAnalysis: {state.get('analysis')}\nComparison notes: {state.get('comparison')}\n"
            f"Known gaps: {gaps}\nWeb research used: {state.get('use_web')} ({state.get('web_reason')})\n\nEVIDENCE:\n{blocks}"
        )
        draft = await chat_json(self.d.router, Task.SYNTHESIS, ReportDraft, system=system, user=user, max_tokens=6000, temperature=0.2)
        return {"draft": draft.model_dump(), "_event": {"message": f"draft with {sum(1 for _ in iter_statements(draft))} claims"}}

    # ------------------------------------------------------------ 14. citation validation
    async def validate_citations(self, state: ResearchState) -> dict[str, Any]:
        evidence = _ev(state)
        draft = ReportDraft.model_validate(state["draft"])
        pages = {u: WebPage.model_validate(p) for u, p in (state.get("web_pages") or {}).items()}
        v = self.d.validator
        ev_checks = {eid: v.check_evidence(e, pages) for eid, e in evidence.items()}
        warnings: list[str] = []

        async def verify(items: list[tuple[str, Statement]]) -> dict[str, dict[str, Any]]:
            results: dict[str, dict[str, Any]] = {}
            to_judge = []
            for sid, s in items:
                s.citations = [c for c in dict.fromkeys(s.citations)]
                if s.kind == "interpretation":
                    results[sid] = {"status": "interpretation", "mechanical": None, "semantic": None}
                    continue
                mech = v.mechanical(s.text, s.citations, evidence, ev_checks)
                results[sid] = {"mechanical": mech.model_dump(), "semantic": None}
                if not s.citations:
                    results[sid]["status"] = "uncited"
                else:
                    to_judge.append({"id": sid, "text": s.text, "citations": s.citations})
            verdicts, w = await v.semantic(to_judge, evidence)
            warnings.extend(w)
            for item in to_judge:
                r = results[item["id"]]
                sem = verdicts.get(item["id"])
                r["semantic"] = sem.model_dump() if sem else None
                mech_ok = r["mechanical"]["ok"]
                if sem is None:
                    r["status"] = "mechanically_verified" if mech_ok else "unverified"
                elif sem.verdict == "supported" and mech_ok:
                    r["status"] = "verified"
                elif sem.verdict in ("supported", "partially_supported") and r["mechanical"]["citations_exist"]:
                    r["status"] = "partially_verified"
                else:
                    r["status"] = "unverified"
            return results

        statements = list(iter_statements(draft))
        results = await verify(statements)

        # One bounded repair pass: rewrite unverified claims so they state only what their evidence says.
        repaired = 0
        retry: list[tuple[str, Statement]] = []
        for sid, s in statements:
            if results[sid].get("status") == "unverified" and s.citations and all(c in evidence for c in s.citations):
                try:
                    rc = await chat_json(
                        self.d.router,
                        Task.REWRITING,
                        RewrittenClaim,
                        system="Rewrite the claim so it states only what the cited sources explicitly say (keep numbers and units exact). Return null if the sources do not support any version of it.",
                        user=f"CLAIM: {s.text}\n\nSOURCES:\n" + "\n\n".join(evidence[c].prompt_block(1200) for c in s.citations),
                        max_tokens=400,
                        temperature=0.0,
                    )
                    if rc.supported_rewrite and rc.supported_rewrite.strip() != s.text.strip():
                        s.text = rc.supported_rewrite.strip()
                        retry.append((sid, s))
                except LLM_ERRORS as exc:
                    warnings.append(f"claim rewrite failed for {sid}: {exc}")
        if retry:
            again = await verify(retry)
            for sid, r in again.items():
                r["rewritten"] = True
                if r["status"] in ("verified", "partially_verified", "mechanically_verified"):
                    repaired += 1
                results[sid] = r

        # Comparison-table rows: mechanical checks (numbers must appear in the cited evidence).
        for t in draft.comparison_tables:
            for i, row in enumerate(t.rows):
                row.citations = [c for c in row.citations if c in evidence]
                mech = v.mechanical(" ".join(row.cells), row.citations, evidence, ev_checks)
                results[f"T:{t.title}:{i}"] = {"status": "verified" if mech.ok else "unverified", "mechanical": mech.model_dump(), "semantic": None, "table_row": True}
        for c in draft.contradictions:
            c.citations = [x for x in c.citations if x in evidence]

        counts: dict[str, int] = {}
        for sid, r in results.items():
            if not r.get("table_row"):
                counts[r["status"]] = counts.get(r["status"], 0) + 1
        summary = {
            "total": sum(counts.values()),
            "verified": counts.get("verified", 0),
            "partially_verified": counts.get("partially_verified", 0) + counts.get("mechanically_verified", 0),
            "unverified": counts.get("unverified", 0) + counts.get("uncited", 0),
            "interpretation": counts.get("interpretation", 0),
            "repaired_by_rewrite": repaired,
            "evidence_checks": {k: c.model_dump() for k, c in ev_checks.items()},
            "web_citation_checks": self.tools.validate_web_citations([e for e in evidence.values() if e.source_type == "web"], pages),
        }
        return {
            "draft": draft.model_dump(),
            "verification": {"statements": results, "summary": summary},
            "warnings": warnings,
            "_event": {"message": f"{summary['verified']} verified, {summary['partially_verified']} partial, {summary['unverified']} unverified"},
        }

    # ------------------------------------------------------------ 15. final report
    async def finalize(self, state: ResearchState) -> dict[str, Any]:
        evidence = _ev(state)
        draft = ReportDraft.model_validate(state["draft"])
        ver = state.get("verification") or {"statements": {}, "summary": {}}
        meta = {
            "verification_summary": {k: v for k, v in ver.get("summary", {}).items() if k not in ("evidence_checks", "web_citation_checks")},
            "warnings": sorted(set(state.get("warnings") or [])),
            **self.d.usage(state["job_id"]),
        }
        md = render_markdown(state["question"], draft, evidence, ver.get("statements", {}), meta)
        report = {
            "draft": draft.model_dump(),
            "verification": ver,
            "web_used": bool(state.get("use_web")),
            "web_reason": state.get("web_reason"),
            "iterations": state.get("iteration", 0),
            "queries": state.get("query_history", []),
            "comparison": state.get("comparison"),
            "warnings": meta["warnings"],
        }
        return {"report_md": md, "report": report, "_event": {"message": "report ready"}}
