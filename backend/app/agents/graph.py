"""The research workflow as a LangGraph StateGraph.

    analyze → plan → decompose → route_sources ─┬─> retrieve_local ─┐
                                                └─> web_research ───┴─> rerank_evidence → inspect_visuals
    → assess_sufficiency ─(insufficient, bounded)─> follow_up ─┬─> retrieve_local / web_research (parallel)
                         └─(sufficient)─> compare_sources → draft_report → validate_citations → finalize
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import ResearchNodes
from app.agents.state import ResearchState
from app.agents.tools import ResearchDeps


def build_graph(deps: ResearchDeps, checkpointer=None):
    n = ResearchNodes(deps)
    g = StateGraph(ResearchState)
    steps = {
        "analyze_question": n.analyze_question,
        "plan_research": n.plan_research,
        "decompose": n.decompose,
        "route_sources": n.route_sources,
        "retrieve_local": n.retrieve_local,
        "web_research": n.web_research,
        "rerank_evidence": n.rerank_evidence,
        "inspect_visuals": n.inspect_visuals,
        "assess_sufficiency": n.assess_sufficiency,
        "follow_up": n.follow_up,
        "compare_sources": n.compare_sources,
        "draft_report": n.draft_report,
        "validate_citations": n.validate_citations,
        "finalize": n.finalize,
    }
    for name, fn in steps.items():
        g.add_node(name, n.node(name)(fn))

    g.add_edge(START, "analyze_question")
    g.add_edge("analyze_question", "plan_research")
    g.add_edge("plan_research", "decompose")
    g.add_edge("decompose", "route_sources")
    g.add_conditional_edges("route_sources", n.fan_out, ["retrieve_local", "web_research"])
    g.add_edge("retrieve_local", "rerank_evidence")
    g.add_edge("web_research", "rerank_evidence")
    g.add_edge("rerank_evidence", "inspect_visuals")
    g.add_edge("inspect_visuals", "assess_sufficiency")
    g.add_conditional_edges("assess_sufficiency", n.after_sufficiency, ["follow_up", "compare_sources"])
    g.add_conditional_edges("follow_up", n.fan_out, ["retrieve_local", "web_research"])
    g.add_edge("compare_sources", "draft_report")
    g.add_edge("draft_report", "validate_citations")
    g.add_edge("validate_citations", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
