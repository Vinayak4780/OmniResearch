"""Notebooks: sources, guide, grounded chat, notes and notebook-scoped deep research over HTTP."""

from __future__ import annotations

import json
import re

from tests import fakes
from tests.test_agent import scripted_llm, sources
from tests.test_api import CORPUS, make_client, wait_until


def notebook_llm(state: dict):
    base = scripted_llm(state)

    def handler(req, label):
        prompt = "\n".join(m.content for m in req.messages)
        if "orientation for a notebook" in prompt:
            state["guide_calls"] = state.get("guide_calls", 0) + 1
            return json.dumps({"summary": "Sources on serving LLMs with limited GPU memory.", "questions": ["How much memory does INT4 save?", "Q2?", "Q3?"]})
        if "answering questions about the user's sources" in prompt:
            state.setdefault("chat_prompts", []).append(prompt)
            src = sources(prompt)
            q = next((k for k, v in src.items() if "7.4 GB" in v), None)
            if q is None:
                return json.dumps({"answer": "The selected sources do not cover that.", "citations": [], "answered_from_sources": False})
            return json.dumps({"answer": f"INT4 weights need 7.4 GB [{q}] and keep 99.9% accuracy [{q}][E77].", "citations": [q]})
        return base(req, label)

    return handler


def test_notebook_workflow(tmp_path):
    client, c = make_client(tmp_path)
    st: dict = {}
    fakes.CHAT_HANDLERS["main"] = notebook_llm(st)
    with client:
        for pid, adapter, cfg in [("llm", "fake_chat", {"label": "l", "script": "main"}), ("emb", "fake_embed", {"label": "e", "dims": 24}), ("rr", "local_lexical_rerank", {})]:
            client.post("/api/providers", json={"id": pid, "name": pid, "adapter": adapter, "config": cfg})
        for task, pid in [("synthesis", "llm"), ("text_embedding", "emb"), ("reranking", "rr")]:
            client.put(f"/api/routes/{task}", json={"task": task, "provider_ids": [pid]})

        nb = client.post("/api/notebooks", json={"title": "GPU serving"}).json()
        nb_id = nb["id"]
        assert client.get(f"/api/notebooks/{nb_id}/guide").json()["summary"] is None  # no sources yet
        r = client.post(f"/api/notebooks/{nb_id}/messages", json={"message": "anything?"})
        assert r.status_code == 400 and "source" in r.json()["detail"]

        ids = {}
        for f in ("synthetic_quantization.pdf", "synthetic_kvcache.pdf"):
            up = client.post("/api/documents", files={"file": (f, (CORPUS / f).read_bytes(), "application/pdf")}, data={"collection": nb_id})
            ids[f] = up.json()["document"]["id"]
        wait_until(lambda: all(s["status"] == "ready" for s in client.get(f"/api/notebooks/{nb_id}/sources").json()))
        listed = {n["id"]: n for n in client.get("/api/notebooks").json()}
        assert listed[nb_id]["source_count"] == 2

        g1 = client.get(f"/api/notebooks/{nb_id}/guide").json()
        g2 = client.get(f"/api/notebooks/{nb_id}/guide").json()
        assert g1["questions"][0].startswith("How much memory") and g1 == g2 and st["guide_calls"] == 1  # cached

        ans = client.post(f"/api/notebooks/{nb_id}/messages", json={"message": "How much memory does INT4 quantization save?"}).json()
        assert ans["role"] == "assistant" and "7.4 GB" in ans["content"]
        cite = next(iter(ans["citations"].values()))
        assert cite["document_id"] == ids["synthetic_quantization.pdf"] and cite["page_no"] == 1 and cite["link"].startswith("/documents/")
        assert ans["check"]["unknown_citations"] == ["E77"] and ans["check"]["numbers_not_in_cited_evidence"] == ["99.9"]

        # selecting only the KV-cache source changes what the chat can see
        ans2 = client.post(f"/api/notebooks/{nb_id}/messages", json={"message": "And the memory saving?", "source_ids": [ids["synthetic_kvcache.pdf"]]}).json()
        assert ans2["check"]["answered_from_sources"] is False and ans2["source_ids"] == [ids["synthetic_kvcache.pdf"]]
        assert "How much memory does INT4" in st["chat_prompts"][-1]  # follow-up carries the conversation
        assert [m["role"] for m in client.get(f"/api/notebooks/{nb_id}/messages").json()] == ["user", "assistant"] * 2

        note = client.post(f"/api/notebooks/{nb_id}/notes", json={"message_id": ans["id"]}).json()
        assert note["title"].startswith("How much memory") and note["citations"]
        client.post(f"/api/notebooks/{nb_id}/notes", json={"content": "My own note"})
        assert len(client.get(f"/api/notebooks/{nb_id}/notes").json()) == 2

        rep = client.post(f"/api/notebooks/{nb_id}/reports", json={"question": "Compare quantization and paging.", "web_mode": "never"}).json()
        wait_until(lambda: client.get(f"/api/research/{rep['id']}").json()["status"] in ("completed", "failed"))
        assert client.get(f"/api/research/{rep['id']}").json()["status"] == "completed"
        assert [r["id"] for r in client.get(f"/api/notebooks/{nb_id}/reports").json()] == [rep["id"]]
        ev = client.get(f"/api/research/{rep['id']}/evidence").json()
        assert ev and {e["document_id"] for e in ev} <= set(ids.values())  # scoped to the notebook

        assert client.delete(f"/api/notebooks/{nb_id}").status_code == 204
        assert client.get(f"/api/notebooks/{nb_id}").status_code == 404
        assert all(client.get(f"/api/documents/{d}").status_code == 404 for d in ids.values())
        assert client.get(f"/api/research/{rep['id']}").status_code == 404


def test_existing_collections_become_notebooks(tmp_path):
    """Migration 5 turns document collections into notebooks so nothing disappears."""
    from app.repositories.db import MIGRATIONS, Database

    db = Database(tmp_path / "old.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)")
    for i, script in enumerate(MIGRATIONS[:4], 1):
        for stmt in [s for s in script.split(";") if s.strip()]:
            db.execute(stmt)
        db.execute("INSERT INTO schema_migrations VALUES (?, 'x')", (i,))
    db.execute("INSERT INTO documents(id, filename, sha256, size_bytes, collection, status, created_at, updated_at) VALUES ('d1','a.pdf','h',1,'papers','ready','2026-01-01','2026-01-02')")
    db.migrate()
    assert db.one("SELECT id, title FROM notebooks") == {"id": "papers", "title": "papers"}
    assert re.match(r"\d", db.one("SELECT MAX(version) v FROM schema_migrations")["v"].__str__())
