"""Routing, fallback, retries, circuit breaking, capability safety and credential handling."""

from __future__ import annotations

import json
import logging

import pytest

from app.observability.logging import JsonFormatter
from app.providers.base import ChatMessage, ChatRequest, NoProviderAvailableError, Task
from app.providers.registry import ConfigConflictError, RouteConfig
from tests import fakes

REQ = ChatRequest(messages=[ChatMessage(role="user", content="question")])


def _chat(registry, pid, label, **cfg):
    return registry.create(pid, "fake_chat", {"label": label, **cfg}, provider_id=pid)


async def test_switching_chat_provider_requires_no_code_change(registry, router):
    _chat(registry, "alpha", "alpha")
    _chat(registry, "beta", "beta")
    registry.set_route(RouteConfig(task=Task.SYNTHESIS, provider_ids=["alpha"]))
    assert (await router.chat(Task.SYNTHESIS, REQ)).text.startswith("[alpha]")
    registry.set_route(RouteConfig(task=Task.SYNTHESIS, provider_ids=["beta"]))
    assert (await router.chat(Task.SYNTHESIS, REQ)).text.startswith("[beta]")


async def test_tasks_route_independently(registry, router):
    _chat(registry, "planner", "planner")
    _chat(registry, "writer", "writer")
    registry.set_route(RouteConfig(task=Task.PLANNING, provider_ids=["planner"]))
    registry.set_route(RouteConfig(task=Task.SYNTHESIS, provider_ids=["writer"], params={"model": "big-model", "temperature": 0.0}))
    assert (await router.chat(Task.PLANNING, REQ)).text.startswith("[planner]")
    r = await router.chat(Task.SYNTHESIS, REQ)
    assert r.text.startswith("[writer]") and r.model == "big-model"


async def test_fallback_order_and_metrics(registry, router, db):
    _chat(registry, "flaky", "flaky", fail_kind="unavailable", max_retries=1)
    _chat(registry, "backup", "backup")
    registry.set_route(RouteConfig(task=Task.SYNTHESIS, provider_ids=["flaky", "backup"]))
    r = await router.chat(Task.SYNTHESIS, REQ)
    assert r.text.startswith("[backup]")
    assert fakes.CALLS["flaky"] == 2  # retried once, then fell back
    rows = db.all("SELECT provider_id, ok, error_kind, fallback_from FROM provider_calls ORDER BY id")
    assert rows == [
        {"provider_id": "flaky", "ok": 0, "error_kind": "unavailable", "fallback_from": None},
        {"provider_id": "flaky", "ok": 0, "error_kind": "unavailable", "fallback_from": None},
        {"provider_id": "backup", "ok": 1, "error_kind": None, "fallback_from": "flaky"},
    ]


async def test_fallback_disabled(registry, router):
    _chat(registry, "flaky", "flaky", fail_kind="auth")
    _chat(registry, "backup", "backup")
    registry.set_route(RouteConfig(task=Task.SYNTHESIS, provider_ids=["flaky", "backup"], fallback_enabled=False))
    with pytest.raises(NoProviderAvailableError) as ei:
        await router.chat(Task.SYNTHESIS, REQ)
    assert [a["provider_id"] for a in ei.value.attempts] == ["flaky"]
    assert fakes.CALLS["flaky"] == 1  # auth errors are not retried


async def test_transient_retry_then_success(registry, router):
    _chat(registry, "rr", "r", fail_kind="rate_limit", fail_times=2, max_retries=2)
    registry.set_route(RouteConfig(task=Task.PLANNING, provider_ids=["rr"]))
    assert (await router.chat(Task.PLANNING, REQ)).text.startswith("[r]")
    assert fakes.CALLS["rr"] == 3


async def test_circuit_breaker_opens(registry, router):
    _chat(registry, "down", "down", fail_kind="unavailable", max_retries=0)
    _chat(registry, "up", "up")
    registry.set_route(RouteConfig(task=Task.PLANNING, provider_ids=["down", "up"]))
    for _ in range(3):
        await router.chat(Task.PLANNING, REQ)
    assert router.breaker("down", __import__("app.providers.base", fromlist=["Capability"]).Capability.CHAT).state == "open"
    await router.chat(Task.PLANNING, REQ)
    assert fakes.CALLS["down"] == 3  # skipped while open


async def test_capability_mismatch_rejected(registry, router):
    registry.create("emb", "fake_embed", {"label": "e"}, provider_id="emb")
    _chat(registry, "chat", "c")
    with pytest.raises(ValueError, match="does not provide capability 'chat'"):
        registry.set_route(RouteConfig(task=Task.SYNTHESIS, provider_ids=["emb"]))
    with pytest.raises(ValueError, match="does not provide capability 'embedding'"):
        registry.set_route(RouteConfig(task=Task.TEXT_EMBEDDING, provider_ids=["chat"]))
    # Even if the DB were edited directly, the router refuses to substitute at runtime.
    registry.db.execute("INSERT INTO capability_routes VALUES ('text_embedding', ?, 1, '{}', 'now')", (json.dumps(["chat"]),))
    with pytest.raises(NoProviderAvailableError) as ei:
        await router.embed(["x"])
    assert ei.value.attempts[0]["kind"] == "unsupported"


async def test_embedding_fallback_only_to_identical_space(registry, router):
    registry.create("e1", "fake_embed", {"label": "space1", "fail_kind": "unavailable", "max_retries": 0}, provider_id="e1")
    registry.create("e2", "fake_embed", {"label": "space2"}, provider_id="e2")
    registry.set_route(RouteConfig(task=Task.TEXT_EMBEDDING, provider_ids=["e1", "e2"]))
    with pytest.raises(NoProviderAvailableError) as ei:
        await router.embed(["x"], provider_id="e1")
    assert any(a["kind"] == "incompatible" for a in ei.value.attempts)
    # An explicitly confirmed replica (same compatibility key) may serve as fallback.
    registry.update("e1", config={"label": "space1", "fail_kind": "unavailable", "max_retries": 0, "compatibility_key": "space-A"})
    registry.create("e3", "fake_embed", {"label": "space1", "compatibility_key": "space-A"}, provider_id="e3")
    registry.set_route(RouteConfig(task=Task.TEXT_EMBEDDING, provider_ids=["e1", "e2", "e3"]))
    r = await router.embed(["x"], provider_id="e1")
    assert len(r.vectors) == 1 and fakes.CALLS.get("e3") == 1  # served by the confirmed replica
    assert fakes.CALLS.get("e2") is None


async def test_secrets_encrypted_and_never_returned(registry, db, secret_box, caplog):
    rec = registry.create("k", "openai_compatible", {"base_url": "https://x.example/v1", "chat_model": "m"}, {"api_key": "sk-THISISSECRET999"})
    assert rec.secrets_set == {"api_key": True}
    assert "sk-THISISSECRET999" not in rec.model_dump_json()
    raw = db.one("SELECT secrets_enc_json FROM provider_configs WHERE id=?", (rec.id,))["secrets_enc_json"]
    assert "sk-THISISSECRET999" not in raw
    assert secret_box.decrypt(json.loads(raw)["api_key"]) == "sk-THISISSECRET999"
    # Keep existing secret when None, delete when ""
    registry.update(rec.id, secrets={"api_key": None})
    assert registry.get_record(rec.id).secrets_set["api_key"] is True
    registry.update(rec.id, secrets={"api_key": ""})
    assert registry.get_record(rec.id).secrets_set["api_key"] is False
    # Log redaction
    rec_ = logging.LogRecord("x", logging.INFO, "f", 1, "calling with key sk-THISISSECRET999 and Authorization: Bearer abcdef123", None, None)
    line = JsonFormatter().format(rec_)
    assert "sk-THISISSECRET999" not in line and "abcdef123" not in line


async def test_safe_delete(registry, router):
    _chat(registry, "used", "u")
    registry.set_route(RouteConfig(task=Task.PLANNING, provider_ids=["used"]))
    with pytest.raises(ConfigConflictError) as ei:
        registry.delete("used")
    assert ei.value.references == ["route:planning"]
    registry.delete("used", force=True)
    assert registry.get_route(Task.PLANNING).provider_ids == []


async def test_disabled_provider_skipped(registry, router):
    _chat(registry, "pa", "a")
    _chat(registry, "pb", "b")
    registry.set_route(RouteConfig(task=Task.PLANNING, provider_ids=["pa", "pb"]))
    registry.update("pa", enabled=False)
    assert (await router.chat(Task.PLANNING, REQ)).text.startswith("[b]")
