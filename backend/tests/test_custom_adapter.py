"""Mode C: a custom API is configured purely through validated configuration."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.providers.base import ChatMessage, ChatRequest, EmbeddingRequest, RerankCandidate, RerankRequest, WebSearchRequest
from app.providers.custom.http_adapter import render_template, select

CUSTOM = {
    "base_url": "https://custom.example/api",
    "auth": {"type": "header", "header_name": "X-Token"},
    "chat": {
        "path": "generate",
        "model": "house-llm-7b",
        "body_template": {"model": "{{model}}", "prompt": "{{prompt}}", "system_prompt": "{{system}}", "options": {"temp": "{{temperature}}"}, "tag": "q={{model}}"},
        "response_mapping": {"text": "output.text", "input_tokens": "meta.tokens.in", "output_tokens": "meta.tokens.out"},
    },
    "embedding": {
        "path": "vectors",
        "model": "house-embed",
        "body_template": {"inputs": "{{texts}}", "kind": "{{input_type}}"},
        "response_mapping": {"vectors": "items[*].vec"},
    },
    "reranking": {
        "path": "score",
        "body_template": {"q": "{{query}}", "docs": "{{documents}}"},
        "response_mapping": {"scores": "scores"},
    },
    "web_search": {
        "method": "GET",
        "path": "search",
        "query_params": {"q": "{{query}}", "n": "{{max_results}}"},
        "response_mapping": {"items": "hits", "url": "link", "title": "name", "snippet": "summary"},
    },
}


def test_template_rendering_is_typed_and_inert():
    t = {"a": "{{texts}}", "b": "x-{{query}}-y", "c": ["{{top_n}}", 3], "d": "{{__import__}}"}
    out = render_template(t, {"texts": ["p", "q"], "query": "Q", "top_n": 5})
    # Names are looked up in a dict only; unknown names render as null (and are rejected at config time).
    assert out == {"a": ["p", "q"], "b": "x-Q-y", "c": [5, 3], "d": None}
    assert select({"a": [{"b": 1}, {"b": 2}]}, "a[*].b") == [1, 2]
    assert select({"a": [{"b": 1}]}, "a[0].b") == 1


@pytest.mark.parametrize(
    "mutation,needle",
    [
        ({"chat": {**CUSTOM["chat"], "body_template": {"x": "{{os_environ}}"}}}, "unknown template variables"),
        ({"chat": {**CUSTOM["chat"], "path": "{{model}}/x"}}, "static"),
        ({"chat": {**CUSTOM["chat"], "path": "https://evil.example/x"}}, "static"),
        ({"chat": {**CUSTOM["chat"], "response_mapping": {}}}, "response_mapping"),
        ({"chat": {**CUSTOM["chat"], "response_mapping": {"text": "a..b"}}}, "Invalid path"),
        ({"extra_headers": {"Authorization": "Bearer x"}}, "credentials"),
    ],
)
def test_custom_config_validation_rejects_unsafe(registry, mutation, needle):
    with pytest.raises(ValueError) as ei:
        registry.create("c", "custom_http", {**CUSTOM, **mutation}, {"api_key": "k"})
    assert needle in str(ei.value)


@respx.mock
async def test_custom_adapter_all_capabilities(registry):
    rec = registry.create("custom", "custom_http", CUSTOM, {"api_key": "custom-secret-1"})
    assert {c.value for c in rec.capabilities} == {"chat", "embedding", "reranking", "web_search"}
    p = registry.get(rec.id)

    chat = respx.post("https://custom.example/api/generate").mock(
        return_value=httpx.Response(200, json={"output": {"text": "answer"}, "meta": {"tokens": {"in": 11, "out": 4}}})
    )
    r = await p.chat(ChatRequest(messages=[ChatMessage(role="system", content="S"), ChatMessage(role="user", content="U")], temperature=0.1))
    assert r.text == "answer" and r.usage.input_tokens == 11
    body = json.loads(chat.calls[0].request.content)
    assert body == {"model": "house-llm-7b", "prompt": "U", "system_prompt": "S", "options": {"temp": 0.1}, "tag": "q=house-llm-7b"}
    assert chat.calls[0].request.headers["x-token"] == "custom-secret-1"

    respx.post("https://custom.example/api/vectors").mock(return_value=httpx.Response(200, json={"items": [{"vec": [0, 2]}, {"vec": [2, 0]}]}))
    e = await p.embed(EmbeddingRequest(texts=["a", "b"]))
    assert e.vectors == [[0.0, 1.0], [1.0, 0.0]]

    respx.post("https://custom.example/api/score").mock(return_value=httpx.Response(200, json={"scores": [0.1, 0.9, 0.5]}))
    rr = await p.rerank(RerankRequest(query="q", candidates=[RerankCandidate(id=x, text=x) for x in "abc"]))
    assert [x.id for x in rr.results] == ["b", "c", "a"]

    search = respx.get("https://custom.example/api/search").mock(
        return_value=httpx.Response(200, json={"hits": [{"link": "https://s.org/1", "name": "One", "summary": "S1"}, {"name": "no link"}]})
    )
    s = await p.search(WebSearchRequest(query="kv cache", max_results=5))
    assert [(x.url, x.title) for x in s.results] == [("https://s.org/1", "One")]
    assert search.calls[0].request.url.params["q"] == "kv cache"


@respx.mock
async def test_custom_adapter_bad_mapping_is_normalized_error(registry):
    rec = registry.create("custom2", "custom_http", {"base_url": "https://custom.example/api", "auth": {"type": "none"}, "chat": CUSTOM["chat"]})
    respx.post("https://custom.example/api/generate").mock(return_value=httpx.Response(200, json={"unexpected": True}))
    from app.providers.base import ProviderError, ProviderErrorKind

    with pytest.raises(ProviderError) as ei:
        await registry.get(rec.id).chat(ChatRequest(messages=[ChatMessage(role="user", content="U")]))
    assert ei.value.kind == ProviderErrorKind.INVALID_RESPONSE
