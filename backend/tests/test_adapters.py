"""Adapters normalize incompatible vendor wire formats into the same application contracts."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    ProviderError,
    ProviderErrorKind,
    RerankCandidate,
    RerankRequest,
    WebSearchRequest,
)

CANDS = [RerankCandidate(id="a", text="KV cache paging"), RerankCandidate(id="b", text="pasta"), RerankCandidate(id="c", text="quantization")]


@respx.mock
async def test_openai_compatible_chat_nim_style(registry):
    rec = registry.create(
        "NIM", "openai_compatible",
        {"base_url": "https://integrate.api.nvidia.com/v1", "chat_model": "meta/llama-3.3-70b-instruct", "supports_json_mode": True},
        {"api_key": "nvapi-SECRET123456"},
    )
    route = respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "meta/llama-3.3-70b-instruct", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})
    )
    p = registry.get(rec.id)
    r = await p.chat(ChatRequest(messages=[ChatMessage(role="system", content="s"), ChatMessage(role="user", content="hi")], json_mode=True))
    assert r.text == "hello" and r.usage.input_tokens == 7 and r.usage.output_tokens == 2
    sent = json.loads(route.calls[0].request.content)
    assert sent["response_format"] == {"type": "json_object"}
    assert route.calls[0].request.headers["authorization"] == "Bearer nvapi-SECRET123456"


@respx.mock
async def test_openai_compatible_no_system_role_and_no_json_mode(registry):
    rec = registry.create("vllm", "openai_compatible", {"base_url": "http://localhost:8000/v1", "chat_model": "m", "supports_system_role": False})
    route = respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]})
    )
    r = await registry.get(rec.id).chat(ChatRequest(messages=[ChatMessage(role="system", content="SYS"), ChatMessage(role="user", content="U")], json_mode=True))
    assert r.text == "ab"
    sent = json.loads(route.calls[0].request.content)
    assert "response_format" not in sent
    assert sent["messages"] == [{"role": "user", "content": "SYS\n\nU"}]
    assert "authorization" not in route.calls[0].request.headers


@respx.mock
async def test_openai_compatible_embeddings_input_type_and_order(registry):
    rec = registry.create("emb", "openai_compatible", {"base_url": "https://e.example/v1", "embedding_model": "nv-embedqa", "embedding_input_type_field": "input_type", "embedding_batch_size": 2})
    route = respx.post("https://e.example/v1/embeddings").mock(
        side_effect=lambda req: httpx.Response(200, json={"data": [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(json.loads(req.content)["input"])))]})
    )
    r = await registry.get(rec.id).embed(EmbeddingRequest(texts=["a", "b", "c"], input_type="query"))
    assert len(r.vectors) == 3 and r.vectors[0] == pytest.approx([0.6, 0.8])
    assert len(route.calls) == 2
    assert json.loads(route.calls[0].request.content)["input_type"] == "query"


@respx.mock
async def test_anthropic_native(registry):
    rec = registry.create("claude", "anthropic", {"chat_model": "claude-sonnet-5-5"}, {"api_key": "sk-ant-XYZXYZXYZ"})
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"model": "claude-sonnet-5-5", "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}})
    )
    r = await registry.get(rec.id).chat(ChatRequest(messages=[ChatMessage(role="system", content="SYS"), ChatMessage(role="user", content="U")]))
    assert r.text == "hi" and r.usage.output_tokens == 1
    sent = json.loads(route.calls[0].request.content)
    assert sent["system"] == "SYS" and sent["messages"] == [{"role": "user", "content": "U"}]
    assert route.calls[0].request.headers["x-api-key"] == "sk-ant-XYZXYZXYZ"


@pytest.mark.parametrize(
    "dialect,path,response",
    [
        ("cohere", "v2/rerank", {"results": [{"index": 2, "relevance_score": 0.7}, {"index": 0, "relevance_score": 0.9}]}),
        ("jina", "v1/rerank", {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 2, "relevance_score": 0.7}]}),
        ("voyage", "v1/rerank", {"data": [{"index": 0, "relevance_score": 0.9}, {"index": 2, "relevance_score": 0.7}]}),
        ("nvidia_nim", "v1/ranking", {"rankings": [{"index": 0, "logit": 5.1}, {"index": 2, "logit": 1.0}]}),
        ("tei", "rerank", [{"index": 2, "score": 0.1}, {"index": 0, "score": 0.95}]),
    ],
)
@respx.mock
async def test_rerank_dialects_normalize(registry, dialect, path, response):
    rec = registry.create(f"rr-{dialect}", "rerank_api", {"dialect": dialect, "base_url": "https://rr.example", "model": "m"})
    respx.post(f"https://rr.example/{path}").mock(return_value=httpx.Response(200, json=response))
    r = await registry.get(rec.id).rerank(RerankRequest(query="kv cache", candidates=CANDS, top_n=2))
    assert [x.id for x in r.results] == ["a", "c"]
    assert [x.rank for x in r.results] == [1, 2]


@respx.mock
async def test_embedding_api_dialects(registry):
    tei = registry.create("tei", "embedding_api", {"dialect": "tei", "base_url": "http://localhost:8080", "model": "bge"})
    respx.post("http://localhost:8080/embed").mock(return_value=httpx.Response(200, json=[[1.0, 0.0], [0.0, 2.0]]))
    r = await registry.get(tei.id).embed(EmbeddingRequest(texts=["x", "y"]))
    assert r.vectors == [[1.0, 0.0], [0.0, 1.0]]
    co = registry.create("co", "embedding_api", {"dialect": "cohere", "base_url": "https://api.cohere.com", "model": "embed-v4.0"}, {"api_key": "k123456"})
    route = respx.post("https://api.cohere.com/v2/embed").mock(return_value=httpx.Response(200, json={"embeddings": {"float": [[0.0, 3.0]]}}))
    r = await registry.get(co.id).embed(EmbeddingRequest(texts=["q"], input_type="query"))
    assert r.vectors == [[0.0, 1.0]]
    assert json.loads(route.calls[0].request.content)["input_type"] == "search_query"


@respx.mock
async def test_search_engines_normalize(registry):
    t = registry.create("tav", "tavily", {}, {"api_key": "tvly-abcdefghij"})
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json={"results": [{"url": "https://a.org/x", "title": "A", "content": "snip", "score": 0.9}]}))
    b = registry.create("brv", "brave", {}, {"api_key": "brave-key-123"})
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(return_value=httpx.Response(200, json={"web": {"results": [{"url": "https://a.org/x", "title": "A", "description": "snip"}]}}))
    e = registry.create("exa", "exa", {}, {"api_key": "exa-key-123"})
    respx.post("https://api.exa.ai/search").mock(return_value=httpx.Response(200, json={"results": [{"url": "https://a.org/x", "title": "A", "text": "snip"}]}))
    s = registry.create("sx", "searxng", {"base_url": "http://localhost:8888"})
    respx.get("http://localhost:8888/search").mock(return_value=httpx.Response(200, json={"results": [{"url": "https://a.org/x", "title": "A", "content": "snip", "engine": "ddg"}]}))
    for pid in (t.id, b.id, e.id, s.id):
        r = await registry.get(pid).search(WebSearchRequest(query="q", max_results=3, recency_days=30, include_domains=["a.org"]))
        assert [(x.url, x.title, x.snippet) for x in r.results] == [("https://a.org/x", "A", "snip")], pid


@respx.mock
async def test_http_errors_are_normalized_and_redirects_refused(registry):
    rec = registry.create("x", "openai_compatible", {"base_url": "https://x.example/v1", "chat_model": "m"}, {"api_key": "sk-supersecretvalue"})
    p = registry.get(rec.id)
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    cases = [
        (httpx.Response(401, json={"error": "bad key sk-supersecretvalue"}), ProviderErrorKind.AUTH),
        (httpx.Response(429, headers={"retry-after": "2"}), ProviderErrorKind.RATE_LIMIT),
        (httpx.Response(503, text="down"), ProviderErrorKind.UNAVAILABLE),
        (httpx.Response(400, text="bad sk-supersecretvalue"), ProviderErrorKind.BAD_REQUEST),
        (httpx.Response(302, headers={"location": "https://evil.example/steal"}), ProviderErrorKind.BAD_REQUEST),
        (httpx.Response(200, text="not json"), ProviderErrorKind.INVALID_RESPONSE),
    ]
    route = respx.post("https://x.example/v1/chat/completions")
    for resp, kind in cases:
        route.mock(return_value=resp)
        with pytest.raises(ProviderError) as ei:
            await p.chat(req)
        assert ei.value.kind == kind
        assert "sk-supersecretvalue" not in str(ei.value.message)
    assert not respx.routes or all("evil.example" not in str(c.request.url) for c in respx.calls)


async def test_blocked_destinations(registry):
    rec = registry.create("meta", "openai_compatible", {"base_url": "http://169.254.169.254/v1", "chat_model": "m"})
    with pytest.raises(ProviderError) as ei:
        await registry.get(rec.id).chat(ChatRequest(messages=[ChatMessage(role="user", content="x")]))
    assert ei.value.kind == ProviderErrorKind.BLOCKED
    with pytest.raises(ValueError):
        registry.create("bad", "openai_compatible", {"base_url": "file:///etc/passwd", "chat_model": "m"})
    with pytest.raises(ValueError):
        registry.create("bad2", "openai_compatible", {"base_url": "https://user:pw@x.example/v1"})
