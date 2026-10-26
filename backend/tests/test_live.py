"""Optional live-provider tests. Skipped unless OMNI_LIVE_TESTS=1 and the relevant variables are set.

  OMNI_LIVE_TESTS=1 OMNI_LIVE_CHAT_BASE_URL=https://integrate.api.nvidia.com/v1 \
  OMNI_LIVE_CHAT_MODEL=meta/llama-3.3-70b-instruct OMNI_LIVE_CHAT_KEY=nvapi-... pytest -m live
"""

from __future__ import annotations

import os

import pytest

from app.core.netguard import OutboundPolicy
from app.providers.base import ChatMessage, ChatRequest, EmbeddingRequest, WebSearchRequest
from app.providers.http import ProviderHttp

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.environ.get("OMNI_LIVE_TESTS") != "1", reason="live tests disabled")]


def need(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"missing {missing}")
    return {n: os.environ[n] for n in names}


@pytest.fixture
async def live_registry(db, secret_box):
    from app.providers.registry import ProviderRegistry

    http = ProviderHttp(OutboundPolicy())  # real DNS + SSRF checks
    yield ProviderRegistry(db, secret_box, http)
    await http.aclose()


async def test_live_openai_compatible_chat(live_registry):
    env = need("OMNI_LIVE_CHAT_BASE_URL", "OMNI_LIVE_CHAT_MODEL", "OMNI_LIVE_CHAT_KEY")
    rec = live_registry.create("live", "openai_compatible", {"base_url": env["OMNI_LIVE_CHAT_BASE_URL"], "chat_model": env["OMNI_LIVE_CHAT_MODEL"]}, {"api_key": env["OMNI_LIVE_CHAT_KEY"]})
    r = await live_registry.get(rec.id).chat(ChatRequest(messages=[ChatMessage(role="user", content="Reply with the single word: ready")], max_tokens=10))
    assert r.text.strip()


async def test_live_openai_compatible_embeddings(live_registry):
    env = need("OMNI_LIVE_EMBED_BASE_URL", "OMNI_LIVE_EMBED_MODEL", "OMNI_LIVE_EMBED_KEY")
    cfg = {"base_url": env["OMNI_LIVE_EMBED_BASE_URL"], "embedding_model": env["OMNI_LIVE_EMBED_MODEL"]}
    if os.environ.get("OMNI_LIVE_EMBED_INPUT_TYPE_FIELD"):
        cfg["embedding_input_type_field"] = os.environ["OMNI_LIVE_EMBED_INPUT_TYPE_FIELD"]
    rec = live_registry.create("live-emb", "openai_compatible", cfg, {"api_key": env["OMNI_LIVE_EMBED_KEY"]})
    r = await live_registry.get(rec.id).embed(EmbeddingRequest(texts=["kv cache paging", "weight quantization"]))
    assert len(r.vectors) == 2 and r.dims > 0


async def test_live_tavily(live_registry):
    env = need("OMNI_LIVE_TAVILY_KEY")
    rec = live_registry.create("tav", "tavily", {}, {"api_key": env["OMNI_LIVE_TAVILY_KEY"]})
    r = await live_registry.get(rec.id).search(WebSearchRequest(query="PagedAttention vLLM paper", max_results=3))
    assert r.results and all(x.url.startswith("http") for x in r.results)


async def test_live_builtin_extractor(live_registry):
    need("OMNI_LIVE_WEB")
    rec = live_registry.create("web", "builtin_web_extractor", {})
    h = await live_registry.get(rec.id).health_check()
    assert h.ok
