"""Local model management: HF inspection, guards and routing (offline; real downloads are exercised manually)."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config.settings import Settings
from app.core.container import Container
from app.core.netguard import OutboundPolicy
from app.providers.base import Task
from app.services.retrieval.vector_store import VectorStore
from tests import fakes


@pytest.fixture
async def c(tmp_path):
    cont = Container(Settings(data_dir=tmp_path), extra_adapters=fakes.FAKE_ADAPTERS, policy=OutboundPolicy(resolve_dns=False), vector_store=VectorStore(), checkpoints=False)
    cont.db.kv_set("bootstrapped", True)
    yield cont
    await cont.stop()


@respx.mock
async def test_hf_spec_reads_onnx_dims_and_pooling(c):
    repo = "acme/tiny-embed"
    respx.get(f"https://huggingface.co/api/models/{repo}").mock(return_value=httpx.Response(200, json={
        "pipeline_tag": "sentence-similarity", "siblings": [{"rfilename": "onnx/model.onnx", "size": 90_000_000}, {"rfilename": "config.json"}]}))
    respx.get(f"https://huggingface.co/{repo}/resolve/main/config.json").mock(return_value=httpx.Response(200, json={"hidden_size": 384}))
    respx.get(f"https://huggingface.co/{repo}/resolve/main/1_Pooling/config.json").mock(return_value=httpx.Response(200, json={"pooling_mode_cls_token": True}))
    spec = await c.local_models._hf_spec("embedding", repo)
    assert spec == {"repo": repo, "model_file": "onnx/model.onnx", "size_gb": 0.09, "description": f"sentence-similarity from Hugging Face ({repo})", "dim": 384, "pooling": "CLS"}

    respx.get("https://huggingface.co/api/models/acme/pytorch-only").mock(return_value=httpx.Response(200, json={"siblings": [{"rfilename": "pytorch_model.bin"}]}))
    with pytest.raises(ValueError, match="no ONNX weights"):
        await c.local_models._hf_spec("embedding", "acme/pytorch-only")


async def test_guards(c):
    lm = c.local_models
    with pytest.raises(ValueError, match="Install the model"):
        lm.use("embedding", "BAAI/bge-small-en-v1.5")
    with pytest.raises(ValueError, match="detector \\+ recognizer"):
        await lm.install("ocr", "../../etc/passwd")
    with pytest.raises(ValueError, match="kind"):
        await lm.install("audio", "x/y")


async def test_use_routes_installed_model_first(c):
    lm = c.local_models
    c.db.kv_set("local_models_installed", {"reranking": ["Xenova/ms-marco-MiniLM-L-6-v2"]})
    c.registry.create("cloud", "fake_chat", {"label": "x"}, provider_id="cloud-x")  # unrelated provider stays
    out = lm.use("reranking", "Xenova/ms-marco-MiniLM-L-6-v2")
    route = c.registry.get_route(Task.RERANKING).provider_ids
    assert route[0] == out["provider_id"] == "local-rr-xenova-ms-marco-minilm-l-6-v2"
    assert c.registry.get(route[0]).config.cache_dir.endswith("fastembed")
    cat = lm.catalog()
    item = next(m for m in cat["kinds"]["reranking"] if m["id"] == "Xenova/ms-marco-MiniLM-L-6-v2")
    assert item["installed"] and item["in_use"] and cat["kinds"]["reranking"][0]["id"] == item["id"]
    with pytest.raises(ValueError, match="in use"):
        lm.remove("reranking", "Xenova/ms-marco-MiniLM-L-6-v2")
