"""Settings → AI Providers: adapters, configurations, credentials, routing, health."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import container, require_admin
from app.core.container import Container
from app.providers.base import TASK_CAPABILITY, TASK_DESCRIPTIONS, ProviderError, Task
from app.providers.registry import ConfigConflictError, RouteConfig

router = APIRouter(prefix="/api", tags=["providers"])


class ProviderIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    adapter: str
    config: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, str | None] = Field(default_factory=dict, description="Write-only. null keeps, '' deletes.")
    enabled: bool = True
    id: str | None = None


class ProviderPatch(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    secrets: dict[str, str | None] | None = None
    enabled: bool | None = None


class DraftTest(BaseModel):
    adapter: str
    config: dict[str, Any]
    secrets: dict[str, str] = Field(default_factory=dict, description='Omitted = use stored value (with provider_id); "" = treat as cleared')
    provider_id: str | None = Field(default=None, description="Reuse stored secrets of this provider when a secret is omitted")


@router.get("/providers/catalog")
def catalog(c: Container = Depends(container)):
    return [d.model_dump() for d in c.registry.catalog()]


@router.get("/providers", dependencies=[Depends(require_admin)])
def list_providers(c: Container = Depends(container)):
    return [r.model_dump() for r in c.registry.list_records()]


@router.post("/providers", dependencies=[Depends(require_admin)], status_code=201)
def create_provider(body: ProviderIn, c: Container = Depends(container)):
    return c.registry.create(body.name, body.adapter, body.config, body.secrets, provider_id=body.id, enabled=body.enabled).model_dump()


@router.get("/providers/{provider_id}", dependencies=[Depends(require_admin)])
def get_provider(provider_id: str, c: Container = Depends(container)):
    return c.registry.get_record(provider_id).model_dump()


@router.patch("/providers/{provider_id}", dependencies=[Depends(require_admin)])
def update_provider(provider_id: str, body: ProviderPatch, c: Container = Depends(container)):
    return c.registry.update(provider_id, name=body.name, config=body.config, secrets=body.secrets, enabled=body.enabled).model_dump()


@router.delete("/providers/{provider_id}", dependencies=[Depends(require_admin)], status_code=204)
def delete_provider(provider_id: str, force: bool = False, c: Container = Depends(container)):
    c.registry.delete(provider_id, force=force)


async def _health(provider) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        r = await provider.health_check()
        return {"ok": r.ok, "latency_ms": r.latency_ms if r.latency_ms is not None else round((time.perf_counter() - t0) * 1000, 1), "detail": r.detail, "checked": r.checked}
    except ProviderError as exc:
        return {"ok": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "error": exc.to_dict()}
    except Exception as exc:
        return {"ok": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "error": {"kind": "adapter_error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"}}


@router.post("/providers/{provider_id}/test", dependencies=[Depends(require_admin)])
async def test_provider(provider_id: str, c: Container = Depends(container)):
    if not c.registry.is_enabled(provider_id):
        raise HTTPException(400, "Provider is disabled")
    return await _health(c.registry.get(provider_id))


@router.post("/providers/test-draft", dependencies=[Depends(require_admin)])
async def test_draft(body: DraftTest, c: Container = Depends(container)):
    secrets = dict(body.secrets)
    if body.provider_id:
        stored = c.registry.get(body.provider_id).secrets if c.registry.is_enabled(body.provider_id) else {}
        for k, v in stored.items():
            secrets.setdefault(k, v)  # an explicit "" in the draft means "cleared" and is kept
    secrets = {k: v for k, v in secrets.items() if v}
    provider = c.registry.build_unsaved(body.adapter, body.config, secrets)
    try:
        return await _health(provider)
    finally:
        await provider.aclose()


@router.get("/providers/{provider_id}/failures", dependencies=[Depends(require_admin)])
def provider_failures(provider_id: str, c: Container = Depends(container)):
    return c.metrics.recent_failures(provider_id)


@router.get("/provider-health", dependencies=[Depends(require_admin)])
def provider_health(c: Container = Depends(container)):
    return {"metrics": c.metrics.summary(), "circuits": c.router.breaker_states(), "recent_failures": c.metrics.recent_failures(limit=25)}


class QuickDiscoverIn(BaseModel):
    preset: str
    api_key: str | None = Field(default=None, description="Write-only; omitted when re-discovering a preset already connected")
    base_url: str | None = None


class QuickApplyIn(QuickDiscoverIn):
    assignments: dict[str, str | None] = Field(default_factory=dict, description="synthesis, planning, vision, fallback, embedding → model id")


def _stored_key(c: Container, preset: str) -> str | None:
    """Reuse the key of an already-saved provider for the same service, so users never paste it twice."""
    pid = c.quick.connected_provider(preset)
    return c.registry.get(pid).secrets.get("api_key") if pid else None


class LocalModelIn(BaseModel):
    kind: str
    model: str = Field(min_length=3, max_length=200)
    repo: str | None = Field(default=None, max_length=200, description="Hugging Face repo id for models not in the built-in catalog")


@router.get("/local-models", dependencies=[Depends(require_admin)])
def local_models(c: Container = Depends(container)):
    return c.local_models.catalog()


@router.get("/local-models/search", dependencies=[Depends(require_admin)])
async def local_models_search(kind: str, q: str = "", c: Container = Depends(container)):
    return await c.local_models.search_hf(kind, q)


@router.post("/local-models/install", dependencies=[Depends(require_admin)])
async def local_models_install(body: LocalModelIn, c: Container = Depends(container)):
    return await c.local_models.install(body.kind, body.model, body.repo)


@router.post("/local-models/use", dependencies=[Depends(require_admin)])
async def local_models_use(body: LocalModelIn, c: Container = Depends(container)):
    return c.local_models.use(body.kind, body.model)


@router.post("/local-models/remove", dependencies=[Depends(require_admin)])
def local_models_remove(body: LocalModelIn, c: Container = Depends(container)):
    c.local_models.remove(body.kind, body.model)
    return {"removed": body.model}


@router.get("/quick-connect/presets")
def quick_presets(c: Container = Depends(container)):
    out = c.quick.presets()
    for p in out:
        p["connected"] = c.quick.connected_provider(p["id"]) is not None
    return out


@router.post("/quick-connect/discover", dependencies=[Depends(require_admin)])
async def quick_discover(body: QuickDiscoverIn, c: Container = Depends(container)):
    return await c.quick.discover(body.preset, body.api_key or _stored_key(c, body.preset), body.base_url)


@router.post("/quick-connect/discover/stream", dependencies=[Depends(require_admin)])
async def quick_discover_stream(body: QuickDiscoverIn, c: Container = Depends(container)):
    """Same as /discover, streamed as NDJSON events (models, probe…, done | error) so results show up one by one.
    Closing the connection cancels the remaining model tests."""
    import json

    from fastapi.responses import StreamingResponse

    key = body.api_key or _stored_key(c, body.preset)

    async def events():
        try:
            async for ev in c.quick.discover_events(body.preset, key, body.base_url):
                yield json.dumps(ev, default=str) + "\n"
        except ValueError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.post("/quick-connect/apply", dependencies=[Depends(require_admin)])
async def quick_apply(body: QuickApplyIn, c: Container = Depends(container)):
    out = c.quick.apply(body.preset, body.api_key or _stored_key(c, body.preset), body.assignments, body.base_url)
    emb = out.get("embedding_provider")
    active = c.index.active()
    if emb and (active is None or active.spec.get("provider_id") != emb):
        # The user chose a new embedder: rebuild the search index with it right away (the old one stays for rollback).
        try:
            c.index.start_reindex(emb)
            out["notes"].append("Rebuilding the document search index with the new embedding model. Search keeps using the "
                                "current index until the new one is ready; you can roll back under Document search index.")
            out["reindex_started"] = True
        except ValueError as exc:
            out["notes"].append(f"Could not start rebuilding the search index: {exc}")
    return out


@router.get("/routes", dependencies=[Depends(require_admin)])
def list_routes(c: Container = Depends(container)):
    out = []
    for t in Task:
        r = c.registry.get_route(t)
        out.append(
            {
                **r.model_dump(),
                "capability": TASK_CAPABILITY[t].value,
                "description": TASK_DESCRIPTIONS[t],
                "inherits_from": "synthesis" if c.router.inherits(t) else None,
                "effective_provider_ids": c.router.route(t).provider_ids,
            }
        )
    return out


@router.put("/routes/{task}", dependencies=[Depends(require_admin)])
def set_route(task: Task, body: RouteConfig, c: Container = Depends(container)):
    if body.task != task:
        raise HTTPException(400, "task mismatch")
    return c.registry.set_route(body).model_dump()


@router.get("/privacy/destinations")
def destinations(c: Container = Depends(container)):
    """Providers that would receive document content (shown as a warning before research/ingestion)."""
    return c.research.external_destinations({})


def install_error_handlers(app) -> None:
    from fastapi.responses import JSONResponse

    from app.providers.base import NoProviderAvailableError
    from app.services.ingestion.service import UploadRejected

    @app.exception_handler(KeyError)
    async def _nf(_, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": f"Not found: {exc.args[0] if exc.args else ''}"})

    @app.exception_handler(ConfigConflictError)
    async def _conflict(_, exc: ConfigConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc), "references": exc.references})

    @app.exception_handler(UploadRejected)
    async def _upload(_, exc: UploadRejected):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(NoProviderAvailableError)
    async def _noprov(_, exc: NoProviderAvailableError):
        return JSONResponse(status_code=503, content={"detail": exc.message, "error": exc.to_dict()})

    @app.exception_handler(ProviderError)
    async def _prov(_, exc: ProviderError):
        return JSONResponse(status_code=502, content={"detail": exc.message, "error": exc.to_dict()})

    @app.exception_handler(ValueError)
    async def _bad(_, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})
