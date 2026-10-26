"""ProviderRouter: selects adapters per task and applies timeouts, retries, circuit breaking,
concurrency limits, capability-safe fallback and metrics.

This is the only entry point application code uses to reach an AI capability.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

from app.observability.logging import get_logger, log_event
from app.observability.metrics import CallRecord, MetricsRecorder
from app.providers.base import (
    TASK_CAPABILITY,
    BaseProvider,
    Capability,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageEmbeddingRequest,
    ImageInput,
    NoProviderAvailableError,
    OCRRequest,
    OCRResponse,
    ProviderError,
    ProviderErrorKind,
    RerankCandidate,
    RerankRequest,
    RerankResponse,
    Task,
    VisionRequest,
    VisionResponse,
    WebExtractionRequest,
    WebPage,
    WebSearchRequest,
    WebSearchResponse,
    fingerprint_hash,
)
from app.providers.registry import ProviderRegistry, RouteConfig
from app.providers.resilience import CircuitBreaker, backoff_delay

T = TypeVar("T")
log = get_logger("omni.router")


class ProviderRouter:
    def __init__(
        self,
        registry: ProviderRegistry,
        metrics: MetricsRecorder,
        *,
        backoff_base_s: float = 0.5,
        breaker_threshold: int = 5,
        breaker_reset_s: float = 30.0,
    ):
        self.registry = registry
        self.metrics = metrics
        self.backoff_base_s = backoff_base_s
        self.breaker_threshold = breaker_threshold
        self.breaker_reset_s = breaker_reset_s
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}
        self._semaphores: dict[str, tuple[int, asyncio.Semaphore]] = {}

    # ---------------------------------------------------------------- state
    def breaker(self, provider_id: str, capability: Capability) -> CircuitBreaker:
        key = (provider_id, capability.value)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(self.breaker_threshold, self.breaker_reset_s)
        return self._breakers[key]

    def breaker_states(self) -> dict[str, dict[str, Any]]:
        return {f"{p}:{c}": b.snapshot() for (p, c), b in self._breakers.items()}

    def _semaphore(self, provider: BaseProvider) -> asyncio.Semaphore:
        limit = provider.config.max_concurrency
        cur = self._semaphores.get(provider.provider_id)
        if cur is None or cur[0] != limit:
            cur = (limit, asyncio.Semaphore(limit))
            self._semaphores[provider.provider_id] = cur
        return cur[1]

    def route(self, task: Task) -> RouteConfig:
        """Chat tasks with no providers of their own inherit the synthesis route (the 'default LLM')."""
        r = self.registry.get_route(task)
        if not r.provider_ids and TASK_CAPABILITY[task] == Capability.CHAT and task != Task.SYNTHESIS:
            base = self.registry.get_route(Task.SYNTHESIS)
            return RouteConfig(task=task, provider_ids=base.provider_ids, fallback_enabled=base.fallback_enabled, params={})
        return r

    def inherits(self, task: Task) -> bool:
        return TASK_CAPABILITY[task] == Capability.CHAT and task != Task.SYNTHESIS and not self.registry.get_route(task).provider_ids

    def is_configured(self, task: Task) -> bool:
        return any(self.registry.is_enabled(p) for p in self.route(task).provider_ids)

    def primary(self, task: Task) -> BaseProvider | None:
        for pid in self.route(task).provider_ids:
            if self.registry.is_enabled(pid):
                return self.registry.get(pid)
        return None

    # ---------------------------------------------------------------- core execution
    async def execute(
        self,
        task: Task,
        fn: Callable[[BaseProvider], Awaitable[T]],
        *,
        provider_ids: list[str] | None = None,
        allow_fallback: bool | None = None,
        candidate_filter: Callable[[BaseProvider], bool] | None = None,
    ) -> T:
        capability = TASK_CAPABILITY[task]
        route = self.route(task)
        ids = provider_ids if provider_ids is not None else route.provider_ids
        fallback = route.fallback_enabled if allow_fallback is None else allow_fallback
        if not ids:
            raise NoProviderAvailableError(task.value)

        attempts: list[dict[str, Any]] = []
        first_id: str | None = None
        for pos, pid in enumerate(ids):
            if pos > 0 and not fallback:
                break
            try:
                provider = self.registry.get(pid)
            except ProviderError as exc:
                attempts.append({"provider_id": pid, **exc.to_dict()})
                continue
            # Capability check at call time: never substitute a provider that lacks the capability.
            if not provider.supports(capability):
                attempts.append({"provider_id": pid, "kind": "unsupported", "message": f"does not support {capability.value}"})
                continue
            if candidate_filter and not candidate_filter(provider):
                attempts.append({"provider_id": pid, "kind": "incompatible", "message": "not compatible with the primary provider for this task"})
                continue
            breaker = self.breaker(pid, capability)
            if not breaker.allow():
                attempts.append({"provider_id": pid, "kind": ProviderErrorKind.CIRCUIT_OPEN.value, "message": "circuit open"})
                continue
            if first_id is None:
                first_id = pid
            try:
                result = await self._call_with_retries(task, capability, provider, fn, fallback_from=first_id if pid != first_id else None)
                breaker.record_success()
                if pid != first_id:
                    log_event(log, "provider fallback used", task=task.value, provider_id=pid, fallback_from=first_id)
                return result
            except ProviderError as exc:
                if exc.retryable:
                    breaker.record_failure(exc.message)
                else:
                    breaker._half_open_in_flight = False  # non-transient errors do not open the circuit
                attempts.append({"provider_id": pid, **exc.to_dict()})
                log_event(log, "provider call failed", task=task.value, provider_id=pid, kind=exc.kind.value, error=exc.message)
        raise NoProviderAvailableError(task.value, attempts)

    async def _call_with_retries(
        self, task: Task, capability: Capability, provider: BaseProvider, fn: Callable[[BaseProvider], Awaitable[T]], fallback_from: str | None
    ) -> T:
        max_attempts = provider.config.max_retries + 1
        sem = self._semaphore(provider)
        last: ProviderError | None = None
        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            try:
                async with sem:
                    result = await asyncio.wait_for(fn(provider), timeout=provider.config.timeout_s * 1.5 + 5)
            except asyncio.TimeoutError:
                err: ProviderError = ProviderError(ProviderErrorKind.TIMEOUT, "Call exceeded router timeout", provider_id=provider.provider_id)
            except ProviderError as exc:
                exc.provider_id = exc.provider_id or provider.provider_id
                err = exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # adapter bug or library error: normalize, never leak details
                err = ProviderError(ProviderErrorKind.INVALID_RESPONSE, f"Adapter error: {type(exc).__name__}: {str(exc)[:200]}", provider_id=provider.provider_id)
            else:
                usage = getattr(result, "usage", None)
                self.metrics.record(
                    CallRecord(
                        provider_id=provider.provider_id,
                        task=task.value,
                        capability=capability.value,
                        ok=True,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        model=getattr(result, "model", None) or provider.model_for(capability),
                        input_tokens=getattr(usage, "input_tokens", None),
                        output_tokens=getattr(usage, "output_tokens", None),
                        cost_usd=provider.estimate_cost(usage) if usage is not None else None,
                        fallback_from=fallback_from,
                        attempt=attempt + 1,
                    )
                )
                return result
            self.metrics.record(
                CallRecord(
                    provider_id=provider.provider_id,
                    task=task.value,
                    capability=capability.value,
                    ok=False,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    model=provider.model_for(capability),
                    error_kind=err.kind.value,
                    error_message=err.message,
                    fallback_from=fallback_from,
                    attempt=attempt + 1,
                )
            )
            last = err
            if not err.retryable or attempt == max_attempts - 1:
                break
            await asyncio.sleep(backoff_delay(attempt, base=self.backoff_base_s, retry_after=err.retry_after))
        assert last is not None
        raise last

    # ---------------------------------------------------------------- typed convenience API
    def _chat_params(self, task: Task, request: ChatRequest) -> ChatRequest:
        params = self.route(task).params
        updates: dict[str, Any] = {}
        if params.get("model") and not request.model:
            updates["model"] = params["model"]
        if "temperature" in params:
            updates["temperature"] = params["temperature"]
        if params.get("max_tokens"):
            updates["max_tokens"] = params["max_tokens"]
        return request.model_copy(update=updates) if updates else request

    async def chat(self, task: Task, request: ChatRequest) -> ChatResponse:
        if TASK_CAPABILITY[task] != Capability.CHAT:
            raise ValueError(f"Task {task.value} is not a chat task")
        request = self._chat_params(task, request)
        primary = self.route(task).provider_ids[:1]

        async def call(p: BaseProvider) -> ChatResponse:
            # A model override set on the route belongs to the preferred provider only.
            req = request if (p.provider_id in primary) else request.model_copy(update={"model": None})
            return await p.chat(req)  # type: ignore[attr-defined]

        return await self.execute(task, call)

    async def embed(
        self, texts: list[str], *, input_type: str = "document", provider_id: str | None = None, task: Task = Task.TEXT_EMBEDDING
    ) -> EmbeddingResponse:
        """Embed texts. When ``provider_id`` is given (an index's pinned provider), fallback is restricted
        to providers that produce the *identical* embedding space (same fingerprint)."""
        req = EmbeddingRequest(texts=texts, input_type=input_type)  # type: ignore[arg-type]
        route = self.route(task)
        ids = [provider_id] + [p for p in route.provider_ids if p != provider_id] if provider_id else route.provider_ids
        if not ids:
            raise NoProviderAvailableError(task.value)
        primary_fp = self.embedding_fingerprint(ids[0])

        def compatible(p: BaseProvider) -> bool:
            try:
                return primary_fp is not None and fingerprint_hash(self._fp_of(p)) == fingerprint_hash(primary_fp)
            except ProviderError:
                return False

        return await self.execute(task, lambda p: p.embed(req), provider_ids=ids, candidate_filter=compatible)  # type: ignore[attr-defined]

    @staticmethod
    def _fp_of(p: BaseProvider) -> dict[str, Any]:
        fp = dict(p.embedding_fingerprint())
        key = getattr(p.config, "compatibility_key", None)
        return {"compatibility_key": key} if key else fp

    def embedding_fingerprint(self, provider_id: str) -> dict[str, Any] | None:
        try:
            return self._fp_of(self.registry.get(provider_id))
        except ProviderError:
            return None

    async def embed_images(self, images: list[ImageInput], texts: list[str] | None = None, *, provider_id: str | None = None) -> EmbeddingResponse:
        req = ImageEmbeddingRequest(images=images, texts=texts or [])
        ids = [provider_id] if provider_id else None
        return await self.execute(Task.IMAGE_EMBEDDING, lambda p: p.embed_images(req), provider_ids=ids, allow_fallback=False if provider_id else None)  # type: ignore[attr-defined]

    async def rerank(self, query: str, candidates: list[RerankCandidate], top_n: int | None = None) -> RerankResponse:
        req = RerankRequest(query=query, candidates=candidates, top_n=top_n)
        return await self.execute(Task.RERANKING, lambda p: p.rerank(req))  # type: ignore[attr-defined]

    async def ocr(self, image: ImageInput, language: str | None = None, *, provider_ids: list[str] | None = None, allow_fallback: bool | None = None) -> OCRResponse:
        req = OCRRequest(image=image, language=language)
        return await self.execute(Task.OCR, lambda p: p.ocr(req), provider_ids=provider_ids, allow_fallback=allow_fallback)  # type: ignore[attr-defined]

    async def vision(self, request: VisionRequest) -> VisionResponse:
        params = self.route(Task.VISION).params
        if params.get("model") and not request.model:
            request = request.model_copy(update={"model": params["model"]})
        primary = self.route(Task.VISION).provider_ids[:1]
        return await self.execute(
            Task.VISION, lambda p: p.analyze(request if p.provider_id in primary else request.model_copy(update={"model": None}))  # type: ignore[attr-defined]
        )

    async def web_search(self, request: WebSearchRequest) -> WebSearchResponse:
        return await self.execute(Task.WEB_SEARCH, lambda p: p.search(request))  # type: ignore[attr-defined]

    async def extract_webpage(self, url: str, max_chars: int = 60_000) -> WebPage:
        req = WebExtractionRequest(url=url, max_chars=max_chars)
        return await self.execute(Task.WEB_EXTRACTION, lambda p: p.extract(req))  # type: ignore[attr-defined]
