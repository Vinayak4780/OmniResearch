"""Quick connect: provider presets, model discovery and one-click routing.

The user picks a preset and pastes an API key. We list the provider's models, classify them by
capability, verify that the best candidates actually respond (many catalogs list models an account
cannot use) and that chat candidates produce valid structured output, then create providers and
routes. Everything still goes through the normal registry/adapters; presets only fill in config.
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.providers.base import (
    Capability,
    ChatMessage,
    ChatRequest,
    ProviderError,
    Task,
    WebSearchRequest,
)
from app.providers.http import join_url
from app.providers.registry import ProviderRegistry, RouteConfig
from app.services.research.llm import extract_json


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    kind: str  # "llm" | "search"
    adapter: str
    base_url: str = ""
    key_url: str = ""  # where users get a key
    key_hint: str = ""
    needs_key: bool = True
    list_style: str = "openai"  # openai | anthropic | none
    base_config: dict[str, Any] = field(default_factory=dict)
    embedding_config: dict[str, Any] = field(default_factory=dict)
    note: str = ""


PRESETS: list[Preset] = [
    Preset("nvidia", "NVIDIA NIM", "llm", "openai_compatible", "https://integrate.api.nvidia.com/v1", "https://build.nvidia.com", "nvapi-…",
           embedding_config={"embedding_input_type_field": "input_type"},
           note="The catalog lists many models that are not enabled for every account; unavailable ones are filtered out."),
    Preset("openrouter", "OpenRouter", "llm", "openai_compatible", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys", "sk-or-…",
           base_config={"extra_headers": {"X-Title": "OmniResearch"}}),
    Preset("openai", "OpenAI", "llm", "openai_compatible", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "sk-…",
           base_config={"supports_json_mode": True}),
    Preset("anthropic", "Anthropic (Claude)", "llm", "anthropic", "https://api.anthropic.com", "https://console.anthropic.com/settings/keys", "sk-ant-…",
           list_style="anthropic"),
    Preset("groq", "Groq", "llm", "openai_compatible", "https://api.groq.com/openai/v1", "https://console.groq.com/keys", "gsk_…"),
    Preset("together", "Together AI", "llm", "openai_compatible", "https://api.together.xyz/v1", "https://api.together.ai/settings/api-keys", ""),
    Preset("deepseek", "DeepSeek", "llm", "openai_compatible", "https://api.deepseek.com/v1", "https://platform.deepseek.com/api_keys", "sk-…"),
    Preset("mistral", "Mistral AI", "llm", "openai_compatible", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys", ""),
    Preset("modelscope", "ModelScope API-Inference", "llm", "openai_compatible", "https://api-inference.modelscope.cn/v1", "https://modelscope.cn/my/myaccesstoken", "ms-…"),
    Preset("ollama", "Ollama (local)", "llm", "openai_compatible", "http://localhost:11434/v1", "", "", needs_key=False,
           note="Runs on this machine; no key needed. Start Ollama and pull a model first."),
    Preset("custom", "Custom endpoint", "llm", "openai_compatible", "", "", "", needs_key=False,
           note="Any OpenAI-compatible server (vLLM, SGLang, LM Studio, LiteLLM, a company gateway). Enter its address; the key is optional."),
    Preset("tavily", "Tavily (web search)", "search", "tavily", key_url="https://app.tavily.com", key_hint="tvly-…", list_style="none"),
    Preset("brave", "Brave Search", "search", "brave", key_url="https://api-dashboard.search.brave.com", list_style="none"),
    Preset("exa", "Exa (web search)", "search", "exa", key_url="https://dashboard.exa.ai/api-keys", list_style="none"),
]
PRESET_BY_ID = {p.id: p for p in PRESETS}

_EXCLUDE = re.compile(
    r"guard|safety|reward|parse|translat|ocdr|clip(?!-)|tts|whisper|audio|speech|transcri|dall-e|flux|stable-diffusion|sdxl|diffusion"
    r"|moderation|detector|calibration|image-gen|imagen|realtime|search-preview|computer-use|codegemma|starcoder|deplot|kosmos|fuyu",
    re.I,
)
_EMBED = re.compile(r"embed|bge-|\be5-|gte-|arctic-embed|nomic-embed|text-embedding", re.I)
_RERANK = re.compile(r"rerank|ranking", re.I)
_VISION = re.compile(r"vision|-vl\b|-vl-|vl-|vlm|omni|llava|pixtral|gpt-4o|gpt-4\.1|gpt-5|claude|gemini|qwen2\.5-vl|qwen3-vl|llama-4|phi-4-multimodal", re.I)
# Name fragments of models that have worked well for research synthesis (checked again by probing).
_PREFERRED = ["nemotron-3-ultra", "claude-opus", "claude-sonnet", "gpt-5", "gpt-4.1", "nemotron-3.5-super", "deepseek-v", "deepseek-chat",
              "qwen3", "llama-3.3-70b", "llama-3.1-405b", "mistral-large", "kimi", "glm", "minimax", "gemini"]


def _size_b(model_id: str) -> float:
    """Largest parameter count mentioned in the id (e.g. '550b', '8x22b', 'a12b' ignored)."""
    best = 0.0
    for m in re.finditer(r"(?<![a-z])(\d+(?:\.\d+)?)b\b", model_id.lower()):
        best = max(best, float(m.group(1)))
    return best


def classify(model_id: str, meta: dict[str, Any] | None = None) -> list[str]:
    meta = meta or {}
    if _EXCLUDE.search(model_id):
        return []
    if _RERANK.search(model_id):
        return ["reranking"]
    if _EMBED.search(model_id) or meta.get("type") == "embedding":
        return ["embedding"]
    caps = ["chat"]
    modalities = ((meta.get("architecture") or {}).get("input_modalities")) or []
    if "image" in modalities or _VISION.search(model_id):
        caps.append("vision")
    return caps


def chat_rank(model_id: str) -> float:
    mid = model_id.lower()
    pref = next((len(_PREFERRED) - i for i, p in enumerate(_PREFERRED) if p in mid), 0)
    penalty = 30 if re.search(r"mini|nano|tiny|small|lite|flash|lightning|bolt|\b[1-9]b\b|e\d+b", mid) else 0
    return pref * 100 + min(_size_b(mid), 1000) / 10 - penalty


PROBE_TIMEOUT_S = 45.0
MAX_EMBED_PROBES = 8


class _Probe(BaseModel):
    ok: bool
    items: list[str]


class QuickConnectService:
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    @staticmethod
    def presets() -> list[dict[str, Any]]:
        return [
            {"id": p.id, "name": p.name, "kind": p.kind, "needs_key": p.needs_key, "key_url": p.key_url, "key_hint": p.key_hint,
             "base_url": p.base_url, "note": p.note}
            for p in PRESETS
        ]

    def connected_provider(self, preset_id: str) -> str | None:
        """An enabled saved provider for this service (same adapter and endpoint) that has a key."""
        p = self._preset(preset_id)
        if p.id == "custom":
            return None
        for rec in self.registry.list_records():
            if not rec.enabled or rec.adapter != p.adapter or (p.needs_key and not rec.secrets_set.get("api_key")):
                continue
            url = str(rec.config.get("base_url") or "").rstrip("/")
            if p.kind == "search" or url == p.base_url.rstrip("/"):
                return rec.id
        return None

    def _preset(self, preset_id: str) -> Preset:
        if preset_id not in PRESET_BY_ID:
            raise ValueError(f"Unknown provider preset '{preset_id}'")
        return PRESET_BY_ID[preset_id]

    def _llm_config(self, p: Preset, base_url: str, **models: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {"timeout_s": 180, "max_retries": 2, **p.base_config, **{k: v for k, v in models.items() if v}}
        if p.adapter == "openai_compatible":
            cfg["base_url"] = base_url
            if models.get("embedding_model"):
                cfg.update(p.embedding_config)
        elif base_url:
            cfg["base_url"] = base_url
        return cfg

    def _secrets(self, p: Preset, api_key: str | None) -> dict[str, str]:
        if p.needs_key and not api_key:
            raise ValueError(f"{p.name} needs an API key")
        return {"api_key": api_key} if api_key else {}

    # ---------------------------------------------------------------- discovery
    async def _list_models(self, p: Preset, base_url: str, api_key: str | None) -> list[tuple[str, dict[str, Any]]]:
        http = self.registry.http
        if p.list_style == "anthropic":
            data = await http.request_json("discover", "GET", join_url(base_url, "v1/models"), params={"limit": 100},
                                           headers={"x-api-key": api_key or "", "anthropic-version": "2023-06-01"}, timeout=30)
            return [(m["id"], m) for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        data = await http.request_json("discover", "GET", join_url(base_url, "models"), headers=headers, timeout=30)
        items = data.get("data", data) if isinstance(data, dict) else data
        return [(m["id"], m) for m in items or [] if isinstance(m, dict) and m.get("id")]

    async def _probe_chat(self, p: Preset, base_url: str, api_key: str | None, model: str) -> dict[str, Any]:
        """Is the model reachable, and does it return valid structured output (needed by the agent)?"""
        prov = self.registry.build_unsaved(p.adapter, self._llm_config(p, base_url, chat_model=model) | {"max_retries": 0, "timeout_s": PROBE_TIMEOUT_S}, self._secrets(p, api_key))
        try:
            resp = await prov.chat(ChatRequest(
                messages=[ChatMessage(role="system", content='Respond with ONE JSON object only: {"ok": true, "items": [three short strings]}'),
                          ChatMessage(role="user", content="List three ways to reduce GPU memory use when serving language models.")],
                max_tokens=4096, temperature=0.0))
            try:
                _Probe.model_validate(extract_json(resp.text))
                return {"available": True, "json_ok": True}
            except Exception:
                return {"available": True, "json_ok": False, "detail": "did not return valid JSON"}
        except ProviderError as exc:
            if exc.kind.value == "timeout":
                return {"available": False, "json_ok": None, "detail": f"no response within {PROBE_TIMEOUT_S:.0f}s"}
            # rate limited: the model exists, but we could not verify it now
            return {"available": True if exc.kind.value == "rate_limit" else False, "json_ok": None, "detail": f"{exc.kind.value}: {exc.message[:120]}"}
        finally:
            await prov.aclose()

    async def _probe_embedding(self, p: Preset, base_url: str, api_key: str | None, model: str) -> dict[str, Any]:
        from app.providers.base import EmbeddingRequest

        prov = self.registry.build_unsaved(p.adapter, self._llm_config(p, base_url, embedding_model=model) | {"max_retries": 0, "timeout_s": 20}, self._secrets(p, api_key))
        try:
            r = await prov.embed(EmbeddingRequest(texts=["probe"], input_type="query"))
            return {"available": True, "dims": r.dims}
        except ProviderError as exc:
            return {"available": False, "detail": f"{exc.kind.value}: {exc.message[:120]}"}
        finally:
            await prov.aclose()

    async def discover(self, preset_id: str, api_key: str | None, base_url: str | None = None, *, max_probes: int = 8) -> dict[str, Any]:
        """Run discovery to completion and return the final result (see :meth:`discover_events`)."""
        final: dict[str, Any] = {}
        async for ev in self.discover_events(preset_id, api_key, base_url, max_probes=max_probes):
            if ev["type"] == "error":
                raise ValueError(ev["detail"])
            if ev["type"] == "done":
                final = ev["result"]
        return final

    async def discover_events(self, preset_id: str, api_key: str | None, base_url: str | None = None, *, max_probes: int = 8):
        """Discovery as a stream of events so the UI can show progress model by model:

        ``models``  - the classified model list, sent as soon as the catalog is fetched (all untested)
        ``probe``   - one model's test result, in completion order, with the updated recommendations
        ``done``    - the final result (same shape as :meth:`discover`)
        ``error``   - a user-facing problem (bad key, unreachable service, nothing usable)
        """
        p = self._preset(preset_id)
        self._secrets(p, api_key)  # fail fast with a clear message when a required key is missing
        base_url = (base_url or p.base_url).rstrip("/")
        if not base_url and p.kind == "llm":
            yield {"type": "error", "detail": "Enter the endpoint address (base URL)"}
            return
        if p.kind == "search":
            prov = self.registry.build_unsaved(p.adapter, {}, self._secrets(p, api_key))
            try:
                r = await prov.search(WebSearchRequest(query="large language model inference", max_results=2))
                res = {"preset": p.id, "kind": "search", "ok": True, "detail": f"{len(r.results)} test results", "models": [], "recommended": {"web_search": p.id}}
            except ProviderError as exc:
                res = {"preset": p.id, "kind": "search", "ok": False, "detail": exc.message, "models": [], "recommended": {}}
            finally:
                await prov.aclose()
            yield {"type": "done", "result": res}
            return

        try:
            listed = await self._list_models(p, base_url, api_key)
        except ProviderError as exc:
            hint = " Check the API key." if exc.kind.value == "auth" else (" Check the address." if exc.kind.value in ("config", "network", "blocked") else "")
            yield {"type": "error", "detail": f"Could not list models from {p.name}: {exc.message}.{hint}"}
            return
        models = []
        for mid, meta in listed:
            caps = classify(mid, meta)
            if caps:
                models.append({"id": mid, "capabilities": caps, "available": None, "json_ok": None, "context_length": meta.get("context_length")})
        if not models:
            yield {"type": "error", "detail": f"{p.name} returned no usable models."}
            return
        chat = sorted([m for m in models if "chat" in m["capabilities"]], key=lambda m: chat_rank(m["id"]), reverse=True)
        vision = [m for m in chat if "vision" in m["capabilities"]]
        # newest embedders usually sort last (higher version numbers), so try those first
        embeds = sorted([m for m in models if "embedding" in m["capabilities"]], key=lambda m: m["id"], reverse=True)
        order = {m["id"]: i for i, m in enumerate(chat)}
        models.sort(key=lambda m: ("chat" not in m["capabilities"], order.get(m["id"], 9999)))
        by_id = {m["id"]: m for m in models}
        chat_probe = list(dict.fromkeys([m["id"] for m in chat[:max_probes]] + [m["id"] for m in vision[:3]]))
        emb_probe = [m["id"] for m in embeds[:MAX_EMBED_PROBES]]
        yield {"type": "models", "preset": p.id, "base_url": base_url, "models": [dict(m) for m in models], "to_test": chat_probe + emb_probe}

        sem = asyncio.Semaphore(8)

        async def run(mid: str, embedding: bool) -> tuple[str, dict[str, Any]]:
            async with sem:
                probe = self._probe_embedding if embedding else self._probe_chat
                return mid, await probe(p, base_url, api_key, mid)

        tasks = [asyncio.create_task(run(m, False)) for m in chat_probe] + [asyncio.create_task(run(m, True)) for m in emb_probe]
        try:
            done = 0
            for fut in asyncio.as_completed(tasks):
                mid, result = await fut
                by_id[mid].update(result)
                done += 1
                yield {"type": "probe", "id": mid, "result": result, "done": done, "total": len(tasks),
                       "recommended": self._recommend(chat, vision, embeds)}
        finally:
            for t in tasks:  # client went away: stop testing (and paying for) the remaining models
                t.cancel()

        rec = self._recommend(chat, vision, embeds)
        if not rec["synthesis"]:
            yield {"type": "error", "detail": f"None of the tested {p.name} models responded. Check the key and that models are enabled for your account."}
            return
        models.sort(key=lambda m: (m["available"] is not True, "chat" not in m["capabilities"], order.get(m["id"], 9999)))
        yield {"type": "done", "result": {"preset": p.id, "kind": "llm", "base_url": base_url, "models": models, "tested": len(chat_probe), "recommended": rec}}

    @staticmethod
    def _recommend(chat: list[dict], vision: list[dict], embeds: list[dict]) -> dict[str, str | None]:
        good = [m for m in chat if m["json_ok"]]
        synthesis = good[0]["id"] if good else next((m["id"] for m in chat if m["available"]), None)
        # Planning: prefer a different, faster model that also passed the JSON check; otherwise reuse synthesis.
        planning = next((m["id"] for m in sorted(good, key=lambda m: _size_b(m["id"]) or 1e9) if m["id"] != synthesis and _size_b(m["id"]) >= 20), synthesis)
        vis = next((m["id"] for m in vision if m["available"]), None)
        return {
            "synthesis": synthesis,
            "planning": planning,
            "vision": vis,
            "reranking": planning,
            "ocr": vis,
            "fallback": next((m["id"] for m in good if m["id"] not in (synthesis, planning) and _size_b(m["id"]) >= 20), None),
            "embedding": next((m["id"] for m in embeds if m.get("available")), None),
        }

    # ---------------------------------------------------------------- apply
    def _upsert(self, pid: str, name: str, adapter: str, config: dict[str, Any], secrets: dict[str, str]) -> str:
        if self.registry.db.one("SELECT 1 FROM provider_configs WHERE id=?", (pid,)):
            self.registry.update(pid, name=name, config=config, secrets=secrets or None, enabled=True)
        else:
            self.registry.create(name, adapter, config, secrets, provider_id=pid)
        return pid

    @staticmethod
    def _pid(preset: str, model: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
        if slug.startswith(preset + "-"):  # "nvidia/nemotron…" under the nvidia preset → "nvidia-nemotron…"
            slug = slug[len(preset) + 1 :]
        return f"{preset}-{slug}"[:64].rstrip("-")

    def _cleanup(self, prefix: str, keep: set[str]) -> list[str]:
        """Remove providers this service created earlier that nothing uses any more."""
        removed = []
        for rec in self.registry.list_records():
            if rec.id in keep or not rec.id.startswith(prefix + "-"):
                continue
            if not self.registry.references(rec.id):
                self.registry.delete(rec.id)
                removed.append(rec.id)
        return removed

    def apply(self, preset_id: str, api_key: str | None, assignments: dict[str, str | None], base_url: str | None = None) -> dict[str, Any]:
        p = self._preset(preset_id)
        secrets = self._secrets(p, api_key)
        changed: dict[str, list[str]] = {}

        def route(task: Task, ids: list[str]) -> None:
            ids = list(dict.fromkeys(i for i in ids if i))
            if ids:
                self.registry.set_route(RouteConfig(task=task, provider_ids=ids))
                changed[task.value] = ids

        if p.kind == "search":
            pid = self._upsert(p.id, p.name, p.adapter, {}, secrets)
            route(Task.WEB_SEARCH, [pid])
            if p.adapter == "tavily":
                route(Task.WEB_EXTRACTION, ["builtin-web", pid] if self.registry.is_enabled("builtin-web") else [pid])
            return {"routes": changed}

        base_url = (base_url or p.base_url).rstrip("/")
        a = {k: (v or None) for k, v in assignments.items()}
        if not a.get("synthesis"):
            raise ValueError("Choose a model for report writing")

        if p.id == "custom" and not base_url:
            raise ValueError("Enter the endpoint address (base URL)")
        host = urlsplit(base_url).hostname or "endpoint"
        # custom endpoints are keyed by host so several can coexist
        prefix = p.id if p.id != "custom" else "custom-" + re.sub(r"[^a-z0-9]+", "-", host.lower())
        label = p.name if p.id != "custom" else host

        def chat_provider(model: str | None, vision: bool = False) -> str | None:
            if not model:
                return None
            cfg = self._llm_config(p, base_url, chat_model=model, vision_model=model if vision else "")
            return self._upsert(self._pid(prefix, model), f"{label} · {model}", p.adapter, cfg, secrets)

        synth = chat_provider(a["synthesis"], vision=a.get("vision") == a["synthesis"])
        plan = chat_provider(a.get("planning") or a["synthesis"], vision=a.get("vision") == (a.get("planning") or a["synthesis"]))
        fb = chat_provider(a.get("fallback"))
        vis = chat_provider(a.get("vision"), vision=True) if a.get("vision") else None
        route(Task.SYNTHESIS, [synth, fb, plan])
        route(Task.VERIFICATION, [synth, fb, plan])
        for t in (Task.PLANNING, Task.DECOMPOSITION, Task.REWRITING, Task.CHAT):  # fast model: chat answers in seconds
            route(t, [plan, synth, fb])
        if vis:
            route(Task.VISION, [vis])
        notes: list[str] = []

        def set_exact(task: Task, ids: list[str]) -> None:
            ids = [i for i in dict.fromkeys(ids) if i and self.registry.db.one("SELECT 1 FROM provider_configs WHERE id=?", (i,))]
            self.registry.set_route(RouteConfig(task=task, provider_ids=ids))
            changed[task.value] = ids

        # Result ranking: the connected chat model by default; "local" = the CPU rerankers; "off" = no reranking.
        openai_like = p.adapter == "openai_compatible"
        rr = a.get("reranking", (a.get("planning") or a["synthesis"]) if openai_like else "local")
        if rr == "off":
            set_exact(Task.RERANKING, [])
        elif rr == "local" or not openai_like:
            set_exact(Task.RERANKING, ["local-cross-encoder", "local-lexical-rerank"])
        else:
            cfg = self._llm_config(p, base_url, chat_model=rr)
            rid = self._upsert(self._pid(prefix, rr)[:58] + "-rerank", f"{label} · {rr} (ranking)", "llm_rerank", cfg, secrets)
            set_exact(Task.RERANKING, [rid, "local-cross-encoder", "local-lexical-rerank"])

        # Scanned pages: the connected vision model by default; "local" = PaddleOCR on this computer; "off" = skip.
        ocr = a.get("ocr", a.get("vision") if openai_like else "local")
        if ocr == "off":
            set_exact(Task.OCR, [])
        elif ocr == "local" or not ocr or not openai_like:
            set_exact(Task.OCR, ["local-paddleocr"])
        else:
            cfg = self._llm_config(p, base_url, vision_model=ocr)
            oid = self._upsert(self._pid(prefix, ocr)[:61] + "-ocr", f"{label} · {ocr} (scanned pages)", "vlm_ocr", cfg, secrets)
            set_exact(Task.OCR, [oid, "local-paddleocr"])
            notes.append("Scanned pages are read by the connected vision model; page images are sent to the service.")
        if a.get("embedding"):
            cfg = self._llm_config(p, base_url, embedding_model=a["embedding"])
            eid = self._upsert(self._pid(prefix, a["embedding"])[:60] + "-emb", f"{label} · {a['embedding']} (embeddings)", p.adapter, cfg, secrets)
            route(Task.TEXT_EMBEDDING, [eid])
        keep = {i for ids in changed.values() for i in ids}
        removed = self._cleanup(prefix, keep)
        return {"routes": changed, "notes": notes, "embedding_provider": changed.get("text_embedding", [None])[0], "removed": removed}
