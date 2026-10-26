"""ProviderRegistry: the adapter catalog plus persisted, user-managed provider configurations."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.netguard import UnsafeURLError
from app.core.secrets import SecretBox
from app.providers.base import (
    TASK_CAPABILITY,
    BaseProvider,
    Capability,
    ProviderDescriptor,
    ProviderError,
    ProviderErrorKind,
    Task,
)
from app.providers.chat.anthropic import AnthropicProvider
from app.providers.chat.openai_compatible import OpenAICompatibleProvider
from app.providers.custom.http_adapter import CustomHTTPProvider
from app.providers.embeddings.embedding_api import EmbeddingAPIProvider
from app.providers.embeddings.local import FastEmbedProvider, HashingEmbeddingProvider
from app.providers.http import ProviderHttp
from app.providers.ocr.paddle import PaddleOCRLocalProvider, PaddleServingOCRProvider
from app.providers.ocr.vlm_ocr import VLMOCRProvider
from app.providers.reranking.llm_rerank import LLMRerankProvider
from app.providers.reranking.local import FastEmbedRerankProvider, LexicalRerankProvider
from app.providers.reranking.rerank_api import RerankAPIProvider
from app.providers.web_search.engines import BraveSearchProvider, ExaSearchProvider, SearXNGProvider, TavilyProvider
from app.providers.webpage_extraction.builtin import BuiltinWebExtractor
from app.repositories.db import Database, utcnow

BUILTIN_ADAPTERS: list[type[BaseProvider]] = [
    OpenAICompatibleProvider,
    AnthropicProvider,
    EmbeddingAPIProvider,
    RerankAPIProvider,
    LLMRerankProvider,
    CustomHTTPProvider,
    HashingEmbeddingProvider,
    FastEmbedProvider,
    LexicalRerankProvider,
    FastEmbedRerankProvider,
    PaddleOCRLocalProvider,
    PaddleServingOCRProvider,
    VLMOCRProvider,
    TavilyProvider,
    BraveSearchProvider,
    ExaSearchProvider,
    SearXNGProvider,
    BuiltinWebExtractor,
]

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")


class ProviderRecord(BaseModel):
    """Public view of a provider configuration. Secrets are never included, only which are set."""

    id: str
    name: str
    adapter: str
    enabled: bool
    config: dict[str, Any]
    secrets_set: dict[str, bool]
    capabilities: list[Capability]
    models: dict[str, str | None]
    created_at: str
    updated_at: str


class RouteConfig(BaseModel):
    task: Task
    provider_ids: list[str] = Field(default_factory=list, description="Preferred first, then ordered fallbacks")
    fallback_enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict, description="Task parameters: model, temperature, max_tokens")


class ConfigConflictError(Exception):
    def __init__(self, message: str, references: list[str]):
        super().__init__(message)
        self.references = references


class ProviderRegistry:
    def __init__(self, db: Database, secret_box: SecretBox, http: ProviderHttp, adapters: list[type[BaseProvider]] | None = None):
        self.db = db
        self.secret_box = secret_box
        self.http = http
        self._adapters: dict[str, type[BaseProvider]] = {}
        for a in adapters or BUILTIN_ADAPTERS:
            self.register_adapter(a)
        self._instances: dict[str, BaseProvider] = {}
        self.instance_kwargs: dict[str, dict[str, Any]] = {}  # adapter -> extra ctor kwargs (tests)

    # ---------------------------------------------------------------- catalog
    def register_adapter(self, cls: type[BaseProvider]) -> None:
        self._adapters[cls.adapter] = cls

    def adapter_class(self, adapter: str) -> type[BaseProvider]:
        if adapter not in self._adapters:
            raise ValueError(f"Unknown adapter '{adapter}'")
        return self._adapters[adapter]

    def catalog(self) -> list[ProviderDescriptor]:
        return [cls.describe() for cls in self._adapters.values()]

    # ---------------------------------------------------------------- validation
    def _validate(self, adapter: str, config: dict[str, Any]) -> BaseModel:
        cls = self.adapter_class(adapter)
        try:
            parsed = cls.config_model.model_validate(config)
        except ValidationError as exc:
            raise ValueError(f"Invalid configuration: {exc.errors(include_url=False, include_input=False)}") from exc
        bad = set(parsed.enabled_capabilities) - set(cls.capabilities)
        if bad:
            raise ValueError(f"Adapter {adapter} does not support capabilities {sorted(c.value for c in bad)}")
        base_url = getattr(parsed, "base_url", None)
        if base_url:
            try:
                self.http.policy.check_syntax(base_url, purpose="provider")
            except UnsafeURLError as exc:
                raise ValueError(f"base_url rejected: {exc}") from exc
        return parsed

    def _encrypt_secrets(self, adapter: str, secrets: dict[str, str | None], existing: dict[str, str]) -> dict[str, str]:
        allowed = set(self.adapter_class(adapter).secret_fields)
        out = dict(existing)
        for k, v in secrets.items():
            if k not in allowed:
                raise ValueError(f"Unknown secret field '{k}' for adapter {adapter}")
            if v is None:
                continue  # keep existing
            if v == "":
                out.pop(k, None)
            else:
                out[k] = self.secret_box.encrypt(v)
        return out

    # ---------------------------------------------------------------- CRUD
    def _row_to_record(self, row: dict[str, Any]) -> ProviderRecord:
        cls = self._adapters.get(row["adapter"])
        config = json.loads(row["config_json"])
        secrets_enc = json.loads(row["secrets_enc_json"])
        caps: list[Capability] = []
        models: dict[str, str | None] = {}
        if cls:
            try:
                inst = self.get(row["id"]) if row["enabled"] else self._build(row)
                caps = sorted(inst.active_capabilities, key=lambda c: c.value)
                models = {c.value: inst.model_for(c) for c in caps}
            except Exception:
                caps = sorted(cls.capabilities, key=lambda c: c.value)
        return ProviderRecord(
            id=row["id"],
            name=row["name"],
            adapter=row["adapter"],
            enabled=bool(row["enabled"]),
            config=config,
            secrets_set={f: f in secrets_enc for f in (cls.secret_fields if cls else [])},
            capabilities=caps,
            models=models,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_records(self) -> list[ProviderRecord]:
        return [self._row_to_record(r) for r in self.db.all("SELECT * FROM provider_configs ORDER BY created_at")]

    def get_record(self, provider_id: str) -> ProviderRecord:
        row = self.db.one("SELECT * FROM provider_configs WHERE id=?", (provider_id,))
        if not row:
            raise KeyError(provider_id)
        return self._row_to_record(row)

    def create(
        self,
        name: str,
        adapter: str,
        config: dict[str, Any],
        secrets: dict[str, str | None] | None = None,
        *,
        provider_id: str | None = None,
        enabled: bool = True,
    ) -> ProviderRecord:
        parsed = self._validate(adapter, config)
        pid = provider_id or re.sub(r"[^a-z0-9_\-]+", "-", name.lower()).strip("-")[:40] + "-" + uuid.uuid4().hex[:6]
        if not _ID_RE.match(pid):
            raise ValueError("Provider id must be 2-64 chars of a-z, 0-9, _ or -")
        if self.db.one("SELECT 1 FROM provider_configs WHERE id=?", (pid,)):
            raise ValueError(f"Provider id '{pid}' already exists")
        enc = self._encrypt_secrets(adapter, secrets or {}, {})
        now = utcnow()
        self.db.execute(
            "INSERT INTO provider_configs VALUES (?,?,?,?,?,?,?,?)",
            (pid, name, adapter, int(enabled), parsed.model_dump_json(), json.dumps(enc), now, now),
        )
        return self.get_record(pid)

    def update(
        self,
        provider_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        secrets: dict[str, str | None] | None = None,
        enabled: bool | None = None,
    ) -> ProviderRecord:
        row = self.db.one("SELECT * FROM provider_configs WHERE id=?", (provider_id,))
        if not row:
            raise KeyError(provider_id)
        new_config = row["config_json"]
        if config is not None:
            new_config = self._validate(row["adapter"], config).model_dump_json()
        enc = json.loads(row["secrets_enc_json"])
        if secrets:
            enc = self._encrypt_secrets(row["adapter"], secrets, enc)
        self.db.execute(
            "UPDATE provider_configs SET name=?, config_json=?, secrets_enc_json=?, enabled=?, updated_at=? WHERE id=?",
            (name or row["name"], new_config, json.dumps(enc), int(row["enabled"] if enabled is None else enabled), utcnow(), provider_id),
        )
        self._invalidate(provider_id)
        return self.get_record(provider_id)

    def references(self, provider_id: str) -> list[str]:
        refs = [f"route:{r['task']}" for r in self.db.all("SELECT task, provider_ids_json FROM capability_routes") if provider_id in json.loads(r["provider_ids_json"])]
        for idx in self.db.all("SELECT id, status, fingerprint_json FROM vector_indexes WHERE status IN ('active','building','validating','ready','previous')"):
            if json.loads(idx["fingerprint_json"]).get("provider_id") == provider_id:
                refs.append(f"vector_index:{idx['id']}({idx['status']})")
        return refs

    def delete(self, provider_id: str, *, force: bool = False) -> None:
        if not self.db.one("SELECT 1 FROM provider_configs WHERE id=?", (provider_id,)):
            raise KeyError(provider_id)
        refs = self.references(provider_id)
        if refs and not force:
            raise ConfigConflictError("Provider is still referenced; remove it from routes (or force) first", refs)
        if force:
            for r in self.db.all("SELECT task, provider_ids_json FROM capability_routes"):
                ids = [i for i in json.loads(r["provider_ids_json"]) if i != provider_id]
                self.db.execute("UPDATE capability_routes SET provider_ids_json=? WHERE task=?", (json.dumps(ids), r["task"]))
        self.db.execute("DELETE FROM provider_configs WHERE id=?", (provider_id,))
        self._invalidate(provider_id)

    # ---------------------------------------------------------------- instances
    def _build(self, row: dict[str, Any]) -> BaseProvider:
        cls = self.adapter_class(row["adapter"])
        config = cls.config_model.model_validate_json(row["config_json"])
        secrets = {k: self.secret_box.decrypt(v) for k, v in json.loads(row["secrets_enc_json"]).items()}
        return cls(row["id"], config, secrets, self.http, **self.instance_kwargs.get(row["adapter"], {}))

    def get(self, provider_id: str) -> BaseProvider:
        if provider_id in self._instances:
            return self._instances[provider_id]
        row = self.db.one("SELECT * FROM provider_configs WHERE id=?", (provider_id,))
        if not row:
            raise ProviderError(ProviderErrorKind.NOT_CONFIGURED, f"Provider '{provider_id}' does not exist", provider_id=provider_id)
        if not row["enabled"]:
            raise ProviderError(ProviderErrorKind.NOT_CONFIGURED, f"Provider '{provider_id}' is disabled", provider_id=provider_id)
        inst = self._build(row)
        self._instances[provider_id] = inst
        return inst

    def is_enabled(self, provider_id: str) -> bool:
        row = self.db.one("SELECT enabled FROM provider_configs WHERE id=?", (provider_id,))
        return bool(row and row["enabled"])

    def _invalidate(self, provider_id: str) -> None:
        self._instances.pop(provider_id, None)

    def build_unsaved(self, adapter: str, config: dict[str, Any], secrets: dict[str, str]) -> BaseProvider:
        """Instantiate a provider from a draft configuration (for 'test connection' before saving)."""
        cls = self.adapter_class(adapter)
        parsed = self._validate(adapter, config)
        return cls("draft", parsed, secrets, self.http, **self.instance_kwargs.get(adapter, {}))

    # ---------------------------------------------------------------- routes
    def get_route(self, task: Task) -> RouteConfig:
        row = self.db.one("SELECT * FROM capability_routes WHERE task=?", (task.value,))
        if not row:
            return RouteConfig(task=task)
        return RouteConfig(task=task, provider_ids=json.loads(row["provider_ids_json"]), fallback_enabled=bool(row["fallback_enabled"]), params=json.loads(row["params_json"]))

    def list_routes(self) -> list[RouteConfig]:
        return [self.get_route(t) for t in Task]

    def set_route(self, route: RouteConfig) -> RouteConfig:
        cap = TASK_CAPABILITY[route.task]
        if len(set(route.provider_ids)) != len(route.provider_ids):
            raise ValueError("Duplicate provider in route")
        for pid in route.provider_ids:
            row = self.db.one("SELECT * FROM provider_configs WHERE id=?", (pid,))
            if not row:
                raise ValueError(f"Unknown provider '{pid}'")
            inst = self._build(row)
            if not inst.supports(cap):
                raise ValueError(f"Provider '{pid}' ({row['adapter']}) does not provide capability '{cap.value}' required by task '{route.task.value}'")
        allowed_params = {"model", "temperature", "max_tokens"} if cap in (Capability.CHAT, Capability.VISION) else {"model"}
        if set(route.params) - allowed_params:
            raise ValueError(f"Unsupported route params {sorted(set(route.params) - allowed_params)}")
        self.db.execute(
            """INSERT INTO capability_routes(task, provider_ids_json, fallback_enabled, params_json, updated_at) VALUES (?,?,?,?,?)
               ON CONFLICT(task) DO UPDATE SET provider_ids_json=excluded.provider_ids_json, fallback_enabled=excluded.fallback_enabled,
               params_json=excluded.params_json, updated_at=excluded.updated_at""",
            (route.task.value, json.dumps(route.provider_ids), int(route.fallback_enabled), json.dumps(route.params), utcnow()),
        )
        return route

    async def aclose(self) -> None:
        for inst in self._instances.values():
            try:
                await inst.aclose()
            except Exception:
                pass
        self._instances.clear()
