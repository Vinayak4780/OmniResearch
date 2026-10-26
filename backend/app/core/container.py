"""Dependency-injection container: builds every service from settings and wires them together."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.core.netguard import OutboundPolicy
from app.core.secrets import SecretBox
from app.observability.logging import get_logger, log_event
from app.observability.metrics import MetricsRecorder
from app.providers.base import BaseProvider, Task
from app.providers.http import ProviderHttp
from app.providers.registry import BUILTIN_ADAPTERS, ProviderRegistry, RouteConfig
from app.providers.router import ProviderRouter
from app.repositories.db import Database
from app.services.ingestion.service import IngestionService
from app.services.localmodels import LocalModelService
from app.services.notebook_studio import NotebookStudioService
from app.services.notebook_web import NotebookWebService
from app.services.notebooks import NotebookService
from app.services.quickconnect import QuickConnectService
from app.services.research.followup import FollowUpService
from app.services.research.service import ResearchService
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.hybrid import Retriever
from app.services.retrieval.index_manager import IndexManager
from app.services.retrieval.vector_store import VectorStore
from app.services.web.research import WebResearchService

log = get_logger("omni.container")


class Container:
    def __init__(
        self,
        settings: Settings,
        *,
        extra_adapters: list[type[BaseProvider]] | None = None,
        policy: OutboundPolicy | None = None,
        vector_store: VectorStore | None = None,
        parser: Any = None,
        checkpoints: bool = True,
        backoff_base_s: float = 0.5,
    ):
        settings.ensure_dirs()
        self.settings = settings
        self.db = Database(settings.db_path)
        self.db.migrate()
        self.secret_box = SecretBox.from_settings(settings.secret_key, settings.data_dir)
        self.policy = policy or OutboundPolicy(
            allow_private_hosts=settings.outbound_allow_private_hosts,
            provider_allowlist=settings.outbound_provider_allowlist,
            blocklist=settings.outbound_blocklist,
        )
        self.http = ProviderHttp(self.policy)
        self.registry = ProviderRegistry(self.db, self.secret_box, self.http, adapters=BUILTIN_ADAPTERS + (extra_adapters or []))
        self.metrics = MetricsRecorder(self.db)
        self.router = ProviderRouter(self.registry, self.metrics, backoff_base_s=backoff_base_s)
        if vector_store is not None:
            self.store = vector_store
        else:
            self.store = VectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key, path=str(settings.data_dir / "qdrant") if not settings.qdrant_url else None)
        self.index = IndexManager(self.db, self.router, self.store)
        self.bm25 = BM25Index(self.db)
        self.retriever = Retriever(self.db, self.router, self.index, self.bm25)
        self.ingestion = IngestionService(settings, self.db, self.router, self.index, parser=parser)
        self.web = WebResearchService(self.db, self.router)
        self.research = ResearchService(
            settings,
            self.db,
            self.router,
            self.retriever,
            self.web,
            render_page=self.ingestion.render_page,
            pdf_path=self.ingestion.pdf_path,
            checkpoint_path=(settings.data_dir / "checkpoints.sqlite3") if checkpoints else None,
        )
        self.followup = FollowUpService(self.db, self.router, self.retriever)
        self.quick = QuickConnectService(self.registry)
        self.local_models = LocalModelService(self)
        self.notebooks = NotebookService(self)
        self.notebook_web = NotebookWebService(self)
        self.studio = NotebookStudioService(self)

    async def start(self) -> None:
        self.bootstrap()
        self.ingestion.start()
        await self.research.start()
        await self.index.resume_interrupted()

    async def stop(self) -> None:
        await self.studio.stop()
        await self.research.stop()
        await self.ingestion.stop()
        await self.index.wait_idle()
        await self.registry.aclose()
        await self.http.aclose()
        self.store.close()
        self.db.close()

    # ---------------------------------------------------------------- first-run configuration
    def bootstrap(self) -> None:
        """Create credential-free local providers on first start, then apply an optional bootstrap file."""
        if not self.db.kv_get("bootstrapped"):
            self._create_local_defaults()
            self.db.kv_set("bootstrapped", True)
        path = os.environ.get("OMNI_PROVIDERS_FILE")
        if path and Path(path).exists():
            self.apply_bootstrap_file(Path(path))

    def _create_local_defaults(self) -> None:
        def make(pid: str, name: str, adapter: str, config: dict) -> bool:
            try:
                self.registry.create(name, adapter, config, provider_id=pid)
                return True
            except ValueError as exc:
                log_event(log, "default provider not created", provider_id=pid, error=str(exc))
                return False

        try:
            import fastembed  # noqa: F401

            have_fastembed = True
        except ImportError:
            have_fastembed = False
        try:
            import paddleocr  # noqa: F401

            have_paddle = True
        except ImportError:
            have_paddle = False

        make("local-hashing", "Local hashing embeddings", "local_hashing", {"dimensions": 512})
        make("local-lexical-rerank", "Local lexical reranker", "local_lexical_rerank", {})
        make("builtin-web", "Built-in webpage extractor", "builtin_web_extractor", {})
        emb = "local-hashing"
        rr = "local-lexical-rerank"
        if have_fastembed:
            if make("local-fastembed", "FastEmbed bge-small (CPU)", "fastembed", {"model": "BAAI/bge-small-en-v1.5"}):
                emb = "local-fastembed"
            if make("local-cross-encoder", "FastEmbed cross-encoder (CPU)", "fastembed_rerank", {}):
                rr = "local-cross-encoder"
        routes = {Task.TEXT_EMBEDDING: [emb], Task.RERANKING: [rr] + ([] if rr == "local-lexical-rerank" else ["local-lexical-rerank"]), Task.WEB_EXTRACTION: ["builtin-web"]}
        if have_paddle and make("local-paddleocr", "PaddleOCR PP-OCRv5 (CPU)", "paddleocr_local", {}):
            routes[Task.OCR] = ["local-paddleocr"]
        for task, ids in routes.items():
            self.registry.set_route(RouteConfig(task=task, provider_ids=ids))

    def apply_bootstrap_file(self, path: Path) -> None:
        """Declarative provider setup for headless/Docker deployments.

        Format: {"providers": [{"id", "name", "adapter", "config", "secrets_env": {"api_key": "ENV_VAR"}}],
                 "routes": {"synthesis": ["id", ...], ...}}
        Existing provider ids are left unchanged (the UI remains the source of truth after first creation).
        """
        data = json.loads(path.read_text())
        for p in data.get("providers", []):
            if self.db.one("SELECT 1 FROM provider_configs WHERE id=?", (p["id"],)):
                continue
            secrets = {k: os.environ.get(env, "") for k, env in (p.get("secrets_env") or {}).items() if os.environ.get(env)}
            try:
                self.registry.create(p.get("name", p["id"]), p["adapter"], p.get("config", {}), secrets, provider_id=p["id"])
            except ValueError as exc:
                log_event(log, "bootstrap provider rejected", provider_id=p.get("id"), error=str(exc))
        for task, ids in (data.get("routes") or {}).items():
            try:
                cur = self.registry.get_route(Task(task))
                if not cur.provider_ids:
                    self.registry.set_route(RouteConfig(task=Task(task), provider_ids=ids))
            except ValueError as exc:
                log_event(log, "bootstrap route rejected", task=task, error=str(exc))
