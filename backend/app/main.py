"""FastAPI application entry point.  Run: ``uvicorn app.main:app``"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_documents, routes_notebooks, routes_providers, routes_research
from app.config.settings import Settings, get_settings
from app.core.container import Container
from app.observability.logging import configure_logging, get_logger, log_event, trace_id_var
from app.providers.base import Task
from app.services.parsing.docling_parser import DoclingParser

log = get_logger("omni.http")
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or Container(settings)
        app.state.container = c
        await c.start()
        log_event(log, "OmniResearch started", data_dir=str(settings.data_dir.resolve()))
        try:
            yield
        finally:
            await c.stop()  # graceful shutdown: cancel jobs, flush, close clients
            log_event(log, "OmniResearch stopped")

    app = FastAPI(title="OmniResearch", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=False)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        tid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        trace_id_var.set(tid)
        t0 = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            log_event(log, "request", method=request.method, path=request.url.path, status=response.status_code, ms=round((time.perf_counter() - t0) * 1000, 1))
        response.headers["x-request-id"] = tid
        return response

    routes_providers.install_error_handlers(app)
    app.include_router(routes_providers.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_research.router)
    app.include_router(routes_notebooks.router)

    @app.get("/api/health")
    def health(request: Request):
        c: Container = request.app.state.container
        try:
            c.db.one("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False
        try:
            c.store.client.get_collections()
            vec_ok = True
        except Exception:
            vec_ok = False
        try:
            import paddleocr  # noqa: F401

            paddle = True
        except ImportError:
            paddle = False
        return {
            "status": "ok" if db_ok and vec_ok else "degraded",
            "database": db_ok,
            "vector_store": {"ok": vec_ok, "mode": "server" if settings.qdrant_url else "embedded"},
            "parser": {"docling_installed": DoclingParser.available(), "configured": settings.parser},
            "paddleocr_installed": paddle,
            "configured_tasks": {t.value: c.router.is_configured(t) for t in Task},
            "admin_token_configured": bool(settings.admin_token),
            "max_upload_mb": settings.max_upload_mb,
        }

    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            candidate = (FRONTEND_DIST / path).resolve()
            if path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
