"""Application settings, loaded from environment variables (prefix ``OMNI_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNI_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("./data"), description="Root for SQLite, uploads, caches, local Qdrant.")
    database_path: Path | None = None

    # Qdrant: when url is empty, the embedded on-disk mode under data_dir/qdrant is used.
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Security
    admin_token: str = Field(default="", description="Bearer token for admin endpoints. Empty = loopback-only access.")
    secret_key: str = Field(default="", description="Fernet key for credential encryption. Empty = generated key file.")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Outbound network controls (SSRF protection)
    outbound_allow_private_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "host.docker.internal", "qdrant", "ollama"],
        description="Hosts that may resolve to private/loopback addresses (for local model servers).",
    )
    outbound_provider_allowlist: list[str] = Field(
        default_factory=list,
        description="If non-empty, provider endpoints must match one of these hostnames (suffix match).",
    )
    outbound_blocklist: list[str] = Field(default_factory=list)

    # Uploads
    max_upload_mb: int = 100
    max_pages_per_document: int = 2000

    # Ingestion
    ingestion_concurrency: int = 2
    ocr_concurrency: int = 1
    ocr_text_threshold_chars: int = 40
    ocr_render_dpi: int = 200
    parser: str = Field(default="auto", description="auto | docling | pymupdf")

    # Chunking
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 200

    # Agent
    agent_max_iterations: int = 3
    agent_max_subqueries: int = 6
    agent_evidence_per_query: int = 8
    agent_max_evidence: int = 40

    # Web
    web_fetch_max_bytes: int = 3_000_000
    web_fetch_timeout_s: float = 20.0
    web_max_redirects: int = 5

    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.database_path or (self.data_dir / "omni.sqlite3")

    def ensure_dirs(self) -> None:
        for sub in ("", "uploads", "cache/ocr", "cache/pages", "qdrant", "reports"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
