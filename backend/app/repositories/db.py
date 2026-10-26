"""SQLite access with versioned migrations.

SQLite is used for metadata, jobs, provider configuration and metrics. Migrations are append-only
SQL scripts; to move to another RDBMS, port these scripts and the small Database helper.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE provider_configs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        adapter TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        config_json TEXT NOT NULL DEFAULT '{}',
        secrets_enc_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE capability_routes (
        task TEXT PRIMARY KEY,
        provider_ids_json TEXT NOT NULL DEFAULT '[]',
        fallback_enabled INTEGER NOT NULL DEFAULT 1,
        params_json TEXT NOT NULL DEFAULT '{}',
        updated_at TEXT NOT NULL
    );
    CREATE TABLE provider_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider_id TEXT NOT NULL,
        task TEXT,
        capability TEXT NOT NULL,
        ok INTEGER NOT NULL,
        latency_ms REAL NOT NULL,
        error_kind TEXT,
        error_message TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cost_usd REAL,
        fallback_from TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX ix_provider_calls_provider ON provider_calls(provider_id, created_at);
    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        sha256 TEXT NOT NULL UNIQUE,
        size_bytes INTEGER NOT NULL,
        collection TEXT NOT NULL DEFAULT 'default',
        page_count INTEGER,
        title TEXT,
        status TEXT NOT NULL,
        error TEXT,
        parser TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE ingestion_jobs (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        stage TEXT,
        progress REAL NOT NULL DEFAULT 0,
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE pages (
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_no INTEGER NOT NULL,
        width REAL,
        height REAL,
        text_chars INTEGER NOT NULL DEFAULT 0,
        extraction_method TEXT NOT NULL,
        ocr_status TEXT,
        ocr_error TEXT,
        ocr_provider TEXT,
        ocr_latency_ms REAL,
        text TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (document_id, page_no)
    );
    CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        page_no INTEGER NOT NULL,
        page_end INTEGER NOT NULL,
        section_path TEXT NOT NULL DEFAULT '',
        element_type TEXT NOT NULL,
        extraction_method TEXT NOT NULL,
        text TEXT NOT NULL,
        bbox_json TEXT,
        source_element_id TEXT,
        chunk_version TEXT NOT NULL,
        content_hash TEXT NOT NULL
    );
    CREATE INDEX ix_chunks_doc ON chunks(document_id);
    CREATE TABLE ocr_cache (
        cache_key TEXT PRIMARY KEY,
        provider_fingerprint TEXT NOT NULL,
        text TEXT NOT NULL,
        lines_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    CREATE TABLE vector_indexes (
        id TEXT PRIMARY KEY,
        collection_name TEXT NOT NULL UNIQUE,
        fingerprint TEXT NOT NULL,
        fingerprint_json TEXT NOT NULL,
        status TEXT NOT NULL,
        progress_done INTEGER NOT NULL DEFAULT 0,
        progress_total INTEGER NOT NULL DEFAULT 0,
        failed_batches INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        activated_at TEXT
    );
    CREATE TABLE indexed_chunks (
        index_id TEXT NOT NULL REFERENCES vector_indexes(id) ON DELETE CASCADE,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (index_id, chunk_id)
    );
    CREATE TABLE kv (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL
    );
    CREATE TABLE research_jobs (
        id TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        status TEXT NOT NULL,
        options_json TEXT NOT NULL DEFAULT '{}',
        stage TEXT,
        report_md TEXT,
        report_json TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE research_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL REFERENCES research_jobs(id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        stage TEXT NOT NULL,
        message TEXT NOT NULL,
        data_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX ix_research_events_job ON research_events(job_id, id);
    CREATE TABLE research_evidence (
        job_id TEXT NOT NULL REFERENCES research_jobs(id) ON DELETE CASCADE,
        evidence_id TEXT NOT NULL,
        data_json TEXT NOT NULL,
        PRIMARY KEY (job_id, evidence_id)
    );
    CREATE TABLE eval_runs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT NOT NULL DEFAULT '{}',
        metrics_json TEXT NOT NULL DEFAULT '{}',
        details_json TEXT NOT NULL DEFAULT '[]',
        error TEXT,
        created_at TEXT NOT NULL,
        finished_at TEXT
    );
    CREATE TABLE web_fetch_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        ok INTEGER NOT NULL,
        status_code INTEGER,
        error TEXT,
        created_at TEXT NOT NULL
    );
    """,
    # 2: follow-up chat on finished investigations
    """
    CREATE TABLE research_chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL REFERENCES research_jobs(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        citations_json TEXT NOT NULL DEFAULT '[]',
        check_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    CREATE INDEX ix_research_chat_job ON research_chat(job_id, id);
    """,
    # 3: attribute provider calls to research jobs (usage survives restarts) and record the model
    """
    ALTER TABLE provider_calls ADD COLUMN job_id TEXT;
    ALTER TABLE provider_calls ADD COLUMN model TEXT;
    CREATE INDEX ix_provider_calls_job ON provider_calls(job_id);
    """,
    # 4: the evaluation dashboard was removed
    """
    DROP TABLE IF EXISTS eval_runs;
    """,
    # 5: notebooks (NotebookLM-style workspace). Existing document collections become notebooks.
    """
    CREATE TABLE notebooks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    INSERT INTO notebooks(id, title, created_at, updated_at)
        SELECT collection, collection, MIN(created_at), MAX(updated_at) FROM documents GROUP BY collection;
    CREATE TABLE notebook_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        citations_json TEXT NOT NULL DEFAULT '{}',
        check_json TEXT NOT NULL DEFAULT '{}',
        source_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    CREATE INDEX ix_notebook_messages ON notebook_messages(notebook_id, id);
    CREATE TABLE notebook_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        citations_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    ALTER TABLE research_jobs ADD COLUMN notebook_id TEXT;
    """,
    # 6: sources added from a link, and Studio outputs (summaries, briefings, FAQs, mind maps, ...)
    """
    ALTER TABLE documents ADD COLUMN source_url TEXT;
    CREATE TABLE notebook_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        status TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        data_json TEXT NOT NULL DEFAULT '{}',
        citations_json TEXT NOT NULL DEFAULT '{}',
        check_json TEXT NOT NULL DEFAULT '{}',
        source_ids_json TEXT NOT NULL DEFAULT '[]',
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX ix_notebook_outputs ON notebook_outputs(notebook_id, id);
    """,
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

    def migrate(self) -> int:
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)")
            current = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
            for version, script in enumerate(MIGRATIONS, start=1):
                if version <= current:
                    continue
                self._conn.execute("BEGIN")
                try:
                    for stmt in [s for s in script.split(";") if s.strip()]:
                        self._conn.execute(stmt)
                    self._conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, utcnow()))
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                current = version
            return current

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: list) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(sql, rows)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value_json FROM kv WHERE key=?", (key,))
        return json.loads(row["value_json"]) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key, value_json) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, json.dumps(value)),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
