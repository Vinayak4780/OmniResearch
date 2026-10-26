from __future__ import annotations

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from app.core.netguard import OutboundPolicy
from app.core.secrets import SecretBox
from app.observability.metrics import MetricsRecorder
from app.providers.http import ProviderHttp
from app.providers.registry import BUILTIN_ADAPTERS, ProviderRegistry
from app.providers.router import ProviderRouter
from app.repositories.db import Database
from tests import fakes


@pytest.fixture(autouse=True)
def _reset_fakes():
    fakes.CALLS.clear()
    fakes.CHAT_HANDLERS.clear()
    fakes.FakeSearchProvider.PAGES = {}
    yield


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.sqlite3")
    d.migrate()
    yield d
    d.close()


@pytest.fixture
def policy():
    return OutboundPolicy(allow_private_hosts=["localhost", "127.0.0.1"], resolve_dns=False)


@pytest_asyncio.fixture
async def http(policy):
    h = ProviderHttp(policy)
    yield h
    await h.aclose()


@pytest.fixture
def secret_box():
    return SecretBox(Fernet.generate_key())


@pytest.fixture
def registry(db, secret_box, http):
    return ProviderRegistry(db, secret_box, http, adapters=BUILTIN_ADAPTERS + fakes.FAKE_ADAPTERS)


@pytest.fixture
def router(registry, db):
    return ProviderRouter(registry, MetricsRecorder(db), backoff_base_s=0.0, breaker_threshold=3, breaker_reset_s=60)
