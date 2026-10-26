"""Credential encryption at rest (Fernet / AES-128-CBC + HMAC)."""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.observability.logging import register_secret


class SecretBox:
    def __init__(self, key: str | bytes):
        self._fernet = Fernet(key if isinstance(key, bytes) else key.encode())

    @classmethod
    def from_settings(cls, secret_key: str, data_dir: Path) -> "SecretBox":
        if secret_key:
            return cls(secret_key)
        key_file = data_dir / "secret.key"
        if key_file.exists():
            return cls(key_file.read_bytes().strip())
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            pass
        return cls(key)

    def encrypt(self, plaintext: str) -> str:
        register_secret(plaintext)
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            value = self._fernet.decrypt(token.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("Stored credential cannot be decrypted with the current secret key") from exc
        register_secret(value)
        return value
