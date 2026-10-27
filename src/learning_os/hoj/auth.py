"""Request-bound HMAC, absolute deadlines and durable replay protection."""

from __future__ import annotations

import hashlib
import json
import hmac
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from learning_os.hoj.contracts import ServiceError


def now_ms() -> int:
    return time.time_ns() // 1000000


def check_deadline(deadline: int) -> None:
    if deadline <= now_ms():
        raise ServiceError(408, "DEADLINE_EXCEEDED")


def signature(key: bytes, path: str, request_id: str, deadline: str, body: bytes) -> str:
    material = "\n".join(("v1", "POST", path, request_id, deadline, hashlib.sha256(body).hexdigest()))
    return hmac.new(key, material.encode("ascii"), hashlib.sha256).hexdigest()


class RequestAuth:
    def __init__(self, key: str, state_path: Path):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("HOJ_RAG_SHARED_KEY must be 32 random bytes encoded as lowercase hex")
        self.key = bytes.fromhex(key)
        self.path = state_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS request_nonce (id TEXT PRIMARY KEY, deadline INTEGER NOT NULL)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def verify(self, path: str, headers, body: bytes) -> int:
        request_id = headers.get("x-hoj-request-id", "")
        deadline = headers.get("x-hoj-deadline", "")
        supplied = headers.get("x-hoj-signature", "")
        if (not re.fullmatch(r"[a-f0-9]{32}", request_id)
                or not re.fullmatch(r"[0-9]{13}", deadline)
                or not re.fullmatch(r"[a-f0-9]{64}", supplied)):
            raise ServiceError(401, "INVALID_SERVICE_AUTH")
        expected = signature(self.key, path, request_id, deadline, body)
        if not hmac.compare_digest(expected, supplied):
            raise ServiceError(401, "INVALID_SERVICE_AUTH")
        end = int(deadline)
        check_deadline(end)
        if end > now_ms() + 120000:
            # Only a signed whole-book PDF ingestion may outlive the normal
            # two-minute service deadline. Other operations retain that bound.
            from learning_os.hoj.pdf_text import MAX_PDF_INGEST_MS
            try:
                payload = json.loads(body) if path == "/internal/v1/ingest" else None
                long_pdf = isinstance(payload, dict) and payload.get("format") == "pdf"
            except (ValueError, UnicodeError, RecursionError):
                long_pdf = False
            if not long_pdf or end > now_ms() + MAX_PDF_INGEST_MS:
                raise ServiceError(401, "INVALID_DEADLINE")
        try:
            with self.connection() as db:
                db.execute("DELETE FROM request_nonce WHERE deadline<=?", (now_ms(),))
                db.execute("INSERT INTO request_nonce VALUES (?,?)", (request_id, end))
        except sqlite3.IntegrityError:
            raise ServiceError(409, "REPLAYED_REQUEST") from None
        return end
