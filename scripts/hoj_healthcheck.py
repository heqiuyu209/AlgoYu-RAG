"""Authenticated local readiness probe; prints no service identity or snippets."""
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid
from learning_os.hoj.auth import signature


def main():
    key = Path(os.environ["HOJ_RAG_SHARED_KEY_FILE"]).read_text(encoding="ascii").strip()
    request_id = uuid.uuid4().hex
    end = str(time.time_ns() // 1000000 + 10000)
    path = "/internal/v1/retrieve"
    body = b'{"query":"readiness","allowedVersions":[],"topK":1}'
    request = urllib.request.Request("http://127.0.0.1:8097" + path, data=body, method="POST", headers={
        "Content-Type": "application/json", "X-HOJ-Request-Id": request_id, "X-HOJ-Deadline": end,
        "X-HOJ-Signature": signature(bytes.fromhex(key), path, request_id, end, body)})
    with urllib.request.urlopen(request, timeout=4) as response:
        if response.status != 200 or json.load(response) != {"items": []}:
            raise ValueError("Internal index not ready")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1) from None
