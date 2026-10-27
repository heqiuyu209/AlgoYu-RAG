"""Signed-service PDF worker orchestration; no input is used as a path."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys

from learning_os.hoj.auth import check_deadline, now_ms
from learning_os.hoj.contracts import ServiceError
from learning_os.hoj.pdf_text import MAX_PDF_BYTES, MAX_PDF_JSON_BYTES, NATIVE_PDF_PARSERS, split_pdf_pages

PDF_FAILURES = {"PDF_INVALID", "PDF_ENCRYPTED", "PDF_TOO_MANY_PAGES", "PDF_CONTENT_STREAM_TOO_LARGE",
                "PDF_UNSUPPORTED_TEXT", "PDF_TOO_MUCH_TEXT", "PDF_NO_TEXT", "PDF_RESOURCE_LIMIT", "PDF_OCR_FAILED"}


def extract_pdf(data: bytes, deadline: int, *, ocr: bool = False) -> dict:
    check_deadline(deadline)
    if not data.startswith(b"%PDF-") or len(data) > MAX_PDF_BYTES:
        raise ServiceError(400, "PDF_INVALID")
    # No service key, provider key, model directory or personal .env goes into
    # the child. PYTHONPATH is operator-owned (also used by isolated tests).
    env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH"}}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    timeout = max(0.001, min(85.0 if ocr else 25.0, (deadline - now_ms()) / 1000))
    grouped = ocr and sys.platform.startswith("linux")
    def kill(child):
        if grouped:
            try:
                os.killpg(child.pid, getattr(signal, "SIGKILL", 9))
            except ProcessLookupError:
                pass
        else:
            child.kill()
    try:
        child = subprocess.Popen([sys.executable, "-m", "learning_os.hoj.pdf_ocr_worker" if ocr else "learning_os.hoj.pdf_worker"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, start_new_session=grouped)
    except OSError:
        raise ServiceError(503, "PDF_PARSER_UNAVAILABLE") from None
    try:
        try:
            raw, _ = child.communicate(input=data, timeout=timeout)
        except subprocess.TimeoutExpired:
            kill(child)
            child.communicate()
            raise ServiceError(400, "PDF_RESOURCE_LIMIT") from None
    finally:
        if child.poll() is None:
            kill(child)
            child.wait(timeout=5)
        if grouped:
            # Reap any OCR descendant even when its immediate parent crashed.
            kill(child)
        for pipe in (child.stdin, child.stdout):
            if pipe is not None:
                pipe.close()
    if child.returncode != 0:
        raise ServiceError(400, "PDF_RESOURCE_LIMIT")
    check_deadline(deadline)
    try:
        if len(raw) > MAX_PDF_JSON_BYTES:
            raise ValueError()
        result = json.loads(raw)
        if result in ({"code": "PDF_PARSER_UNAVAILABLE"}, {"code": "PDF_OCR_UNAVAILABLE"}):
            raise ServiceError(503, result["code"])
        if isinstance(result, dict) and set(result) == {"code"} and result["code"] in PDF_FAILURES:
            raise ServiceError(400, result["code"])
        if (not isinstance(result, dict) or set(result) != {"sourceHash", "pages", "content", "emptyPages", "parser"}
                or result["sourceHash"] != hashlib.sha256(data).hexdigest()
                or result["parser"] not in ({"pdfium-tesseract-chi_sim-eng-v1"} if ocr else NATIVE_PDF_PARSERS)
                or type(result["pages"]) is not int or not isinstance(result["content"], str)):
            raise ValueError()
        pages = split_pdf_pages(result["content"])
        if (not isinstance(result["emptyPages"], list) or any(type(p) is not int for p in result["emptyPages"])
                or result["pages"] != len(pages) or result["emptyPages"] != [i for i, text in enumerate(pages, 1) if not text.strip()]):
            raise ValueError()
        return result
    except ServiceError:
        raise
    except Exception:
        raise ServiceError(502, "INVALID_PDF_RESULT") from None
