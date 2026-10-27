"""Local full-page OCR. Binary stdin, fixed JSON stdout, no user paths or URLs.

This opt-in path rasterizes *every* physical page, including pages with both
embedded text and scanned text. It never changes the original PDF. Normal
text extraction remains in pdf_worker and does not import OCR dependencies.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from learning_os.hoj.pdf_text import MAX_PDF_BYTES, MAX_PDF_CHARACTERS, MAX_PDF_PAGES, PDF_PAGE_SEPARATOR
from learning_os.hoj.pdf_worker import PdfFailure

OCR_PARSER = "pdfium-tesseract-chi_sim-eng-v1"
MAX_PIXELS = 16000000
MAX_SIDE = 10000


def image_size(width: float, height: float) -> tuple[int, int]:
    scale = 200 / 72
    if not all(math.isfinite(n) and n > 0 for n in (width, height)):
        raise PdfFailure("PDF_INVALID")
    size = math.ceil(width * scale), math.ceil(height * scale)
    if max(size) > MAX_SIDE or size[0] * size[1] > MAX_PIXELS:
        raise PdfFailure("PDF_RESOURCE_LIMIT")
    return size


def recognize(image, directory: Path, deadline: float) -> str:
    # Both names are private, server-generated constants. No PDF filename is
    # ever passed to a shell or interpreted by Tesseract as an input location.
    source = directory / "page.png"
    target = directory / "page.txt"
    image.save(source, format="PNG")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PdfFailure("PDF_RESOURCE_LIMIT")
    env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP"}}
    env.update(OMP_THREAD_LIMIT="2", OMP_NUM_THREADS="2")
    try:
        with target.open("wb") as output:
            result = subprocess.run(["tesseract", str(source), "stdout", "-l", "chi_sim+eng", "--psm", "3"],
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                                    timeout=min(20.0, remaining), env=env, check=False)
        if result.returncode != 0:
            raise PdfFailure("PDF_OCR_FAILED")
    except FileNotFoundError:
        raise PdfFailure("PDF_OCR_UNAVAILABLE") from None
    except subprocess.TimeoutExpired:
        raise PdfFailure("PDF_RESOURCE_LIMIT") from None
    if target.stat().st_size > 800000:
        raise PdfFailure("PDF_TOO_MUCH_TEXT")
    try:
        text = target.read_text(encoding="utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n").replace("\f", " ").strip()
        if "\0" in text or any(ord(c) < 32 and c not in "\n\t" for c in text):
            raise ValueError()
        return text
    except (UnicodeError, ValueError):
        raise PdfFailure("PDF_UNSUPPORTED_TEXT") from None


def parse(data: bytes) -> dict:
    if not data.startswith(b"%PDF-") or len(data) > MAX_PDF_BYTES:
        raise PdfFailure("PDF_INVALID")
    # Pypdf performs the same encryption/page policy checks as the native path.
    from pypdf import PdfReader
    with io.BytesIO(data) as stream:
        reader = PdfReader(stream, strict=True)
        if reader.is_encrypted:
            raise PdfFailure("PDF_ENCRYPTED")
        count = len(reader.pages)
        if not 1 <= count <= MAX_PDF_PAGES:
            raise PdfFailure("PDF_TOO_MANY_PAGES")
    import pypdfium2 as pdfium
    end = time.monotonic() + 80.0
    pages, empty, total = [], [], 0
    # Closing the PdfDocument also releases the original binary backing store.
    with pdfium.PdfDocument(data) as document, tempfile.TemporaryDirectory(prefix="hoj-ocr-") as temp:
        if len(document) != count:
            raise PdfFailure("PDF_INVALID")
        for number in range(count):
            if time.monotonic() >= end:
                raise PdfFailure("PDF_RESOURCE_LIMIT")
            page = document[number]
            try:
                image_size(*page.get_size())  # Check before allocating a bitmap.
                bitmap = page.render(scale=200 / 72, rev_byteorder=True)
                try:
                    with bitmap.to_pil() as image:
                        text = recognize(image, Path(temp), end)
                finally:
                    bitmap.close()
            finally:
                page.close()
            if not text:
                empty.append(number + 1)
            total += len(text) + (len(PDF_PAGE_SEPARATOR) if pages else 0)
            if total > MAX_PDF_CHARACTERS:
                raise PdfFailure("PDF_TOO_MUCH_TEXT")
            pages.append(text)
    if len(empty) == count:
        raise PdfFailure("PDF_NO_TEXT")
    return {"sourceHash": hashlib.sha256(data).hexdigest(), "pages": count,
            "content": PDF_PAGE_SEPARATOR.join(pages), "emptyPages": empty, "parser": OCR_PARSER}


def main():
    if sys.platform.startswith("linux"):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    try:
        data = sys.stdin.buffer.read(MAX_PDF_BYTES + 1)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            result = parse(data)
    except PdfFailure as error:
        result = {"code": error.code}
    except ImportError:
        result = {"code": "PDF_OCR_UNAVAILABLE"}
    except MemoryError:
        result = {"code": "PDF_RESOURCE_LIMIT"}
    except Exception:
        result = {"code": "PDF_INVALID"}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


if __name__ == "__main__":
    main()
