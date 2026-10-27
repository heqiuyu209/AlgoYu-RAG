"""One bounded PDF parse per process. Only raw stdin and fixed JSON stdout.

Not an index, model, file-path API, OCR engine, or provider client. Runtime
limits are applied before importing pypdf in the deployed Linux container.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import sys

from learning_os.hoj.pdf_text import MAX_PDF_BYTES, MAX_PDF_CHARACTERS, MAX_PDF_PAGES, PDF_PAGE_SEPARATOR


class PdfFailure(Exception):
    def __init__(self, code):
        self.code = code


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", " ").strip()
    if "\x00" in text:
        raise PdfFailure("PDF_UNSUPPORTED_TEXT")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeError:
        raise PdfFailure("PDF_UNSUPPORTED_TEXT") from None
    return text


def receipt(data: bytes, pages: list[str], parser: str) -> dict:
    content = PDF_PAGE_SEPARATOR.join(pages)
    if len(content) > MAX_PDF_CHARACTERS:
        raise PdfFailure("PDF_TOO_MUCH_TEXT")
    empty = [number for number, text in enumerate(pages, 1) if not text]
    if len(empty) == len(pages):
        raise PdfFailure("PDF_NO_TEXT")
    return {"sourceHash": hashlib.sha256(data).hexdigest(), "pages": len(pages),
            "content": content, "emptyPages": empty, "parser": parser}


def mupdf_text(data: bytes, expected_pages: int) -> dict:
    # Reuse the original RAG project's local extractor. This is text extraction, not
    # OCR, and never drops/replaces undecodable glyphs to force a success.
    import pymupdf
    document = pymupdf.open(stream=data, filetype="pdf")
    pages, total = [], 0
    try:
        if len(document) != expected_pages:
            raise PdfFailure("PDF_INVALID")
        for number in range(expected_pages):
            page = document[number]
            text = clean_text(page.get_text("text", sort=False))
            total += len(text) + (len(PDF_PAGE_SEPARATOR) if pages else 0)
            if total > MAX_PDF_CHARACTERS:
                raise PdfFailure("PDF_TOO_MUCH_TEXT")
            pages.append(text)
    finally:
        document.close()
    return receipt(data, pages, "pymupdf-text-v1")


def parse(data: bytes) -> dict:
    if not data.startswith(b"%PDF-") or len(data) > MAX_PDF_BYTES:
        raise PdfFailure("PDF_INVALID")
    from pypdf import PdfReader
    with io.BytesIO(data) as stream:
        reader = PdfReader(stream, strict=True)
        if reader.is_encrypted:
            raise PdfFailure("PDF_ENCRYPTED")
        count = len(reader.pages)
        if not 1 <= count <= MAX_PDF_PAGES:
            raise PdfFailure("PDF_TOO_MANY_PAGES")
        pages, total, fallback = [], 0, False
        for number, page in enumerate(reader.pages, 1):
            # A compressed content stream can be much larger than the upload.
            # The subprocess memory/CPU limits bound decompression as well.
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 4 * 1024 * 1024:
                raise PdfFailure("PDF_CONTENT_STREAM_TOO_LARGE")
            # Even after finding unsupported text, validate every subsequent
            # decompressed stream before allowing the alternate extractor.
            if fallback:
                continue
            try:
                text = clean_text(page.extract_text() or "")
            except (PdfFailure, UnicodeError):
                fallback = True
                continue
            total += len(text) + (len(PDF_PAGE_SEPARATOR) if pages else 0)
            if total > MAX_PDF_CHARACTERS:
                raise PdfFailure("PDF_TOO_MUCH_TEXT")
            pages.append(text)
        if not fallback:
            return receipt(data, pages, "pypdf-6.21.1")
    return mupdf_text(data, count)


def main():
    if sys.platform.startswith("linux"):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    result = None
    try:
        data = sys.stdin.buffer.read(MAX_PDF_BYTES + 1)
        # Parser diagnostics must not reveal input fragments/metadata to logs.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            result = parse(data)
    except PdfFailure as error:
        result = {"code": error.code}
    except ImportError:
        result = {"code": "PDF_PARSER_UNAVAILABLE"}
    except MemoryError:
        result = {"code": "PDF_RESOURCE_LIMIT"}
    except Exception:
        result = {"code": "PDF_INVALID"}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


if __name__ == "__main__":
    main()
