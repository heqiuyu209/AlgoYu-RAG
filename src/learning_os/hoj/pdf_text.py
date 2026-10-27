"""Canonical extracted PDF text. The original binary stays in HOJ, not here."""

MAX_PDF_BYTES = 32 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_PDF_CHARACTERS = 300000
MAX_PDF_JSON_BYTES = 2 * 1024 * 1024
MAX_PDF_INGEST_BYTES = 4 * 1024 * 1024
NATIVE_PDF_PARSERS = frozenset({"pypdf-6.21.1", "pymupdf-text-v1"})
MAX_PDF_INGEST_MS = 20 * 60 * 1000
PDF_PAGE_SEPARATOR = "\n\f\n"


def split_pdf_pages(content: str) -> list[str]:
    pages = content.split(PDF_PAGE_SEPARATOR)
    if (not 1 <= len(pages) <= MAX_PDF_PAGES or len(content) > MAX_PDF_CHARACTERS
            or not any(p.strip() for p in pages) or any("\f" in p or "\x00" in p or "\r" in p for p in pages)):
        raise ValueError("Invalid canonical PDF text")
    return pages
