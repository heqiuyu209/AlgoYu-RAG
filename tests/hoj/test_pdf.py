"""PDF bytes are synthetic in-memory fixtures, never user documents."""

import io
import hashlib
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from qdrant_client import QdrantClient

from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.hoj.index import KnowledgeIndex, digest
from learning_os.hoj.pdf_extract import extract_pdf
from learning_os.hoj.pdf_text import MAX_PDF_BYTES, PDF_PAGE_SEPARATOR
from learning_os.hoj.pdf_worker import parse, PdfFailure
from learning_os.hoj.pdf_ocr_worker import parse as parse_ocr, image_size, OCR_PARSER


def pdf_bytes(texts=("Prefix sum: S[r]-S[l-1].", "Dijkstra requires nonnegative weights."), encrypted=False):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    font_ref = writer._add_object(font)
    for text in texts:
        page = writer.add_blank_page(width=595, height=842)
        if text:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(("BT /F1 12 Tf 60 750 Td (" + escaped + ") Tj ET").encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
    if encrypted:
        writer.encrypt("fixture-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_real_pdf_worker_preserves_original_hash_and_physical_page_numbers():
    data = pdf_bytes(("Prefix sum: S[r]-S[l-1].", "", "Dijkstra requires nonnegative weights."))
    result = extract_pdf(data, now_ms() + 60000)
    assert result["sourceHash"] == hashlib.sha256(data).hexdigest()
    assert result["pages"] == 3 and result["emptyPages"] == [2]
    assert result["content"].split(PDF_PAGE_SEPARATOR) == ["Prefix sum: S[r]-S[l-1].", "", "Dijkstra requires nonnegative weights."]


@pytest.mark.parametrize("data,code", [(b"not-a-pdf", "PDF_INVALID"), (pdf_bytes(("", "")), "PDF_NO_TEXT"),
                                      (pdf_bytes(encrypted=True), "PDF_ENCRYPTED"),
                                      (pdf_bytes(tuple("" for _ in range(201))), "PDF_TOO_MANY_PAGES")])
def test_rejected_sources_return_fixed_codes_without_text_or_password(data, code):
    with pytest.raises(ServiceError) as error:
        extract_pdf(data, now_ms() + 60000)
    assert error.value.code == code
    assert "fixture-password" not in str(error.value)


def test_byte_limit_and_deadline_fail_before_spawning_a_parser():
    with patch("learning_os.hoj.pdf_extract.subprocess.Popen") as child:
        with pytest.raises(ServiceError):
            extract_pdf(b"%PDF-" + b"x" * MAX_PDF_BYTES, now_ms() + 60000)
        with pytest.raises(ServiceError):
            extract_pdf(pdf_bytes(), now_ms() - 1)
        child.assert_not_called()


def test_worker_timeout_is_killed_reaped_and_does_not_inherit_service_identity(monkeypatch):
    import subprocess
    from unittest.mock import MagicMock
    child = MagicMock()
    child.communicate.side_effect = [subprocess.TimeoutExpired("fixture", 1), (b"", b"")]
    child.poll.return_value = -9
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY", "synthetic-must-not-reach-parser")
    with patch("learning_os.hoj.pdf_extract.subprocess.Popen", return_value=child) as spawn:
        with pytest.raises(ServiceError) as error:
            extract_pdf(pdf_bytes(), now_ms() + 60000)
        assert error.value.code == "PDF_RESOURCE_LIMIT"
        assert "HOJ_RAG_SHARED_KEY" not in spawn.call_args.kwargs["env"]
    child.kill.assert_called_once()
    child.stdin.close.assert_called_once()
    child.stdout.close.assert_called_once()


def test_expanded_content_stream_is_bounded_inside_the_worker():
    with pytest.raises(PdfFailure) as error:
        parse(pdf_bytes(("x" * (4 * 1024 * 1024 + 1),)))
    assert error.value.code == "PDF_CONTENT_STREAM_TOO_LARGE"


@pytest.mark.parametrize("text", ["unmapped\x00text", "unmapped\udcfftext"])
def test_unmapped_font_text_fails_without_returning_invalid_unicode(text):
    from types import SimpleNamespace
    page = SimpleNamespace(get_contents=lambda: None, extract_text=lambda: text)
    with patch("pypdf.PdfReader", return_value=SimpleNamespace(is_encrypted=False, pages=[page])):
        with patch("learning_os.hoj.pdf_worker.mupdf_text", side_effect=PdfFailure("PDF_UNSUPPORTED_TEXT")) as fallback:
            with pytest.raises(PdfFailure) as error:
                parse(b"%PDF-fixture")
            fallback.assert_called_once_with(b"%PDF-fixture", 1)
    assert error.value.code == "PDF_UNSUPPORTED_TEXT"


def test_unmapped_pypdf_font_uses_complete_strict_native_mupdf_extraction():
    data = pdf_bytes(("First original page.", "", "Last original page."))
    with patch("pypdf._page.PageObject.extract_text", return_value="bad\0font"):
        result = parse(data)
    assert result["parser"] == "pymupdf-text-v1"
    assert result["pages"] == 3 and result["emptyPages"] == [2]
    assert result["sourceHash"] == hashlib.sha256(data).hexdigest()
    assert result["content"] == "First original page.\n\f\n\n\f\nLast original page."


def test_native_fallback_cannot_bypass_later_content_stream_guard():
    data = pdf_bytes(("Unmapped first page.", "x" * (4 * 1024 * 1024 + 1)))
    with patch("pypdf._page.PageObject.extract_text", return_value="bad\0font"):
        with patch("learning_os.hoj.pdf_worker.mupdf_text") as fallback:
            with pytest.raises(PdfFailure) as error:
                parse(data)
            fallback.assert_not_called()
    assert error.value.code == "PDF_CONTENT_STREAM_TOO_LARGE"


def test_whole_book_result_body_accepts_more_than_one_mib_without_invalid_unicode():
    from unittest.mock import MagicMock
    import json
    data = pdf_bytes()
    receipt = dict(sourceHash=hashlib.sha256(data).hexdigest(), pages=1,
                   content="🙂" * 270001, emptyPages=[], parser="pymupdf-text-v1")
    child = MagicMock(returncode=0)
    child.poll.return_value = 0
    raw = json.dumps(receipt, ensure_ascii=False).encode()
    assert len(raw) > 1048576
    child.communicate.return_value = (raw, b"")
    with patch("learning_os.hoj.pdf_extract.subprocess.Popen", return_value=child):
        assert extract_pdf(data, now_ms() + 60000) == receipt


def test_pdf_page_text_is_literal_even_if_it_looks_like_markdown(tmp_path):
    class Embedding:
        dimension = 3
        def embed(self, texts): return [[1.0, 0.0, 0.0] for _ in texts]
        def embed_query(self, query): return [1.0, 0.0, 0.0]
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, Embedding(), tmp_path / "pdf.sqlite3")
        content = "# Not a section override\nFirst physical page" + PDF_PAGE_SEPARATOR + "" + PDF_PAGE_SEPARATOR + "Last physical page"
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="PDF fixture", format="pdf", content=content)
        assert index.ingest(doc, now_ms() + 60000)["chunks"] == 2
        ref = VersionRef(documentId=doc.documentId, version=1, contentHash=doc.contentHash)
        items = index.retrieve(Retrieve(query="page", allowedVersions=[ref]), now_ms() + 60000)["items"]
        assert {item["headingPath"] for item in items} == {"第 1 页", "第 3 页"}
        assert all(item["excerpt"] in content for item in items)
    finally:
        client.close()


def test_ocr_renders_every_page_even_with_existing_text_and_keeps_empty_physical_page(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    # Real PDFium rendering, only the recognizer is a deterministic fixture.
    data = pdf_bytes(("Header on mixed page", "", "Native text on final page"))
    with patch("learning_os.hoj.pdf_ocr_worker.recognize", side_effect=["页眉与图片中的算法正文", "", "最后一页正文"]) as recognize:
        result = parse_ocr(data)
    assert recognize.call_count == 3  # Existing text must not make a mixed page skip OCR.
    assert result == {"sourceHash": hashlib.sha256(data).hexdigest(), "pages": 3,
                      "content": "页眉与图片中的算法正文\n\f\n\n\f\n最后一页正文", "emptyPages": [2], "parser": OCR_PARSER}


@pytest.mark.parametrize("size", [(float("inf"), 500), (0, 842), (-1, 842), (100000, 842), (2000, 2000)])
def test_ocr_rejects_unbounded_page_before_raster_allocation(size):
    with pytest.raises(PdfFailure):
        image_size(*size)


def test_ocr_empty_results_and_encryption_do_not_return_partial_success(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with patch("learning_os.hoj.pdf_ocr_worker.recognize", return_value=""):
        with pytest.raises(PdfFailure) as error:
            parse_ocr(pdf_bytes(("header", "")))
        assert error.value.code == "PDF_NO_TEXT"
    with patch("learning_os.hoj.pdf_ocr_worker.recognize") as recognize:
        with pytest.raises(PdfFailure) as error:
            parse_ocr(pdf_bytes(encrypted=True))
        assert error.value.code == "PDF_ENCRYPTED"
        recognize.assert_not_called()


def test_ocr_parent_bounds_and_kills_the_whole_process_group(monkeypatch):
    import subprocess
    from unittest.mock import MagicMock
    child = MagicMock(pid=43210)
    child.communicate.side_effect = [subprocess.TimeoutExpired("fixture", 1), (b"", b"")]
    child.poll.return_value = -9
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY", "not-for-ocr")
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-ocr")
    with patch("learning_os.hoj.pdf_extract.sys.platform", "linux"), patch("learning_os.hoj.pdf_extract.os.killpg", create=True) as kill:
        with patch("learning_os.hoj.pdf_extract.subprocess.Popen", return_value=child) as spawn:
            with pytest.raises(ServiceError) as error:
                extract_pdf(pdf_bytes(), now_ms() + 110000, ocr=True)
            assert error.value.code == "PDF_RESOURCE_LIMIT"
            assert spawn.call_args.args[0][-1].endswith("pdf_ocr_worker")
            assert spawn.call_args.kwargs["start_new_session"] is True
            assert not {"HOJ_RAG_SHARED_KEY", "OPENAI_API_KEY"} & spawn.call_args.kwargs["env"].keys()
            assert child.communicate.call_args_list[0].kwargs["timeout"] == 85.0
            assert kill.call_count >= 1


def test_ocr_receipt_is_strictly_bound_to_mode_and_source():
    from unittest.mock import MagicMock
    data = pdf_bytes()
    receipt = {"sourceHash": hashlib.sha256(data).hexdigest(), "pages": 1,
               "content": "本地识别的文字", "emptyPages": [], "parser": OCR_PARSER}
    child = MagicMock(returncode=0)
    child.poll.return_value = 0
    import json
    for mutation in ({}, {"parser": "pypdf-6.21.1"}, {"parser": "pymupdf-text-v1"}, {"sourceHash": "f" * 64}, {"emptyPages": [1]}):
        child.communicate.return_value = (json.dumps(receipt | mutation, ensure_ascii=False).encode(), b"")
        with patch("learning_os.hoj.pdf_extract.sys.platform", "win32"), patch("learning_os.hoj.pdf_extract.subprocess.Popen", return_value=child):
            if mutation:
                with pytest.raises(ServiceError) as error:
                    extract_pdf(data, now_ms() + 110000, ocr=True)
                assert error.value.code == "INVALID_PDF_RESULT"
            else:
                assert extract_pdf(data, now_ms() + 110000, ocr=True) == receipt
