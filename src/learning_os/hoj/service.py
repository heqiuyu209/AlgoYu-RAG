"""Internal HTTP transport. Never loads provider keys or the personal .env.

One worker owns the local Qdrant directory. HOJ owns the durable job queue;
HTTP ingestion is synchronous to that worker, asynchronous to the browser.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from learning_os.hoj.auth import RequestAuth, check_deadline
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef, VideoAsset, VideoSlice
from learning_os.hoj.video_extract import VideoExtractor
from learning_os.hoj.index import KnowledgeIndex
from learning_os.hoj.pdf_extract import extract_pdf
from learning_os.hoj.pdf_text import MAX_PDF_BYTES, MAX_PDF_INGEST_BYTES

logger = logging.getLogger(__name__)
MAX_BODY = 1048576


def service_key() -> str:
    value = os.environ.get("HOJ_RAG_SHARED_KEY", "")
    file = os.environ.get("HOJ_RAG_SHARED_KEY_FILE", "")
    if value and file:
        raise ValueError("Configure one service identity source, not both")
    if file:
        try:
            path = Path(file)
            if path.stat().st_size > 128:
                raise ValueError()
            return path.read_text(encoding="ascii").strip()
        except Exception:
            raise ValueError("Internal service identity file unavailable") from None
    return value


def create_app(index: KnowledgeIndex, auth: RequestAuth, video: VideoExtractor | None = None) -> FastAPI:
    video = video or VideoExtractor.from_env()
    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            index.client.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    # At most one HTTP ingestion can occupy a worker slot. Reserve the other
    # for retrieval/deletion while bounded document batches yield the model.
    capacity = threading.BoundedSemaphore(2)
    ingest_capacity = threading.BoundedSemaphore(1)
    pdf_capacity = threading.BoundedSemaphore(1)
    video_capacity = threading.BoundedSemaphore(1)

    @app.exception_handler(ServiceError)
    async def service_error(_request, error):
        return JSONResponse({"code": error.code}, status_code=error.status)

    @app.middleware("http")
    async def authenticated_body(request: Request, call_next):
        if request.method != "POST" or request.url.path not in {
                "/internal/v1/ingest", "/internal/v1/retrieve", "/internal/v1/delete", "/internal/v1/extract-pdf", "/internal/v1/extract-pdf-ocr",
                "/internal/v1/probe-video", "/internal/v1/transcribe-video"}:
            return JSONResponse({"code": "NOT_FOUND"}, status_code=404)
        pdf_request = request.url.path in {"/internal/v1/extract-pdf", "/internal/v1/extract-pdf-ocr"}
        expected_type = "application/pdf" if pdf_request else "application/json"
        if request.url.query or request.headers.get("content-type", "").split(";")[0].strip() != expected_type:
            return JSONResponse({"code": "INVALID_REQUEST"}, status_code=400)
        raw = bytearray()
        limit = MAX_PDF_BYTES if pdf_request else MAX_PDF_INGEST_BYTES if request.url.path == "/internal/v1/ingest" else MAX_BODY
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > limit:
                return JSONResponse({"code": "BODY_TOO_LARGE"}, status_code=413)
        try:
            request.state.deadline = auth.verify(request.url.path, request.headers, bytes(raw))
            request.state.body = bytes(raw)
        except ServiceError as error:
            return JSONResponse({"code": error.code}, status_code=error.status)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def run(request, contract, action):
        try:
            body = contract.model_validate_json(request.state.body)
        except ValidationError:
            # Pydantic's detailed error includes input excerpts. Never reflect it.
            raise ServiceError(400, "INVALID_CONTRACT") from None
        if isinstance(body, Retrieve) and not body.allowedVersions:
            check_deadline(request.state.deadline)
            return {"items": []}
        ingest_slot = isinstance(body, Ingest)
        if ingest_slot and not ingest_capacity.acquire(blocking=False):
            raise ServiceError(429, "SERVICE_BUSY")
        try:
            if not capacity.acquire(blocking=False):
                raise ServiceError(429, "SERVICE_BUSY")
            try:
                check_deadline(request.state.deadline)
                return action(body, request.state.deadline)
            except ServiceError:
                raise
            except Exception as error:
                # Upstream exceptions can contain paths, snippets or HTTP bodies.
                logger.warning("HOJ RAG operation failed (%s)", type(error).__name__)
                raise ServiceError(503, "INDEX_UNAVAILABLE") from None
            finally:
                capacity.release()
        finally:
            if ingest_slot:
                ingest_capacity.release()

    @app.post("/internal/v1/ingest")
    def ingest(request: Request):
        return run(request, Ingest, index.ingest)

    @app.post("/internal/v1/retrieve")
    def retrieve(request: Request):
        return run(request, Retrieve, index.retrieve)

    @app.post("/internal/v1/delete")
    def delete(request: Request):
        return run(request, VersionRef, index.delete)

    @app.post("/internal/v1/extract-pdf")
    @app.post("/internal/v1/extract-pdf-ocr")
    def pdf(request: Request):
        if not pdf_capacity.acquire(blocking=False):
            raise ServiceError(429, "SERVICE_BUSY")
        try:
            if request.url.path.endswith("-ocr"):
                return extract_pdf(request.state.body, request.state.deadline, ocr=True)
            return extract_pdf(request.state.body, request.state.deadline)
        finally:
            pdf_capacity.release()

    @app.post("/internal/v1/probe-video")
    @app.post("/internal/v1/transcribe-video")
    def media(request: Request):
        contract = VideoSlice if request.url.path.endswith("/transcribe-video") else VideoAsset
        try:
            body = contract.model_validate_json(request.state.body)
        except ValidationError:
            raise ServiceError(400,"INVALID_CONTRACT") from None
        if not video_capacity.acquire(blocking=False):
            raise ServiceError(429,"SERVICE_BUSY")
        try:
            return video.extract(body,request.state.deadline)
        except ServiceError:
            raise
        except Exception as error:
            logger.warning("HOJ video operation failed (%s)",type(error).__name__)
            raise ServiceError(503,"VIDEO_ASR_UNAVAILABLE") from None
        finally:
            video_capacity.release()

    return app


def from_env() -> FastAPI:
    """Uvicorn factory: no remote model names, URLs, or auto-download fallback."""
    retrieval_mode = os.environ.get("HOJ_RAG_RETRIEVAL_MODE", "dense")
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("HOJ_RAG_RETRIEVAL_MODE must be dense or hybrid")
    rerank_path = os.environ.get("HOJ_RAG_RERANK_MODEL_PATH", "")
    rerank_engine = os.environ.get("HOJ_RAG_RERANK_ENGINE", "pytorch")
    graph_path = os.environ.get("HOJ_RAG_RERANK_GRAPH_PATH", "")
    context_projection = os.environ.get("HOJ_RAG_CONTEXT_PROJECTION", "0")
    if rerank_engine not in {"pytorch", "onnx"}:
        raise ValueError("HOJ_RAG_RERANK_ENGINE must be pytorch or onnx")
    if context_projection not in {"0", "1"}:
        raise ValueError("HOJ_RAG_CONTEXT_PROJECTION must be 0 or 1")
    if rerank_engine == "onnx" and (not rerank_path or not graph_path):
        raise ValueError("ONNX requires local HOJ_RAG_RERANK_MODEL_PATH and HOJ_RAG_RERANK_GRAPH_PATH")
    if graph_path and rerank_engine != "onnx":
        raise ValueError("HOJ_RAG_RERANK_GRAPH_PATH requires the onnx engine")
    if context_projection == "1" and (retrieval_mode != "dense" or rerank_engine != "onnx"):
        raise ValueError("Context projection requires dense retrieval and the pinned ONNX reranker")
    if rerank_path and retrieval_mode != "dense":
        raise ValueError("Passage reranking requires dense retrieval")
    from qdrant_client import QdrantClient
    from learning_os.infrastructure.embedders.bge_m3_embedder import BGEM3Embedder

    directory = Path(os.environ.get("HOJ_RAG_DATA", "data/hoj")).resolve()
    auth = RequestAuth(service_key(), directory / "state.sqlite3")
    model = Path(os.environ.get("HOJ_RAG_MODEL_PATH", "")).resolve()
    if not os.environ.get("HOJ_RAG_MODEL_PATH") or not model.is_dir():
        raise ValueError("HOJ_RAG_MODEL_PATH must point to an installed local BGE-M3 model directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    reranker = None
    if rerank_path:
        if rerank_engine == "onnx":
            from learning_os.hoj.onnx_reranker import OnnxPassageReranker
            reranker = OnnxPassageReranker(Path(rerank_path).resolve(), Path(graph_path).absolute())
        else:
            from learning_os.hoj.reranker import PassageReranker
            reranker = PassageReranker(Path(rerank_path).resolve())
    embedder = BGEM3Embedder(model_name=str(model), device=os.environ.get("HOJ_RAG_DEVICE", "cpu"), batch_size=8)
    # A cold model load must finish before the service accepts timed requests.
    embedder.embed_query("算法学习")
    if reranker is not None:
        from learning_os.hoj.auth import now_ms
        from learning_os.hoj.runtime import inference_owner
        reranker.score("What is a prefix sum?", ["A prefix sum accumulates an array."],
                       now_ms() + 120000, inference_owner(embedder))
    client = QdrantClient(path=str(directory / "qdrant"))
    try:
        if context_projection == "1":
            from learning_os.hoj.dual_index import ContextIndex, CONTEXT_PROJECTION_REVISION
            from learning_os.hoj.paired_index import PairedKnowledgeIndex
            revision = CONTEXT_PROJECTION_REVISION
            context = ContextIndex(client, embedder, directory / f"context-r{revision}.sqlite3",
                collection=f"hoj_knowledge_context_r{revision}")
            index = PairedKnowledgeIndex(client, embedder, directory / "state.sqlite3",
                context=context, retrieval_mode=retrieval_mode, reranker=reranker)
        else:
            index = KnowledgeIndex(client, embedder, directory / "state.sqlite3",
                retrieval_mode=retrieval_mode, reranker=reranker)
        return create_app(index, auth)
    except BaseException:
        client.close()
        raise
