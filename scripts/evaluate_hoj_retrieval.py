"""Explicit real local BGE-M3 evaluation. No live HOJ DB/index or provider IO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dataset_file(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or PureWindowsPath(relative).drive:
        raise ValueError("Invalid dataset path")
    path = directory / relative.replace("\\", "/")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Dataset file is outside its directory")
    return path


def load_dataset(directory: Path):
    from learning_os.hoj.contracts import Ingest
    from learning_os.hoj.index import KnowledgeIndex
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    freeze = directory / "freeze.json"
    verified = set()
    if freeze.is_file():
        for relative, expected in json.loads(freeze.read_text(encoding="utf-8"))["files"].items():
            source = dataset_file(directory, relative)
            if sha(source.read_bytes()) != expected:
                raise ValueError("Frozen dataset changed: " + relative)
            verified.add(source.resolve())
    manifest = directory / "documents.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("Invalid document manifest")
    else:
        # Preserve existing Markdown-only datasets and their stable identities.
        entries = [{"id": path.stem, "title": path.read_text(encoding="utf-8").splitlines()[0].removeprefix("# "),
                    "format": "markdown", "file": "documents/" + path.name}
                   for path in sorted((directory / "documents").glob("*.md"))]
    docs = {}
    required = {directory / "cases.json"}
    if manifest.is_file():
        required.add(manifest)
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {"id", "title", "format", "file"}
                or not isinstance(entry["id"], str) or not re.fullmatch(r"[a-z0-9_-]{1,80}", entry["id"])
                or entry["id"] in docs):
            raise ValueError("Invalid or duplicate document identity")
        path = dataset_file(directory, entry["file"])
        required.add(path)
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        docs[entry["id"]] = Ingest(documentId=sha(("hoj-eval-v1:" + entry["id"]).encode())[:32], version=1,
                                   contentHash=sha(text.encode()), title=entry["title"],
                                   format=entry["format"], content=text)
    if freeze.is_file() and not {path.resolve() for path in required} <= verified:
        raise ValueError("Unfrozen dataset input")
    if not cases or not docs or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty dataset or duplicate question IDs")
    # Validate explicit labels against the same production parser/chunker before
    # measuring. This does not embed or edit the data in response to a miss.
    chunker = object.__new__(KnowledgeIndex)
    from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker
    chunker.chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=150)
    chunks = {slug: chunker.chunks(doc) for slug, doc in docs.items()}
    for case in cases:
        if case["split"] not in {"calibration", "heldout"} or not 0 < len(case["query"]) <= 2000:
            raise ValueError("Invalid split/question: " + case["id"])
        scope = case.get("documents", list(docs))
        if not isinstance(scope, list) or any(d not in docs for d in scope) or len(scope) != len(set(scope)):
            raise ValueError("Invalid allowed document scope: " + case["id"])
        for label in case["relevant"]:
            if label["document"] not in scope or not label["evidence"] or any(not s for s in label["evidence"]):
                raise ValueError("Invalid ground truth: " + case["id"])
            if not any(c.metadata.chapter.split(" / ")[-1] == label["heading"]
                       and all(span in c.content for span in label["evidence"])
                       for c in chunks[label["document"]]):
                raise ValueError("Ground truth has no complete source chunk: " + case["id"])
        if not case["relevant"] and (not case.get("reason") or case.get("category") not in {"near", "far", "scope"}):
            raise ValueError("Unsupported questions need an explicit rationale: " + case["id"])
    return docs, cases, chunks


def citation_location(document, hit: dict) -> dict:
    """Validate locations against the source, separately from ranking labels.

    These are source-text checks, not a claim that PDF/ASR extraction is accurate
    or that a generated answer is entailed by its citations.
    """
    if document.format == "video":
        from learning_os.hoj.video_text import VideoText
        start, end = VideoText.parse(document.content).citation(hit["headingPath"], hit["excerpt"])
        return {"startMs": start, "endMs": end}
    if document.format == "pdf":
        from learning_os.hoj.pdf_text import split_pdf_pages
        match = re.fullmatch(r"第 ([1-9][0-9]{0,2}) 页", hit["headingPath"])
        if not match:
            raise ValueError("Invalid PDF citation page")
        page = int(match.group(1))
        pages = split_pdf_pages(document.content)
        if page > len(pages) or not hit["excerpt"].strip() or hit["excerpt"] not in pages[page - 1]:
            raise ValueError("PDF citation is not on that physical page")
        return {"page": page}
    if not hit["excerpt"].strip() or hit["excerpt"] not in document.content:
        raise ValueError("Citation is not in the source")
    return {}


def format_summaries(cases: list[dict], measured: list[dict], docs: dict) -> dict:
    from learning_os.hoj.evaluation import measure, summarize
    by_id = {row["id"]: row for row in measured}
    result = {}
    for format in sorted({doc.format for doc in docs.values()}):
        rows = []
        for case in cases:
            labels = [label for label in case["relevant"] if docs[label["document"]].format == format]
            if labels:
                rows.append(measure(case | {"relevant": labels}, by_id[case["id"]]["hits"]))
        result[format] = {split: summarize([row for row in rows if split == "all" or row["split"] == split])
                          for split in ("all", "calibration", "heldout")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "evaluation/hoj-xcpc-v1")
    parser.add_argument("--min-recall-at-3", type=float, default=0.90)
    parser.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="dense")
    args = parser.parse_args()
    if not args.model.is_dir() or not math.isfinite(args.min_recall_at_3) or not 0 <= args.min_recall_at_3 <= 1:
        parser.error("Use an installed local model directory and a recall gate in [0,1]")
    report = {"executionComplete": False, "cleaned": False, "retrievalGatePassed": False,
              "semanticAnswerValidation": "NOT_RUN_NO_CHAT_MODEL", "runtimeThresholdChanged": False,
              "externalModelCalls": 0, "liveDatabaseOrIndexAccessed": False,
              "embedding": "LOCAL_BGE_M3", "modelDirectory": str(args.model.resolve()), "cases": []}
    report["retrievalMode"] = args.retrieval_mode
    status, temporary = 1, None
    started = time.monotonic()
    try:
        print("Preparing offline Python/model runtime; no provider or live index is used.", flush=True)
        import torch
        from qdrant_client import QdrantClient
        from learning_os.hoj.auth import now_ms
        from learning_os.hoj.contracts import Retrieve, VersionRef
        from learning_os.hoj.evaluation import measure, summarize, threshold_study
        from learning_os.hoj.index import KnowledgeIndex
        from learning_os.infrastructure.embedders.bge_m3_embedder import BGEM3Embedder
        docs, cases, chunks = load_dataset(args.dataset)
        report["dataset"] = {"version": args.dataset.name,
                             "casesSha256": sha((args.dataset / "cases.json").read_bytes()),
                             "documents": {slug: doc.contentHash for slug, doc in docs.items()},
                             "documentContracts": {slug: {"title": doc.title, "format": doc.format} for slug, doc in docs.items()},
                             "manifestSha256": sha((args.dataset / "documents.json").read_bytes())
                             if (args.dataset / "documents.json").is_file() else None}
        report["runtimeVersions"] = {p: importlib.metadata.version(p) for p in ("torch", "sentence-transformers", "qdrant-client")}
        source_root = Path(__file__).resolve().parents[1]
        report["implementationHashes"] = {name: sha((source_root / name).read_bytes()) for name in (
            "scripts/evaluate_hoj_retrieval.py", "src/learning_os/hoj/index.py", "src/learning_os/hoj/evaluation.py", "src/learning_os/hoj/lexical.py",
            "src/learning_os/infrastructure/embedders/bge_m3_embedder.py",
            "src/learning_os/infrastructure/chunkers/recursive_chunker.py")}
        report["modelConfiguration"] = {name: sha((args.model / name).read_bytes())
                                         for name in ("config.json", "modules.json", "config_sentence_transformers.json")
                                         if (args.model / name).is_file()}
        torch.set_num_threads(4)
        embedder = BGEM3Embedder(model_name=str(args.model.resolve()), device="cpu", batch_size=8)
        assert len(embedder.embed_query("算法学习")) == 1024
        print(f"Loaded local model. Fixed documents={len(docs)}, queries={len(cases)}", flush=True)
        with tempfile.TemporaryDirectory(prefix="hoj-retrieval-eval-") as directory:
            temporary = Path(directory)
            client = QdrantClient(path=str(temporary / "qdrant"))
            try:
                index = KnowledgeIndex(client, embedder, temporary / "state.sqlite3", retrieval_mode=args.retrieval_mode)
                sources = {doc.documentId: (slug, doc) for slug, doc in docs.items()}
                for slug, doc in docs.items():
                    index.ingest(doc, now_ms() + 120000)
                report["indexedChunks"] = client.count(index.collection, exact=True).count
                report["chunksByDocument"] = {slug: len(part) for slug, part in chunks.items()}
                for number, case in enumerate(cases, 1):
                    refs = [VersionRef(**{k: getattr(docs[slug], k) for k in ("documentId", "version", "contentHash")})
                            for slug in case.get("documents", list(docs))]
                    query_started = time.monotonic()
                    found = index.retrieve(Retrieve(query=case["query"], allowedVersions=refs, topK=5), now_ms() + 120000)
                    hits = []
                    for item in found["items"]:
                        slug, doc = sources[item["documentId"]]
                        assert slug in case.get("documents", list(docs))
                        assert item["version"] == doc.version and item["contentHash"] == doc.contentHash
                        assert sha(item["excerpt"].encode()) == item["chunkHash"]
                        assert item["chunkId"] == sha(f"{index.namespace}:{doc.key()}:{item['chunkIndex']}:{item['chunkHash']}".encode())
                        original = chunks[slug][item["chunkIndex"]]
                        assert original.content == item["excerpt"] and original.metadata.chapter == item["headingPath"]
                        hits.append({"document": slug, "headingPath": item["headingPath"],
                                     "excerpt": item["excerpt"], "score": item["score"], "chunkId": item["chunkId"],
                                     "format": doc.format, **citation_location(doc, item)})
                    measured = measure(case, hits)
                    measured["elapsedMs"] = round((time.monotonic() - query_started) * 1000)
                    report["cases"].append(measured)
                    if number % 6 == 0 or number == len(cases):
                        print(f"Evaluated {number}/{len(cases)}", flush=True)
                for split in ("all", "calibration", "heldout"):
                    report[split] = summarize([r for r in report["cases"] if split == "all" or r["split"] == split])
                report["byFormat"] = format_summaries(cases, report["cases"], docs)
                report["sourceLocationsVerified"] = True
                report["thresholdStudy"] = threshold_study(report["cases"])
                report["integrityPassed"] = True
                report["executionComplete"] = True
                report["retrievalGate"] = {"minEvidenceRecallAt3": args.min_recall_at_3, "requiredSplits": ["all", "heldout"]}
                report["retrievalGatePassed"] = all(report[split]["evidenceRecall"]["3"] >= args.min_recall_at_3
                                                    for split in ("all", "heldout"))
                status = 0 if report["retrievalGatePassed"] else 2
            finally:
                client.close()
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["elapsedSeconds"] = round(time.monotonic() - started, 2)
        report["cleaned"] = temporary is not None and not temporary.exists()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: report.get(k) for k in ("executionComplete", "cleaned", "retrievalGatePassed", "all", "heldout", "failure")},
                         ensure_ascii=False, indent=2), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
