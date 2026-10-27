"""Explicit offline integration: Java -> signed HTTP -> local BGE-M3/Qdrant.

Owns its Uvicorn thread, temporary key/index and Maven child; waits for shutdown
before cleanup. No existing HOJ services, .env or model provider are accessed.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hoj-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be an installed local BGE-M3 directory")
    pom = args.hoj_root / "HOJ-master/hoj-springboot/pom.xml"
    if not pom.is_file():
        parser.error("--hoj-root must be the HOJ Git root")
    maven = shutil.which("mvn.cmd" if os.name == "nt" else "mvn")
    if not maven:
        parser.error("Maven must be installed")

    import torch
    import uvicorn
    from qdrant_client import QdrantClient
    from learning_os.hoj.auth import RequestAuth
    from learning_os.hoj.index import KnowledgeIndex
    from learning_os.hoj.service import create_app
    from learning_os.infrastructure.embedders.bge_m3_embedder import BGEM3Embedder

    torch.set_num_threads(4)
    report = {"validated": False, "cleaned": False, "embedding": "LOCAL_BGE_M3", "externalModelCalls": 0}
    with tempfile.TemporaryDirectory(prefix="hoj-rag-bridge-") as temporary:
        directory = Path(temporary)
        key = secrets.token_hex(32)
        key_file = directory / "service.key"
        key_file.write_text(key, encoding="ascii")
        embedder = BGEM3Embedder(model_name=str(args.model.resolve()), device="cpu", batch_size=8)
        # Warm up locally; a model import isn't a retrieval request deadline.
        assert len(embedder.embed_query("算法学习")) == 1024
        store = QdrantClient(path=str(directory / "qdrant"))
        index = KnowledgeIndex(store, embedder, directory / "state.sqlite3")
        app = create_app(index, RequestAuth(key, directory / "state.sqlite3"))
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False,
                                               timeout_graceful_shutdown=5, timeout_keep_alive=1))
        thread = threading.Thread(target=server.run, name="isolated-hoj-rag", daemon=True)
        thread.start()
        try:
            limit = time.monotonic() + 15
            while not server.started and thread.is_alive() and time.monotonic() < limit:
                time.sleep(0.05)
            if not server.started:
                raise RuntimeError("isolated RAG server did not start")
            env = os.environ.copy()
            env["HOJ_RAG_PROBE_URL"] = f"http://127.0.0.1:{port}"
            env["HOJ_RAG_PROBE_KEY_FILE"] = str(key_file)
            command = [maven, "-o", "-f", str(pom), "-pl", "DataBackup", "-am", "test",
                       "-Dtest=RagServiceClientTest,RagServiceLocalProbe", "-Dsurefire.failIfNoSpecifiedTests=false",
                       "-DskipTests=false", "-Dlombok.version=1.18.46"]
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("wb") as log:
                result = subprocess.run(command, cwd=args.hoj_root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError(f"Java probe failed; inspect {args.log}")
            # Require real testcases: BUILD SUCCESS alone can hide a lost fork.
            reports = args.hoj_root / "HOJ-master/hoj-springboot/DataBackup/target/surefire-reports"
            total = 0
            for name, expected in (("RagServiceClientTest", 12), ("RagServiceLocalProbe", 1)):
                suite = ET.parse(reports / f"TEST-top.hcode.hoj.manager.oj.ai.{name}.xml").getroot()
                assert int(suite.attrib["tests"]) == expected and len(suite.findall("testcase")) == expected
                assert all(int(suite.attrib.get(field, 0)) == 0 for field in ("failures", "errors", "skipped"))
                total += expected
            report.update(validated=True, javaTests=total, localIndexPoints=store.count(index.collection).count)
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            if thread.is_alive():
                faulthandler.dump_traceback()
                raise RuntimeError("isolated RAG server did not stop")
            store.close()
    report["cleaned"] = True
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
