# Learning OS — Technical Documentation

> **Version:** 1.0.0
> **Status:** Stable
> **Tests:** 135 passed

HOJ integration work is tracked separately in [the internal adapter guide](docs/hoj-integration.md). It does not expose the personal CLI or model configuration to HOJ users; the historical count above is not an integration test result.

---

## 1. Architecture Overview

Learning OS follows **Clean Architecture** with a **Plugin System** based on
Dependency Injection.

```
┌─────────────────────────────────────────────────┐
│                   CLI (Presentation)             │
│  los ingest | search | ask | stats | config      │
├─────────────────────────────────────────────────┤
│              Application Layer                    │
│  IngestPipeline   SearchService   AskService     │
├─────────────────────────────────────────────────┤
│                Domain Layer                       │
│  Document · Chunk · ChunkMetadata                │
│  Parser(ABC) · Chunker(ABC) · Embedder(ABC)      │
│  VectorStore(ABC) · Retriever(ABC) · LLM(ABC)    │
├─────────────────────────────────────────────────┤
│            Infrastructure Layer                   │
│  MarkdownParser · RecursiveChunker               │
│  BGEM3Embedder · QdrantStore                     │
│  VectorRetriever · OpenAICompatibleLLM · AnthropicLLM│
└─────────────────────────────────────────────────┘
```

Dependency direction: **outer layers depend inward**. Domain depends on nothing.

---

## 2. Data Pipeline

### 2.1 Ingest Flow

```
File.markdown
    → Parser.parse()           → Document (markdown content + source metadata)
    → Chunker.chunk()          → list[Chunk] (each with inherited + positional metadata)
    → Embedder.embed()         → list[Chunk] (with embedding vectors attached)
    → VectorStore.store()      → persisted in Qdrant
```

### 2.2 Search Flow

```
Query string
    → Embedder.embed_query()   → query vector
    → VectorStore.search()     → list[Chunk] (ranked by similarity)
    → display to user
```

### 2.3 Ask Flow

```
Query string
    → Retriever.retrieve()     → list[SearchResult] (context)
    → LLM.chat()               → response string
    → display to user
```

---

## 3. Plugin System

Every major component is a plugin. Adding a new plugin requires:

1. Implement the ABC from `learning_os.domain.interfaces`.
2. Register it in the appropriate `registry.py`.
3. Set the `type` in `config.yaml`.

### 3.1 Available Registries

| Registry | ABC | Default | Config Key |
|----------|-----|---------|-----------|
| `PARSER_REGISTRY` | `Parser` | `markdown` | `parser.type` |
| `CHUNKER_REGISTRY` | `Chunker` | `recursive` | `chunker.type` |
| `EMBEDDER_REGISTRY` | `Embedder` | `bge-m3` | `embedder.type` |
| `VECTOR_STORE_REGISTRY` | `VectorStore` | `qdrant` | `vector_store.type` |
| `RETRIEVER_REGISTRY` | `Retriever` | (manual injection) | `retriever.type` |
| `LLM_REGISTRY` | `LLM` | `openai-compatible` | `llm.type` |

### 3.2 Example: Adding a New Parser

```python
# src/learning_os/infrastructure/parsers/pdf_parser.py
from learning_os.domain.interfaces import Parser
from learning_os.domain.models import Document

class PDFParser(Parser):
    def parse(self, path):
        ...  # convert PDF to markdown

    def supports(self, path):
        return path.suffix.lower() == ".pdf"
```

```python
# src/learning_os/infrastructure/parsers/registry.py
PARSER_REGISTRY["pdf"] = PDFParser
```

```yaml
# config.yaml
parser:
    type: "pdf"
```

---

## 4. Configuration Reference

Full config file (`config.yaml`):

```yaml
learning_os:
    paths:
        data_dir: "./data"
        log_dir: "./logs"

    parser:
        type: "auto"        # auto-detect by extension

    chunker:
        type: "recursive"
        chunk_size: 1000
        chunk_overlap: 200
        separators: ["\n\n", "\n", "。", ". ", " "]

    embedder:
        type: "bge-m3"
        model_name: "BAAI/bge-m3"
        device: "cpu"
        batch_size: 32
        normalize: true

    vector_store:
        type: "qdrant"
        host: "localhost"
        port: 6333
        collection_name: "learning_os"
        vector_size: 1024
        distance: "Cosine"

    retriever:
        type: "vector"
        top_k: 10

    llm:
        type: "openai-compatible"
        model: "deepseek-chat"
        base_url: "https://api.deepseek.com"
        api_key_env: "DEEPSEEK_API_KEY"
        max_tokens: 4096
        temperature: 0.3
        timeout: 60.0
        max_retries: 3
        retry_delay: 1.0

    logging:
        level: "INFO"
        format: "text"
        file_enabled: true
        file_path: "./logs/learning_os.log"
```

Config resolution: `LEARNING_OS_CONFIG` env var → `./config.yaml` in CWD.

---

## 5. CLI Reference

```bash
# Ingest documents
los ingest ./notes/ --subject AI --topic Transformer

# Search
los search "attention mechanism" --top-k 5 --subject AI

# Ask with LLM
los ask "How does multi-head attention work?" --subject AI

# Stats
los stats

# Show config
los config-show
```

---

## 6. Key Design Decisions

### 6.1 SHA-256 Idempotency

`Document.id` is the SHA-256 hash of the file bytes. Re-ingesting the same
file produces the same ID → **upsert, not duplicate**. Line endings are
normalized to LF for platform independence.

### 6.2 SHA-256 → UUID Mapping

Qdrant requires UUID or u64 point IDs. First 32 hex chars of the SHA-256
are converted to a standard UUID via `uuid.UUID(hex=chunk_id[:32])`.

### 6.3 Lazy Model Loading

`BGEM3Embedder` does NOT load the 2GB model at construction time. The model
is loaded on the first call to `embed()` or `embed_query()`, so CLI startup
remains instant.

### 6.4 Payload Flattening

Qdrant filters only work on top-level payload keys. The `QdrantStore` flattens
`ChunkMetadata` fields into the payload root so that `MetadataFilter` works
correctly.

### 6.5 OpenAI-Compatible LLM

`OpenAICompatibleLLM` covers: DeepSeek, OpenRouter, SiliconFlow, vLLM, LM Studio,
Ollama, OneAPI, LocalAI — all through a single implementation with
configurable `base_url`.

### 6.6 Anthropic System Prompt

Anthropic's API takes `system` as a top-level parameter, not a message.
`AnthropicLLM` automatically extracts the first `role="system"` message and
passes it correctly.

---

## 7. Project Structure

```
learning-os/
├── pyproject.toml
├── config.yaml
├── README.md
├── src/
│   └── learning_os/
│       ├── domain/
│       │   ├── models.py          # Document, Chunk, ChunkMetadata, SearchResult
│       │   ├── interfaces.py      # All ABCs + Message + MetadataFilter
│       │   └── exceptions.py      # LearningOSError + 13 subclasses
│       ├── application/
│       │   └── ingest_pipeline.py # Parse → clean → chunk → embed → store
│       ├── infrastructure/
│       │   ├── config_loader.py   # YAML config loader
│       │   ├── logging_config.py  # Unified logging configuration
│       │   ├── container.py       # DI container
│       │   ├── parsers/
│       │   │   ├── markdown_parser.py
│       │   │   └── registry.py
│       │   ├── cleaners/
│       │   │   ├── markdown_cleaner.py
│       │   │   └── registry.py
│       │   ├── chunkers/
│       │   │   ├── recursive_chunker.py
│       │   │   └── registry.py
│       │   ├── embedders/
│       │   │   ├── bge_m3_embedder.py
│       │   │   └── registry.py
│       │   ├── vector_stores/
│       │   │   ├── qdrant_store.py
│       │   │   └── registry.py
│       │   ├── retrievers/
│       │   │   ├── vector_retriever.py
│       │   │   └── registry.py
│       │   └── llms/
│       │       ├── openai_compatible_llm.py
│       │       ├── anthropic_llm.py
│       │       └── registry.py
│       └── cli/
│           └── main.py            # Typer CLI
├── tests/
│   ├── domain/                    # Models, contracts, and exceptions
│   ├── infrastructure/
│   │   ├── test_container.py      # Configuration-driven assembly
│   │   ├── parsers/
│   │   ├── chunkers/
│   │   ├── embedders/
│   │   ├── vector_stores/
│   │   ├── retrievers/
│   │   └── llms/
│   ├── e2e/                       # Component flow, real pipeline, and CLI
│   └── benchmarks/                # run.py
└── web/
    └── app.py                     # Streamlit entry point
```

---

## 8. Developer Guide

### 8.1 Setup

```bash
pip install -e ".[dev]"
cp config.yaml.example config.yaml
# Edit config.yaml with your API keys
```

### 8.2 Running Tests

```bash
pytest                          # all tests
pytest tests/domain/            # domain only
pytest tests/e2e/               # E2E only
python tests/benchmarks/run.py  # benchmarks
```

On Windows, use the project test runner if pytest cannot access its default
temporary directory:

```powershell
.\scripts\run_tests.ps1
# Optional: select the Python interpreter explicitly.
.\scripts\run_tests.ps1 -PythonExecutable "C:\path\to\python.exe"
```

The runner uses a fresh, dedicated temporary directory for each invocation,
disables bytecode and pytest cache writes, and returns pytest's exit code. It
does not change system directory permissions or access the configured live
vector database. Never pass a project or data directory to `--basetemp`, because
pytest clears that directory before running tests.

Chunking regression tests preserve spaces, punctuation, and line breaks: with
zero overlap, concatenating chunk contents must reproduce the chunker's input
exactly. Boundary whitespace is rebalanced into non-empty source slices; overlap
may be reduced to respect the size limit. If a whitespace span cannot satisfy
all three constraints (complete text, bounded size, non-empty chunks), ingestion
raises `ChunkingError` before embedding or replacing stored data. Normalize such
input explicitly in a cleaner or increase the chunk size; the chunker does not
silently discard characters.

### 8.3 Running the CLI

```bash
los --help
los ingest ./notes/ --subject AI
los search "attention"
los ask "What is attention?" --subject AI
```

### 8.4 Qdrant Setup

For production: run Qdrant in Docker or as a service.
For testing: use `location: ":memory:"` in config (no server needed).

```bash
# Docker Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 8.5 Naming Conventions

- Packages, modules, functions, methods, and variables use `snake_case`.
  Classes use `PascalCase`; constants and registry mappings use `UPPER_SNAKE_CASE`.
- A concrete component module includes its role: `markdown_parser.py`,
  `markdown_cleaner.py`, `recursive_chunker.py`, `bge_m3_embedder.py`,
  `qdrant_store.py`, `vector_retriever.py`, and `openai_compatible_llm.py`.
  Keep established provider/model names and acronyms in class names, such as
  `OpenAICompatibleLLM` and `BGEM3Embedder`.
- Shared modules retain conventional names: `models.py`, `interfaces.py`,
  `exceptions.py`, `container.py`, and each plugin package's `registry.py`.
  Entry points remain `cli/main.py` and `web/app.py`.
- Tests mirror the implementation layers and use `test_<component>.py`.
  A suite covering several related components may use a plural name such as
  `test_llms.py`; scenario tests live under `tests/e2e/`.
- Module renames must update imports, documentation, and string-based patch
  targets together. Configuration keys and plugin type strings are separate
  public contracts and are not renamed to match Python filenames.

---

## 9. Benchmark Results

From `tests/benchmarks/run.py`:

| Metric | Value |
|--------|-------|
| Parse (100-paragraph doc) | 3,615 KB/s |
| Chunk (size=1000) | 200 chunks in ~1ms |
| Qdrant store (500 chunks) | 0.44 ms/chunk |
| Qdrant search | 2.71 ms/query |
| End-to-end retrieve | 2.97 ms/query |
| Memory (10K chunks) | ~0.5 MB Python overhead |

---

## 10. Future Directions (V2+)

| Feature | Status |
|---------|--------|
| PDF / DOCX / PPTX parsers | V2 |
| Bilibili / YouTube subtitles | V2 |
| GitHub README / Wiki | V2 |
| Hybrid retriever (BM25 + Vector) | V2 |
| Reranker (Jina / Cohere) | V2 |
| Agent / Memory / Workflow | V2 |
| Knowledge Graph | V2 |
| Web UI | V2 |
| FastAPI server | V2 |
