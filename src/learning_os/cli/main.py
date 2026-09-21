"""
Learning OS CLI — entry point.

Registered as the ``los`` console script in pyproject.toml.

Commands::

    los ingest FILE [FILE ...]  --subject AI --topic Transformer
    los search "attention"      --top_k 10
    los ask "How does attention work?"
    los stats
    los rebuild
    los config-show
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import yaml

from learning_os.infrastructure.config import load_config
from learning_os.infrastructure.container import Container
from learning_os.infrastructure.logging_ import setup_logging

# ── registry imports (side-effect: populate global registry dicts) ──
import learning_os.infrastructure.parsers.registry  # noqa: F401
import learning_os.infrastructure.chunkers.registry  # noqa: F401
import learning_os.infrastructure.embedders.registry  # noqa: F401
import learning_os.infrastructure.llms.registry  # noqa: F401
import learning_os.infrastructure.retrievers.registry  # noqa: F401
import learning_os.infrastructure.vector_stores.registry  # noqa: F401

app = typer.Typer(
    name="los",
    help="Learning OS — personal knowledge operating system.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# callback
# ══════════════════════════════════════════════════════════════════════

@app.callback()
def _main(
    ctx: typer.Context,
    config_path: str = typer.Option(
        None,
        "--config", "-c",
        help="Path to config.yaml (default: ./config.yaml or $LEARNING_OS_CONFIG).",
    ),
) -> None:
    """Bootstrap logging and the DI container; attach to context."""
    cfg = load_config(config_path)
    setup_logging(cfg["learning_os"]["logging"])

    from learning_os.infrastructure.container import (
        CHUNKER_REGISTRY,
        EMBEDDER_REGISTRY,
        LLM_REGISTRY,
        PARSER_REGISTRY,
        VECTOR_STORE_REGISTRY,
        CLEANER_REGISTRY,
    )

    # Wire plugin registries into the container.
    from learning_os.infrastructure.chunkers.registry import (
        CHUNKER_REGISTRY as CR,
    )
    from learning_os.infrastructure.cleaners.registry import (
        CLEANER_REGISTRY as CLR,
    )
    from learning_os.infrastructure.embedders.registry import (
        EMBEDDER_REGISTRY as ER,
    )
    from learning_os.infrastructure.llms.registry import LLM_REGISTRY as LR
    from learning_os.infrastructure.parsers.registry import (
        PARSER_REGISTRY as PR,
    )
    from learning_os.infrastructure.vector_stores.registry import (
        VECTOR_STORE_REGISTRY as VSR,
    )

    CLEANER_REGISTRY.update(CLR)
    CHUNKER_REGISTRY.update(CR)
    EMBEDDER_REGISTRY.update(ER)
    LLM_REGISTRY.update(LR)
    PARSER_REGISTRY.update(PR)
    VECTOR_STORE_REGISTRY.update(VSR)

    container = Container(cfg)
    ctx.obj = container


# ══════════════════════════════════════════════════════════════════════
# ingest
# ══════════════════════════════════════════════════════════════════════


@app.command()
def ingest(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(..., help="Files or directories to ingest."),
    subject: str = typer.Option(
        "", "--subject", "-s", help="Subject tag for all ingested files.",
    ),
    topic: str = typer.Option(
        "", "--topic", "-t", help="Topic tag for all ingested files.",
    ),
    language: str = typer.Option(
        "auto", "--lang", "-l", help="Language code (zh, en, auto).",
    ),
) -> None:
    """Ingest documents into the knowledge base."""
    container: Container = ctx.obj
    cfg = container.config

    # 1. 直接向容器索要装配好的完整流水线
    try:
        pipeline = container.ingest_pipeline()
    except Exception as e:
        typer.echo(f"❌ Pipeline initialization failed: {e}", err=True)
        raise typer.Exit(1)

    # 2. 覆盖 CLI 传入的标签（Tags）
    if subject:
        pipeline.chunker._subject = subject
    if topic:
        pipeline.chunker._topic = topic
    if language and language != "auto":
        pipeline.chunker._language = language

    # 准备 Parser 路由表
    import learning_os.infrastructure.parsers.registry as pr

    total_files = 0
    # 3. 遍历文件
    for glob_pattern in paths:
        p = Path(glob_pattern)
        files = list(p.rglob("*")) if p.is_dir() else [p]

        for fp in files:
            if not fp.is_file():
                continue

            # 动态寻找能处理该后缀的 Parser
            matched_parser = None
            for name, cls in pr.PARSER_REGISTRY.items():
                parser_inst = cls()
                if parser_inst.supports(fp):
                    matched_parser = parser_inst
                    break

            if matched_parser is None:
                typer.echo(f"  ⏭  No parser for: {fp} (skipped)")
                continue

            # 把匹配到的 Parser 临时换进流水线
            pipeline.parser = matched_parser

            # 4. 核心：一句代码跑完（解析 -> 清洗 -> 分块 -> 向量化 -> 存储）
            try:
                chunks_count = pipeline.run(fp)
                if chunks_count > 0:
                    typer.echo(f"  ✓  {fp} → {chunks_count} chunks")
                    total_files += 1
            except Exception as e:
                typer.echo(f"  ✗  {fp} failed: {e}", err=True)
                if cfg.get("ingest", {}).get("stop_on_error"):
                    raise

    typer.secho(f"\n📚 Successfully ingested {total_files} document(s).", fg=typer.colors.GREEN)


# ══════════════════════════════════════════════════════════════════════
# search
# ══════════════════════════════════════════════════════════════════════


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results."),
    subject: str = typer.Option(None, "--subject", help="Filter by subject."),
    topic: str = typer.Option(None, "--topic", help="Filter by topic."),
    source_type: str = typer.Option(None, "--type", help="Filter by source type."),
) -> None:
    """Search the knowledge base."""
    container: Container = ctx.obj

    try:
        embedder = container.embedder()
        store = container.vector_store()
        if embedder is None or store is None:
            typer.echo("❌ Embedder or vector store not configured.", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Init failed: {e}", err=True)
        raise typer.Exit(1)

    from learning_os.infrastructure.retrievers.vector_retriever import (
        VectorRetriever,
    )
    from learning_os.domain.interfaces import MetadataFilter

    retriever = VectorRetriever(embedder=embedder, vector_store=store)
    filt = MetadataFilter(
        subject=subject,
        topic=topic,
        source_type=source_type,
    ) if any([subject, topic, source_type]) else None

    try:
        results = retriever.retrieve(query, top_k=top_k, filter_=filt)
    except Exception as e:
        typer.echo(f"❌ Search failed: {e}", err=True)
        raise typer.Exit(1)

    if not results:
        typer.echo("No results found.")
        return

    for r in results:
        c = r.chunk
        typer.echo("─" * 60)
        typer.echo(
            f"#{r.rank + 1}  score={r.score:.4f}  "
            f"[{c.metadata.subject}/{c.metadata.topic}]  {c.metadata.source}"
        )
        typer.echo(c.content[:300] + ("..." if len(c.content) > 300 else ""))


# ══════════════════════════════════════════════════════════════════════
# ask
# ══════════════════════════════════════════════════════════════════════


@app.command()
def ask(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Question to ask."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Context chunks to retrieve."),
    subject: str = typer.Option(None, "--subject", help="Filter context by subject."),
) -> None:
    """Ask a question using retrieved context + LLM."""
    container: Container = ctx.obj

    try:
        embedder = container.embedder()
        store = container.vector_store()
        llm = container.llm()
        if embedder is None or store is None or llm is None:
            typer.echo("❌ Embedder, store, or LLM not configured.", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Init failed: {e}", err=True)
        raise typer.Exit(1)

    from learning_os.infrastructure.retrievers.vector_retriever import (
        VectorRetriever,
    )
    from learning_os.domain.interfaces import Message, MetadataFilter

    retriever = VectorRetriever(embedder=embedder, vector_store=store)
    filt = MetadataFilter(subject=subject) if subject else None

    # 1. Retrieve context
    try:
        results = retriever.retrieve(query, top_k=top_k, filter_=filt)
    except Exception as e:
        typer.echo(f"❌ Retrieval failed: {e}", err=True)
        raise typer.Exit(1)

    if not results:
        typer.echo("No relevant documents found.")
        return

    context = "\n\n".join(
        f"[{i+1}] {r.chunk.content}" for i, r in enumerate(results)
    )

    # 2. Build messages
    messages = [
        Message(
            role="system",
            content=(
                "You are a knowledgeable assistant. Answer the user's "
                "question using ONLY the provided context. If the "
                "context does not contain the answer, say so."
            ),
        ),
        Message(
            role="user",
            content=f"Context:\n{context}\n\nQuestion: {query}",
        ),
    ]

    # 3. Generate
    typer.echo("🤔 Thinking...")
    try:
        answer = llm.chat(messages)
        typer.echo(f"\n{answer}")
    except Exception as e:
        typer.echo(f"❌ LLM call failed: {e}", err=True)
        raise typer.Exit(1)


# ══════════════════════════════════════════════════════════════════════
# stats
# ══════════════════════════════════════════════════════════════════════


@app.command()
def stats(ctx: typer.Context) -> None:
    """Show knowledge base statistics."""
    container: Container = ctx.obj

    try:
        store = container.vector_store()
        if store is None:
            typer.echo("❌ No vector store configured.", err=True)
            raise typer.Exit(1)
        count = store.count()
    except Exception as e:
        typer.echo(f"❌ Stats failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Total chunks: {count}")


# ══════════════════════════════════════════════════════════════════════
# rebuild
# ══════════════════════════════════════════════════════════════════════


@app.command()
def rebuild(ctx: typer.Context) -> None:
    """Rebuild the vector index from scratch."""
    typer.echo(" rebuild — drops and re-creates the Qdrant collection.")
    if not typer.confirm("This will permanently delete ALL data in the vector store. Continue?"):
        raise typer.Abort()

    container: Container = ctx.obj
    store = container.vector_store()

    if store is None:
        typer.echo("❌ No vector store configured.", err=True)
        raise typer.Exit(1)

    typer.echo("Resetting vector store...")
    store.reset()
    typer.secho("✓ Vector store has been completely reset. You can now run 'los ingest' again.", fg=typer.colors.GREEN)


# ══════════════════════════════════════════════════════════════════════
# config-show
# ══════════════════════════════════════════════════════════════════════


@app.command()
def config_show(ctx: typer.Context) -> None:
    """Print the current effective configuration."""
    container: Container = ctx.obj
    # Pretty-print the config (hide sensitive keys).
    cfg = dict(container.config)
    # Redact API keys
    cfg_str = yaml.dump(cfg, default_flow_style=False, allow_unicode=True)
    typer.echo(cfg_str)
