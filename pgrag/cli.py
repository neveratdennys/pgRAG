from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pgrag.config import load_settings
from pgrag.db import PgVectorStore
from pgrag.db import init_database
from pgrag.embeddings import build_embedder, infer_embedding_dimensions
from pgrag.ingest import dump_summary_json, ingest_path, reindex_embeddings
from pgrag.rag import RagEngine
from pgrag.retrieval import HybridRetriever, RetrievalOverrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgrag-cli",
        description="pgvector-backed RAG CLI with hybrid retrieval and semantic reranking",
    )
    parser.add_argument(
        "--config",
        default="config.local.yaml",
        help="Path to YAML config file (default: config.local.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="List configured model aliases")
    models.add_argument("--details", action="store_true", help="Show model endpoint details")

    init_db = sub.add_parser("init-db", help="Initialize pgvector extension and base schema")
    init_db.add_argument(
        "--skip-extension",
        action="store_true",
        help="Skip CREATE EXTENSION vector",
    )

    ingest = sub.add_parser("ingest", help="Ingest files into pgvector")
    ingest.add_argument("path", help="File or directory to ingest")
    ingest.add_argument("--source", default=None, help="Override source id")
    ingest.add_argument("--title", default=None, help="Optional title override")
    ingest.add_argument("--chunk-size", type=int, default=1600, help="Chunk size in chars")
    ingest.add_argument("--chunk-overlap", type=int, default=250, help="Chunk overlap in chars")
    ingest.add_argument("--dry-run", action="store_true", help="Parse/chunk only, no DB writes")
    ingest.add_argument(
        "--use-local-embeddings",
        action="store_true",
        help="Use local hash embeddings instead of configured embedding API",
    )
    ingest.add_argument("--json", action="store_true", help="Print structured JSON summary")

    reindex = sub.add_parser("reindex", help="Recompute embeddings for stored chunks")
    reindex.add_argument("--source", default=None, help="Only reindex one source")
    reindex.add_argument(
        "--use-local-embeddings",
        action="store_true",
        help="Use local hash embeddings instead of configured embedding API",
    )
    reindex.add_argument("--json", action="store_true", help="Print structured JSON summary")

    retrieve = sub.add_parser("retrieve", help="Run retrieval pipeline without LLM generation")
    retrieve.add_argument("query", help="Query text")
    retrieve.add_argument("--profile", default=None, help="Retrieval profile name")
    retrieve.add_argument(
        "--hybrid-mode",
        choices=["rrf", "weighted", "vector_only", "lexical_only"],
        default=None,
        help="Override hybrid fusion mode",
    )
    retrieve.add_argument("--vector-k", type=int, default=None, help="Override vector candidate count")
    retrieve.add_argument("--lexical-k", type=int, default=None, help="Override lexical candidate count")
    retrieve.add_argument("--final-k", type=int, default=None, help="Override final retrieved chunk count")
    retrieve.add_argument(
        "--rerank-mode",
        choices=["none", "llm"],
        default=None,
        help="Override semantic reranker mode",
    )
    retrieve.add_argument("--rerank-top-n", type=int, default=None, help="Override reranker top_n")
    retrieve.add_argument("--rerank-model", default=None, help="Override reranker model alias")
    retrieve.add_argument(
        "--source-filter",
        action="append",
        default=None,
        help="Filter retrieval to one source (repeatable)",
    )
    retrieve.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    retrieve.add_argument("--debug-retrieval", action="store_true", help="Print retrieval trace")
    retrieve.add_argument(
        "--use-local-embeddings",
        action="store_true",
        help="Use local hash embeddings for retrieval query embedding",
    )

    ask = sub.add_parser("ask", help="Single-shot RAG question")
    ask.add_argument("question", help="Question to ask")
    ask.add_argument("--model", default=None, help="Generation model alias")
    ask.add_argument("--profile", default=None, help="Retrieval profile name")
    ask.add_argument(
        "--hybrid-mode",
        choices=["rrf", "weighted", "vector_only", "lexical_only"],
        default=None,
        help="Override hybrid fusion mode",
    )
    ask.add_argument("--vector-k", type=int, default=None, help="Override vector candidate count")
    ask.add_argument("--lexical-k", type=int, default=None, help="Override lexical candidate count")
    ask.add_argument("--final-k", type=int, default=None, help="Override final retrieved chunk count")
    ask.add_argument(
        "--rerank-mode",
        choices=["none", "llm"],
        default=None,
        help="Override semantic reranker mode",
    )
    ask.add_argument("--rerank-top-n", type=int, default=None, help="Override reranker top_n")
    ask.add_argument("--rerank-model", default=None, help="Override reranker model alias")
    ask.add_argument(
        "--source-filter",
        action="append",
        default=None,
        help="Filter retrieval to one source (repeatable)",
    )
    ask.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    ask.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Generation max output tokens",
    )
    ask.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    ask.add_argument("--debug-retrieval", action="store_true", help="Print retrieval trace")
    ask.add_argument(
        "--use-local-embeddings",
        action="store_true",
        help="Use local hash embeddings for retrieval query embedding",
    )

    chat = sub.add_parser("chat", help="Interactive RAG chat session")
    chat.add_argument("--model", default=None, help="Generation model alias")
    chat.add_argument("--profile", default=None, help="Retrieval profile name")
    chat.add_argument(
        "--hybrid-mode",
        choices=["rrf", "weighted", "vector_only", "lexical_only"],
        default=None,
        help="Override hybrid fusion mode",
    )
    chat.add_argument("--vector-k", type=int, default=None, help="Override vector candidate count")
    chat.add_argument("--lexical-k", type=int, default=None, help="Override lexical candidate count")
    chat.add_argument("--final-k", type=int, default=None, help="Override final retrieved chunk count")
    chat.add_argument(
        "--rerank-mode",
        choices=["none", "llm"],
        default=None,
        help="Override semantic reranker mode",
    )
    chat.add_argument("--rerank-top-n", type=int, default=None, help="Override reranker top_n")
    chat.add_argument("--rerank-model", default=None, help="Override reranker model alias")
    chat.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    chat.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Generation max output tokens",
    )
    chat.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    chat.add_argument("--debug-retrieval", action="store_true", help="Print retrieval trace")
    chat.add_argument(
        "--use-local-embeddings",
        action="store_true",
        help="Use local hash embeddings for retrieval query embedding",
    )

    return parser


def _build_retrieval_overrides(args: argparse.Namespace) -> RetrievalOverrides:
    return RetrievalOverrides(
        profile_name=args.profile,
        hybrid_mode=args.hybrid_mode,
        vector_k=args.vector_k,
        lexical_k=args.lexical_k,
        final_k=args.final_k,
        rerank_mode=args.rerank_mode,
        rerank_top_n=args.rerank_top_n,
        source_filters=args.source_filter,
        debug=args.debug_retrieval,
        rerank_model_alias=args.rerank_model,
    )


def _print_chunks(chunks) -> None:
    if not chunks:
        print("\nNo chunks were retrieved.")
        return
    print("\nRetrieved Context:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[C{i}] {chunk.title} | {chunk.source} | idx={chunk.chunk_index}")
        print(chunk.text)


def _cmd_models(config_path: str, details: bool) -> int:
    settings = load_settings(config_path, require_model_profiles=False)
    if not settings.models:
        print("No fully configured models found in config.")
        return 0

    print("Configured models:")
    for alias, cfg in settings.models.items():
        if details:
            print(f"- {alias}: {cfg.model} @ {cfg.endpoint}")
        else:
            print(f"- {alias}")
    return 0


def _cmd_init_db(config_path: str, skip_extension: bool) -> int:
    settings = load_settings(config_path, require_model_profiles=False)
    inferred_dimensions: int | None = None
    if settings.embeddings and settings.embeddings.dimensions is None:
        inferred_dimensions = infer_embedding_dimensions(settings)

    result = init_database(
        settings,
        skip_extension=skip_extension,
        embedding_dimensions=inferred_dimensions,
    )

    provider = settings.embeddings.provider if settings.embeddings else "local_hash"
    print(
        "Initialized database schema: "
        f"embedding_provider={provider} "
        f"embedding_type={result.get('embedding_type', 'unknown')} "
        f"table={result['table']} "
        f"embedding_dimensions={result['embedding_dimensions']} "
        f"vector_index={result['vector_index']} "
        f"vector_index_mode={result.get('vector_index_mode', 'unknown')} "
        f"lexical_index={result['lexical_index']}"
    )
    if result.get("vector_index_error"):
        print(f"Vector index warning: {result['vector_index_error']}")
    return 0


def _cmd_ingest(config_path: str, args: argparse.Namespace) -> int:
    settings = load_settings(config_path, require_model_profiles=False)
    summary = ingest_path(
        settings=settings,
        path=args.path,
        source_override=args.source,
        title_override=args.title,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_local_embeddings=args.use_local_embeddings,
    )
    if args.json:
        print(dump_summary_json(summary))
    else:
        print(
            "Ingestion completed: "
            f"files_seen={summary.files_seen} "
            f"files_indexed={summary.files_indexed} "
            f"chunks_upserted={summary.chunks_upserted} "
            f"local_embeddings={summary.used_local_embeddings} "
            f"embedding_dimensions={summary.embedding_dimensions}"
        )
        if summary.skipped_files:
            print("Skipped files:")
            for path in summary.skipped_files:
                print(f"- {path}")
    return 0


def _cmd_reindex(config_path: str, args: argparse.Namespace) -> int:
    settings = load_settings(config_path, require_model_profiles=False)
    summary = reindex_embeddings(
        settings=settings,
        source_filter=args.source,
        force_local_embeddings=args.use_local_embeddings,
    )
    if args.json:
        print(dump_summary_json(summary))
    else:
        print(
            "Reindex completed: "
            f"chunks_processed={summary.chunks_processed} "
            f"source_filter={summary.source_filter or '*'} "
            f"local_embeddings={summary.used_local_embeddings} "
            f"embedding_dimensions={summary.embedding_dimensions}"
        )
    return 0


def _cmd_ask(config_path: str, args: argparse.Namespace) -> int:
    settings = load_settings(config_path, require_model_profiles=True)
    engine = RagEngine(settings, force_local_embeddings=args.use_local_embeddings)
    result = engine.ask(
        question=args.question,
        model_alias=args.model,
        retrieval_overrides=_build_retrieval_overrides(args),
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    print(result.text)
    if args.show_context:
        _print_chunks(result.chunks)
    if args.debug_retrieval:
        print("\nRetrieval Trace:")
        print(json.dumps(result.retrieval_trace, indent=2))
    return 0


def _cmd_retrieve(config_path: str, args: argparse.Namespace) -> int:
    settings = load_settings(config_path, require_model_profiles=False)
    store = PgVectorStore(settings)
    embedder = build_embedder(
        settings=settings,
        force_local=args.use_local_embeddings,
        dimensions_override=store.get_embedding_dimensions(),
    )
    retriever = HybridRetriever(settings=settings, store=store, embedder=embedder)

    result = retriever.retrieve(
        query=args.query,
        overrides=_build_retrieval_overrides(args),
    )

    print(f"Retrieved {len(result.chunks)} chunks.")
    if args.show_context:
        _print_chunks(result.chunks)
    if args.debug_retrieval:
        print("\nRetrieval Trace:")
        print(json.dumps(result.trace, indent=2))
    return 0


def _cmd_chat(config_path: str, args: argparse.Namespace) -> int:
    settings = load_settings(config_path, require_model_profiles=True)
    engine = RagEngine(settings, force_local_embeddings=args.use_local_embeddings)

    print("Interactive chat started. Type 'exit' or 'quit' to end.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\nYou> ").strip()
        except EOFError:
            print("\nSession ended.")
            return 0

        if question.lower() in {"exit", "quit"}:
            print("Session ended.")
            return 0
        if not question:
            continue

        result = engine.ask(
            question=question,
            model_alias=args.model,
            retrieval_overrides=_build_retrieval_overrides(args),
            history=history,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        history.append((question, result.text))
        print(f"\nAssistant> {result.text}")
        if args.show_context:
            _print_chunks(result.chunks)
        if args.debug_retrieval:
            print("\nRetrieval Trace:")
            print(json.dumps(result.retrieval_trace, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {
        "models": lambda ns: _cmd_models(ns.config, ns.details),
        "init-db": lambda ns: _cmd_init_db(ns.config, ns.skip_extension),
        "ingest": lambda ns: _cmd_ingest(ns.config, ns),
        "reindex": lambda ns: _cmd_reindex(ns.config, ns),
        "retrieve": lambda ns: _cmd_retrieve(ns.config, ns),
        "ask": lambda ns: _cmd_ask(ns.config, ns),
        "chat": lambda ns: _cmd_chat(ns.config, ns),
    }

    try:
        return int(handlers[args.command](args))
    except Exception as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
