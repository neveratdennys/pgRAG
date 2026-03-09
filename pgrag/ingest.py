from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pgrag.config import Settings
from pgrag.db import PgVectorStore
from pgrag.embeddings import build_embedder


@dataclass
class IngestSummary:
    files_seen: int
    files_indexed: int
    chunks_upserted: int
    skipped_files: list[str]
    used_local_embeddings: bool
    embedding_dimensions: int


@dataclass
class ReindexSummary:
    chunks_processed: int
    source_filter: str | None
    used_local_embeddings: bool
    embedding_dimensions: int


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "head", "svg", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "section", "article", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "head", "svg", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self.parts.append(text + " ")


def _looks_like_html(raw: str) -> bool:
    probe = raw[:4000].lower()
    return any(token in probe for token in ("<html", "<body", "<article", "<div", "<p>", "<h1"))


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_text(raw: str) -> str:
    if not _looks_like_html(raw):
        return _normalize_text(raw)

    article_blocks = re.findall(
        r"<article\b.*?</article>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    html_blob = "\n".join(article_blocks) if article_blocks else raw
    parser = _HTMLTextExtractor()
    parser.feed(html_blob)
    parser.close()
    return _normalize_text("".join(parser.parts))


def _split_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size < 200:
        raise ValueError("chunk_size must be >= 200.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    limit = len(text)
    while start < limit:
        end = min(start + chunk_size, limit)
        if end < limit:
            candidate_break = text.rfind("\n", start + int(chunk_size * 0.5), end)
            if candidate_break <= start:
                candidate_break = text.rfind(" ", start + int(chunk_size * 0.5), end)
            if candidate_break > start:
                end = candidate_break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= limit:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path '{path}' does not exist.")

    accepted = {
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".log",
    }
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in accepted]
    files.sort()
    return files


def ingest_path(
    settings: Settings,
    path: str,
    *,
    source_override: str | None = None,
    title_override: str | None = None,
    dry_run: bool = False,
    chunk_size: int = 1600,
    chunk_overlap: int = 250,
    force_local_embeddings: bool = False,
) -> IngestSummary:
    root = Path(path).expanduser().resolve()
    files = _iter_files(root)

    store = PgVectorStore(settings)
    db_dimensions = store.get_embedding_dimensions()
    if db_dimensions is None:
        raise RuntimeError(
            "Database schema is not initialized. Run `python -m pgrag init-db` first."
        )

    embedder = build_embedder(
        settings,
        force_local=force_local_embeddings,
        dimensions_override=db_dimensions,
    )

    indexed_files = 0
    upserted_chunks = 0
    skipped: list[str] = []

    for file_path in files:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        text = _extract_text(raw)
        if not text:
            skipped.append(str(file_path))
            continue

        chunks = _split_chunks(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            skipped.append(str(file_path))
            continue

        if source_override:
            if len(files) == 1:
                source = source_override
            else:
                relative = file_path.relative_to(root).as_posix()
                source = f"{source_override.rstrip('/')}/{relative}"
        else:
            if root.is_file():
                source = file_path.name
            else:
                source = file_path.relative_to(root).as_posix()

        title = title_override or file_path.stem
        metadata = {
            "path": str(file_path),
            "relative_path": file_path.relative_to(root).as_posix() if root.is_dir() else file_path.name,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "parser": "html" if _looks_like_html(raw) else "plain",
        }

        if dry_run:
            indexed_files += 1
            upserted_chunks += len(chunks)
            continue

        embeddings = [embedder.embed(chunk) for chunk in chunks]
        upserted = store.upsert_chunks(
            source=source,
            title=title,
            chunks=chunks,
            embeddings=embeddings,
            base_metadata=metadata,
        )
        indexed_files += 1
        upserted_chunks += upserted

    return IngestSummary(
        files_seen=len(files),
        files_indexed=indexed_files,
        chunks_upserted=upserted_chunks,
        skipped_files=skipped,
        used_local_embeddings=embedder.__class__.__name__ == "LocalHashEmbedder",
        embedding_dimensions=embedder.dimensions,
    )


def reindex_embeddings(
    settings: Settings,
    *,
    source_filter: str | None = None,
    force_local_embeddings: bool = False,
) -> ReindexSummary:
    store = PgVectorStore(settings)
    db_dimensions = store.get_embedding_dimensions()
    if db_dimensions is None:
        raise RuntimeError(
            "Database schema is not initialized. Run `python -m pgrag init-db` first."
        )

    embedder = build_embedder(
        settings,
        force_local=force_local_embeddings,
        dimensions_override=db_dimensions,
    )

    chunks = store.fetch_chunks_for_reindex(source=source_filter)
    for chunk in chunks:
        vector = embedder.embed(chunk.content)
        store.update_chunk_embedding(chunk.id, vector)

    return ReindexSummary(
        chunks_processed=len(chunks),
        source_filter=source_filter,
        used_local_embeddings=embedder.__class__.__name__ == "LocalHashEmbedder",
        embedding_dimensions=embedder.dimensions,
    )


def dump_summary_json(summary: Any) -> str:
    return json.dumps(vars(summary), indent=2)
