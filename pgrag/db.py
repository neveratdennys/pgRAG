from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pgrag.config import Settings

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None


@dataclass
class StoredChunk:
    id: int
    source: str
    title: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]


def _require_psycopg() -> Any:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is not installed. Run: pip install -r requirements.txt"
        )
    return psycopg


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.10f}" for v in values) + "]"


def _parse_vector_dimension(format_type: str | None) -> int | None:
    if not format_type:
        return None
    match = re.search(r"(?:vector|halfvec)\((\d+)\)", format_type)
    if not match:
        return None
    return int(match.group(1))


def _parse_embedding_type(format_type: str | None) -> str | None:
    if not format_type:
        return None
    match = re.search(r"(vector|halfvec)\(\d+\)", format_type)
    if not match:
        return None
    return str(match.group(1))


class PgVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.psycopg = _require_psycopg()

    def _connect(self):
        return self.psycopg.connect(**self.settings.database.connection_kwargs())

    def get_embedding_dimensions(self) -> int | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = 'chunks'
                      AND a.attname = 'embedding'
                      AND a.atttypid > 0
                      AND a.attnum > 0
                      AND NOT a.attisdropped;
                    """
                )
                row = cur.fetchone()
                if not row:
                    return None
                return _parse_vector_dimension(row[0])

    def get_embedding_storage_type(self) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = 'chunks'
                      AND a.attname = 'embedding'
                      AND a.atttypid > 0
                      AND a.attnum > 0
                      AND NOT a.attisdropped;
                    """
                )
                row = cur.fetchone()
                if row:
                    parsed = _parse_embedding_type(row[0])
                    if parsed in {"vector", "halfvec"}:
                        return parsed
        configured = self.settings.database.embedding_type.lower()
        if configured in {"vector", "halfvec"}:
            return configured
        return "vector"

    def init_database(
        self,
        skip_extension: bool = False,
        embedding_dimensions: int | None = None,
    ) -> dict[str, Any]:
        configured_dimensions = embedding_dimensions
        if configured_dimensions is None:
            configured_dimensions = (
                self.settings.embeddings.dimensions
                if (self.settings.embeddings and self.settings.embeddings.dimensions)
                else 1536
            )
        if configured_dimensions < 1 or configured_dimensions > 20000:
            raise ValueError("Embedding dimensions must be between 1 and 20000.")

        embedding_storage_type = self.settings.database.embedding_type.lower()
        if embedding_storage_type not in {"vector", "halfvec"}:
            raise ValueError("database.embedding_type must be 'vector' or 'halfvec'.")

        table_sql = f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT 'untitled',
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding {embedding_storage_type}({configured_dimensions}) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            fts tsvector GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(title, '') || ' ' || content)
            ) STORED,
            UNIQUE (source, chunk_index)
        );
        """

        vector_index_name = "none"
        vector_index_mode = "exact_scan"
        vector_index_error: str | None = None

        with self._connect() as conn:
            with conn.cursor() as cur:
                if not skip_extension:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute(table_sql)

                opclass = f"{embedding_storage_type}_cosine_ops"
                cur.execute("SAVEPOINT pgrag_vector_index_sp;")
                try:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_cosine_idx
                        ON chunks USING hnsw (embedding {opclass});
                        """
                    )
                    vector_index_name = "chunks_embedding_hnsw_cosine_idx"
                    vector_index_mode = "hnsw"
                    cur.execute("RELEASE SAVEPOINT pgrag_vector_index_sp;")
                except Exception as exc:
                    cur.execute("ROLLBACK TO SAVEPOINT pgrag_vector_index_sp;")
                    cur.execute("RELEASE SAVEPOINT pgrag_vector_index_sp;")
                    vector_index_error = str(exc)
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_fts_idx
                    ON chunks USING gin (fts);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_source_idx
                    ON chunks (source);
                    """
                )

            conn.commit()

        existing_dimensions = self.get_embedding_dimensions()
        if existing_dimensions is None:
            raise RuntimeError("Could not determine 'chunks.embedding' dimensions.")
        existing_type = self.get_embedding_storage_type()
        if existing_type != embedding_storage_type:
            raise ValueError(
                "Configured embedding type does not match database schema. "
                f"config={embedding_storage_type}, db={existing_type}."
            )
        if existing_dimensions != configured_dimensions:
            raise ValueError(
                "Configured embedding dimensions do not match database schema. "
                f"config={configured_dimensions}, db={existing_dimensions}. "
                "Use matching dimensions or recreate the table."
            )

        return {
            "table": "chunks",
            "embedding_type": existing_type,
            "embedding_dimensions": existing_dimensions,
            "vector_index": vector_index_name,
            "vector_index_mode": vector_index_mode,
            "vector_index_error": vector_index_error,
            "lexical_index": "chunks_fts_idx",
        }

    def upsert_chunks(
        self,
        source: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        base_metadata: dict[str, Any] | None = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        if not chunks:
            return 0

        metadata_seed = dict(base_metadata or {})
        cast_type = self.get_embedding_storage_type()

        with self._connect() as conn:
            with conn.cursor() as cur:
                for i, chunk_text in enumerate(chunks):
                    metadata = dict(metadata_seed)
                    metadata["chunk_length"] = len(chunk_text)
                    metadata["chunk_index"] = i
                    cur.execute(
                        f"""
                        INSERT INTO chunks (source, title, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s::{cast_type}, %s::jsonb)
                        ON CONFLICT (source, chunk_index)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW();
                        """,
                        (
                            source,
                            title or "untitled",
                            i,
                            chunk_text,
                            _vector_literal(embeddings[i]),
                            json.dumps(metadata),
                        ),
                    )
            conn.commit()

        return len(chunks)

    def fetch_chunks_for_reindex(self, source: str | None = None) -> list[StoredChunk]:
        sql = """
        SELECT id, source, title, chunk_index, content, metadata
        FROM chunks
        """
        params: list[Any] = []
        if source:
            sql += " WHERE source = %s"
            params.append(source)
        sql += " ORDER BY source, chunk_index"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        chunks: list[StoredChunk] = []
        for row in rows:
            chunks.append(
                StoredChunk(
                    id=int(row[0]),
                    source=str(row[1]),
                    title=str(row[2]),
                    chunk_index=int(row[3]),
                    content=str(row[4]),
                    metadata=dict(row[5] or {}),
                )
            )
        return chunks

    def update_chunk_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        cast_type = self.get_embedding_storage_type()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE chunks
                    SET embedding = %s::{cast_type},
                        updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (_vector_literal(embedding), chunk_id),
                )
            conn.commit()

    def search_vector(
        self,
        query_vector: list[float],
        k: int,
        metric: str,
        source_filters: list[str] | None = None,
        ef_search: int | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not query_vector:
            return []

        metric_key = metric.lower()
        if metric_key not in {"cosine", "l2", "ip"}:
            raise ValueError(f"Unsupported vector metric '{metric}'.")

        operator = {
            "cosine": "<=>",
            "l2": "<->",
            "ip": "<#>",
        }[metric_key]
        cast_type = self.get_embedding_storage_type()

        where_parts: list[str] = []
        params: list[Any] = []
        if source_filters:
            where_parts.append("source = ANY(%s)")
            params.append(source_filters)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        sql = f"""
        SELECT id, source, title, chunk_index, content, metadata, (embedding {operator} %s::{cast_type}) AS distance
        FROM chunks
        {where_sql}
        ORDER BY embedding {operator} %s::{cast_type}
        LIMIT %s;
        """
        vector_param = _vector_literal(query_vector)
        full_params = [vector_param] + params + [vector_param, int(k)]

        with self._connect() as conn:
            with conn.cursor() as cur:
                if ef_search and ef_search > 0:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)};")
                cur.execute(sql, full_params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row[6])
            if metric_key == "cosine":
                score = 1.0 - distance
            elif metric_key == "l2":
                score = 1.0 / (1.0 + distance)
            else:
                # pgvector <#> returns negative inner product distance.
                score = -distance

            if score < min_score:
                continue

            results.append(
                {
                    "id": int(row[0]),
                    "source": str(row[1]),
                    "title": str(row[2]),
                    "chunk_index": int(row[3]),
                    "content": str(row[4]),
                    "metadata": dict(row[5] or {}),
                    "vector_score": score,
                    "vector_distance": distance,
                    "vector_rank": rank,
                }
            )

        return results

    def search_lexical(
        self,
        query: str,
        k: int,
        language: str = "english",
        source_filters: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        language = language.lower()
        where_parts: list[str] = []
        params: list[Any] = []

        if language == "english":
            where_parts.append("fts @@ websearch_to_tsquery('english', %s)")
            params.append(query)
        else:
            where_parts.append(
                "to_tsvector(%s, coalesce(title, '') || ' ' || content) @@ websearch_to_tsquery(%s, %s)"
            )
            params.extend([language, language, query])

        if source_filters:
            where_parts.append("source = ANY(%s)")
            params.append(source_filters)

        where_sql = "WHERE " + " AND ".join(where_parts)

        if language == "english":
            rank_expr = "ts_rank_cd(fts, websearch_to_tsquery('english', %s))"
            rank_params = [query]
        else:
            rank_expr = (
                "ts_rank_cd(to_tsvector(%s, coalesce(title, '') || ' ' || content), "
                "websearch_to_tsquery(%s, %s))"
            )
            rank_params = [language, language, query]

        sql = f"""
        SELECT id, source, title, chunk_index, content, metadata, {rank_expr} AS lexical_score
        FROM chunks
        {where_sql}
        ORDER BY lexical_score DESC
        LIMIT %s;
        """
        all_params = rank_params + params + [int(k)]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, all_params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            lexical_score = float(row[6])
            if lexical_score < min_score:
                continue
            results.append(
                {
                    "id": int(row[0]),
                    "source": str(row[1]),
                    "title": str(row[2]),
                    "chunk_index": int(row[3]),
                    "content": str(row[4]),
                    "metadata": dict(row[5] or {}),
                    "lexical_score": lexical_score,
                    "lexical_rank": rank,
                }
            )

        return results


def init_database(
    settings: Settings,
    skip_extension: bool = False,
    embedding_dimensions: int | None = None,
) -> dict[str, Any]:
    return PgVectorStore(settings).init_database(
        skip_extension=skip_extension,
        embedding_dimensions=embedding_dimensions,
    )
