from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from pgrag.config import RetrievalProfileConfig, Settings
from pgrag.db import PgVectorStore
from pgrag.embeddings import Embedder
from pgrag.foundry import FoundryChatClient


@dataclass
class RetrievedChunk:
    id: int
    source: str
    title: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]
    vector_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    vector_rank: int | None = None
    lexical_rank: int | None = None


@dataclass
class RetrievalOverrides:
    profile_name: str | None = None
    hybrid_mode: str | None = None
    vector_k: int | None = None
    lexical_k: int | None = None
    final_k: int | None = None
    rerank_mode: str | None = None
    rerank_top_n: int | None = None
    source_filters: list[str] | None = None
    debug: bool = False
    rerank_model_alias: str | None = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    trace: dict[str, Any]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    if max_v <= min_v:
        return {k: 1.0 for k in values}
    return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}


def _ranked_dict(items: list[dict[str, Any]], key: str) -> dict[int, int]:
    ranked: dict[int, int] = {}
    for rank, item in enumerate(items, start=1):
        ranked[int(item[key])] = rank
    return ranked


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: PgVectorStore,
        embedder: Embedder,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder

    def _resolve_profile(self, overrides: RetrievalOverrides) -> tuple[str, RetrievalProfileConfig]:
        profile_name = overrides.profile_name or self.settings.retrieval.default_profile
        profile = self.settings.retrieval.profiles.get(profile_name)
        if not profile:
            raise ValueError(
                f"Unknown retrieval profile '{profile_name}'. "
                f"Available profiles: {', '.join(sorted(self.settings.retrieval.profiles.keys()))}"
            )

        # Lightweight copy by rebuilding dataclass members.
        resolved = RetrievalProfileConfig(
            vector=type(profile.vector)(**vars(profile.vector)),
            lexical=type(profile.lexical)(**vars(profile.lexical)),
            hybrid=type(profile.hybrid)(**vars(profile.hybrid)),
            reranker=type(profile.reranker)(**vars(profile.reranker)),
            context=type(profile.context)(**vars(profile.context)),
            final_k=profile.final_k,
        )

        if overrides.hybrid_mode:
            resolved.hybrid.mode = overrides.hybrid_mode.lower()
        if overrides.vector_k:
            resolved.vector.k = int(overrides.vector_k)
        if overrides.lexical_k:
            resolved.lexical.k = int(overrides.lexical_k)
        if overrides.final_k:
            resolved.final_k = int(overrides.final_k)
            resolved.reranker.keep_n = min(resolved.reranker.keep_n, resolved.final_k)
            resolved.context.max_chunks = min(resolved.context.max_chunks, resolved.final_k)
        if overrides.rerank_mode:
            resolved.reranker.mode = overrides.rerank_mode.lower()
        if overrides.rerank_top_n:
            resolved.reranker.top_n = int(overrides.rerank_top_n)
        if overrides.rerank_model_alias:
            resolved.reranker.model_alias = overrides.rerank_model_alias

        return profile_name, resolved

    def _fuse(
        self,
        vector_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        profile: RetrievalProfileConfig,
    ) -> list[RetrievedChunk]:
        by_id: dict[int, RetrievedChunk] = {}

        for row in vector_rows:
            cid = int(row["id"])
            by_id[cid] = RetrievedChunk(
                id=cid,
                source=row["source"],
                title=row["title"],
                chunk_index=int(row["chunk_index"]),
                text=row["content"],
                metadata=row["metadata"],
                vector_score=float(row["vector_score"]),
                vector_rank=int(row["vector_rank"]),
            )

        for row in lexical_rows:
            cid = int(row["id"])
            if cid not in by_id:
                by_id[cid] = RetrievedChunk(
                    id=cid,
                    source=row["source"],
                    title=row["title"],
                    chunk_index=int(row["chunk_index"]),
                    text=row["content"],
                    metadata=row["metadata"],
                )
            by_id[cid].lexical_score = float(row["lexical_score"])
            by_id[cid].lexical_rank = int(row["lexical_rank"])

        mode = profile.hybrid.mode
        if mode == "vector_only":
            for chunk in by_id.values():
                chunk.fusion_score = chunk.vector_score or 0.0
            return sorted(by_id.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)

        if mode == "lexical_only":
            for chunk in by_id.values():
                chunk.fusion_score = chunk.lexical_score or 0.0
            return sorted(by_id.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)

        vector_ranks = _ranked_dict(vector_rows, "id")
        lexical_ranks = _ranked_dict(lexical_rows, "id")

        if mode == "rrf":
            for cid, chunk in by_id.items():
                score = 0.0
                if cid in vector_ranks:
                    score += profile.hybrid.vector_weight * (
                        1.0 / (profile.hybrid.rrf_k + vector_ranks[cid])
                    )
                if cid in lexical_ranks:
                    score += profile.hybrid.lexical_weight * (
                        1.0 / (profile.hybrid.rrf_k + lexical_ranks[cid])
                    )
                chunk.fusion_score = score
            return sorted(by_id.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)

        vector_scores = {
            int(row["id"]): float(row.get("vector_score", 0.0))
            for row in vector_rows
        }
        lexical_scores = {
            int(row["id"]): float(row.get("lexical_score", 0.0))
            for row in lexical_rows
        }
        norm_vector = _normalize_scores(vector_scores)
        norm_lexical = _normalize_scores(lexical_scores)

        for cid, chunk in by_id.items():
            chunk.fusion_score = (
                profile.hybrid.vector_weight * norm_vector.get(cid, 0.0)
                + profile.hybrid.lexical_weight * norm_lexical.get(cid, 0.0)
            )

        return sorted(by_id.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)

    def _rerank_llm(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        profile: RetrievalProfileConfig,
        model_alias: str,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        model_cfg = self.settings.models.get(model_alias)
        if not model_cfg:
            return chunks, {
                "status": "skipped",
                "reason": f"reranker model alias '{model_alias}' is not configured",
            }

        client = FoundryChatClient(model_cfg)
        limited = chunks[: profile.reranker.top_n]
        if not limited:
            return chunks, {"status": "skipped", "reason": "no candidates"}

        item_lines: list[str] = []
        for idx, chunk in enumerate(limited, start=1):
            preview = chunk.text[: profile.context.max_chars_per_chunk].strip()
            item_lines.append(
                f"ID: {idx}\n"
                f"source: {chunk.source}\n"
                f"title: {chunk.title}\n"
                f"chunk_index: {chunk.chunk_index}\n"
                f"text: {preview}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a semantic reranker for RAG retrieval. "
                    "Score each candidate for relevance to the user query on a 0 to 1 scale. "
                    "Return only strict JSON: "
                    '{"scores":[{"id":1,"score":0.95},{"id":2,"score":0.20}]}. '
                    "Do not include any text outside JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query:\n{query}\n\n"
                    "Candidates:\n\n"
                    + "\n\n---\n\n".join(item_lines)
                ),
            },
        ]

        started = time.perf_counter()
        raw = client.complete(
            messages=messages,
            temperature=profile.reranker.temperature,
            max_tokens=profile.reranker.max_tokens,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        parsed = _extract_json_object(raw)
        if not parsed:
            return chunks, {
                "status": "fallback",
                "reason": "failed to parse reranker JSON",
                "elapsed_ms": elapsed_ms,
            }

        raw_scores = parsed.get("scores")
        if not isinstance(raw_scores, list):
            return chunks, {
                "status": "fallback",
                "reason": "reranker payload missing 'scores' list",
                "elapsed_ms": elapsed_ms,
            }

        score_by_idx: dict[int, float] = {}
        for row in raw_scores:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id"))
                score = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > len(limited):
                continue
            score_by_idx[idx] = score

        if not score_by_idx:
            return chunks, {
                "status": "fallback",
                "reason": "reranker returned no valid scores",
                "elapsed_ms": elapsed_ms,
            }

        reranked: list[RetrievedChunk] = []
        untouched = chunks[profile.reranker.top_n :]
        for idx, chunk in enumerate(limited, start=1):
            score = score_by_idx.get(idx)
            if score is None:
                continue
            chunk.rerank_score = score
            if score >= profile.reranker.min_score:
                reranked.append(chunk)

        reranked.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        for chunk in limited:
            if chunk not in reranked:
                reranked.append(chunk)

        return reranked + untouched, {
            "status": "ok",
            "elapsed_ms": elapsed_ms,
            "scored": len(score_by_idx),
            "top_n": len(limited),
        }

    def retrieve(self, query: str, overrides: RetrievalOverrides) -> RetrievalResult:
        profile_name, profile = self._resolve_profile(overrides)
        source_filters = overrides.source_filters

        timings_ms: dict[str, float] = {}
        trace: dict[str, Any] = {
            "profile": profile_name,
            "hybrid_mode": profile.hybrid.mode,
            "vector_metric": profile.vector.metric,
            "reranker_mode": profile.reranker.mode,
            "source_filters": source_filters or [],
        }

        started = time.perf_counter()
        query_vector = self.embedder.embed(query)
        timings_ms["embed_query"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        vector_rows = self.store.search_vector(
            query_vector=query_vector,
            k=profile.vector.k,
            metric=profile.vector.metric,
            source_filters=source_filters,
            ef_search=profile.vector.ef_search,
            min_score=profile.vector.min_score,
        )
        timings_ms["vector_search"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        lexical_rows = self.store.search_lexical(
            query=query,
            k=profile.lexical.k,
            language=profile.lexical.language,
            source_filters=source_filters,
            min_score=profile.lexical.min_score,
        )
        timings_ms["lexical_search"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        fused = self._fuse(vector_rows=vector_rows, lexical_rows=lexical_rows, profile=profile)
        timings_ms["hybrid_fusion"] = round((time.perf_counter() - started) * 1000, 2)

        rerank_trace: dict[str, Any] = {"status": "disabled"}
        rerank_mode = profile.reranker.mode
        if rerank_mode == "llm":
            selected_alias = (
                profile.reranker.model_alias
                or overrides.rerank_model_alias
                or self.settings.app.default_model
            )
            fused, rerank_trace = self._rerank_llm(
                query=query,
                chunks=fused,
                profile=profile,
                model_alias=selected_alias,
            )

        final_chunks = fused[: profile.final_k]
        trace["candidate_counts"] = {
            "vector": len(vector_rows),
            "lexical": len(lexical_rows),
            "fused": len(fused),
            "final": len(final_chunks),
        }
        trace["timings_ms"] = timings_ms
        trace["rerank"] = rerank_trace

        if overrides.debug:
            trace["top_candidates"] = [
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "vector_score": chunk.vector_score,
                    "lexical_score": chunk.lexical_score,
                    "fusion_score": chunk.fusion_score,
                    "rerank_score": chunk.rerank_score,
                }
                for chunk in final_chunks
            ]

        return RetrievalResult(chunks=final_chunks, trace=trace)
