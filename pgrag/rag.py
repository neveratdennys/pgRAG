from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pgrag.config import Settings
from pgrag.db import PgVectorStore
from pgrag.embeddings import build_embedder
from pgrag.foundry import FoundryChatClient
from pgrag.retrieval import HybridRetriever, RetrievalOverrides, RetrievedChunk


@dataclass
class Answer:
    text: str
    chunks: list[RetrievedChunk]
    retrieval_trace: dict[str, Any]


class RagEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        force_local_embeddings: bool = False,
    ) -> None:
        self.settings = settings
        self.store = PgVectorStore(settings)
        db_dimensions = self.store.get_embedding_dimensions()
        self.embedder = build_embedder(
            settings,
            force_local=force_local_embeddings,
            dimensions_override=db_dimensions,
        )
        self.retriever = HybridRetriever(
            settings=settings,
            store=self.store,
            embedder=self.embedder,
        )

    def _chat_client(self, model_alias: str | None) -> FoundryChatClient:
        alias = model_alias or self.settings.app.default_model
        config = self.settings.models.get(alias)
        if not config:
            supported = ", ".join(sorted(self.settings.models.keys())) or "none"
            raise ValueError(f"Unknown model alias '{alias}'. Supported: {supported}")
        return FoundryChatClient(config)

    def _build_context(self, chunks: list[RetrievedChunk], max_chars: int) -> str:
        if not chunks:
            return "No retrieved context."

        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            clipped = chunk.text[:max_chars].strip()
            lines.append(
                f"[C{idx}] source={chunk.source} title={chunk.title} chunk_index={chunk.chunk_index}\n{clipped}"
            )
        return "\n\n".join(lines)

    def ask(
        self,
        question: str,
        *,
        model_alias: str | None = None,
        retrieval_overrides: RetrievalOverrides | None = None,
        history: list[tuple[str, str]] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> Answer:
        overrides = retrieval_overrides or RetrievalOverrides()
        retrieval = self.retriever.retrieve(question, overrides=overrides)

        profile_name = retrieval.trace.get("profile", self.settings.retrieval.default_profile)
        profile = self.settings.retrieval.profiles[profile_name]

        max_context = min(profile.context.max_chunks, self.settings.app.max_context_chunks)
        selected_chunks = retrieval.chunks[:max_context]
        context_text = self._build_context(
            chunks=selected_chunks,
            max_chars=profile.context.max_chars_per_chunk,
        )

        history_text = ""
        if history:
            lines: list[str] = []
            for user_turn, assistant_turn in history[-6:]:
                lines.append(f"User: {user_turn}")
                lines.append(f"Assistant: {assistant_turn}")
            history_text = "\n".join(lines)

        messages = [
            {"role": "system", "content": self.settings.app.system_prompt},
            {
                "role": "user",
                "content": (
                    "Answer using the provided context when relevant. "
                    "If context is insufficient, say so clearly. "
                    "When you use context, cite chunk ids like [C1], [C2].\n\n"
                    f"Conversation history:\n{history_text or 'No prior conversation.'}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context_text}"
                ),
            },
        ]

        text = self._chat_client(model_alias).complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return Answer(text=text, chunks=selected_chunks, retrieval_trace=retrieval.trace)
