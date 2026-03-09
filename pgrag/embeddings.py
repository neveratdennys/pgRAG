from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from pgrag.config import Settings
from pgrag.foundry import FoundryEmbeddingClient
from pgrag.ollama import OllamaEmbeddingClient


class Embedder:
    dimensions: int | None

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


@dataclass
class FoundryEmbedder(Embedder):
    client: FoundryEmbeddingClient
    dimensions: int | None

    def embed(self, text: str) -> list[float]:
        values = self.client.embed(text)
        if self.dimensions is None:
            self.dimensions = len(values)
        if len(values) != self.dimensions:
            raise ValueError(
                "Embedding vector size mismatch. "
                f"Expected {self.dimensions}, got {len(values)}."
            )
        return values


@dataclass
class OllamaEmbedder(Embedder):
    client: OllamaEmbeddingClient
    dimensions: int | None

    def embed(self, text: str) -> list[float]:
        values = self.client.embed(text)
        if self.dimensions is None:
            self.dimensions = len(values)
        if len(values) != self.dimensions:
            raise ValueError(
                "Embedding vector size mismatch. "
                f"Expected {self.dimensions}, got {len(values)}."
            )
        return values


@dataclass
class LocalHashEmbedder(Embedder):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        dims = self.dimensions
        if dims <= 0:
            raise ValueError("Local hash embedding dimensions must be > 0.")

        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return [0.0] * dims

        values = [0.0] * dims
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[idx] += sign

            idx2 = int.from_bytes(digest[8:12], byteorder="big", signed=False) % dims
            values[idx2] += 0.5 * sign

        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            return values
        return [v / norm for v in values]


def build_embedder(
    settings: Settings,
    *,
    force_local: bool = False,
    dimensions_override: int | None = None,
) -> Embedder:
    configured_dimensions = dimensions_override
    if configured_dimensions is None and settings.embeddings:
        configured_dimensions = settings.embeddings.dimensions

    if force_local:
        return LocalHashEmbedder(dimensions=configured_dimensions or 1536)

    if settings.embeddings:
        provider = settings.embeddings.provider.lower()
        if provider in {"azure_foundry", "foundry"}:
            return FoundryEmbedder(
                client=FoundryEmbeddingClient(settings.embeddings),
                dimensions=configured_dimensions,
            )
        if provider == "ollama":
            return OllamaEmbedder(
                client=OllamaEmbeddingClient(settings.embeddings),
                dimensions=configured_dimensions,
            )
        raise ValueError(
            f"Unsupported embeddings.provider '{settings.embeddings.provider}'."
        )

    return LocalHashEmbedder(dimensions=configured_dimensions or 1536)


def infer_embedding_dimensions(settings: Settings) -> int:
    embedder = build_embedder(settings, force_local=False, dimensions_override=None)
    vector = embedder.embed("dimension probe")
    return len(vector)
