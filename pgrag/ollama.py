from __future__ import annotations

from typing import Any

import requests

from pgrag.config import EmbeddingConfig


class OllamaError(RuntimeError):
    pass


class OllamaEmbeddingClient:
    def __init__(self, embed_config: EmbeddingConfig) -> None:
        self.embed_config = embed_config
        self.base_url = embed_config.base_url.rstrip("/")
        self.timeout_seconds = int(embed_config.timeout_seconds)

    def embed(self, text: str) -> list[float]:
        payload: dict[str, Any] = {
            "model": self.embed_config.model,
            "input": text,
        }
        if self.embed_config.keep_alive:
            payload["keep_alive"] = self.embed_config.keep_alive

        url = f"{self.base_url}/api/embed"
        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise OllamaError(f"Ollama embed failed ({response.status_code}): {response.text}")

        data = response.json()
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                return [float(v) for v in first]

        legacy = data.get("embedding")
        if isinstance(legacy, list):
            return [float(v) for v in legacy]

        raise OllamaError(f"Unexpected Ollama embeddings payload: {data}")
