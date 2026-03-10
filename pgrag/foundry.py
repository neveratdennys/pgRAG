from __future__ import annotations

from typing import Any

import requests

from pgrag.config import EmbeddingConfig, ModelConfig


class FoundryError(RuntimeError):
    pass


def _build_url(endpoint: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return endpoint.rstrip("/") + "/" + path.lstrip("/")


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content)


def _messages_to_response_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": m.get("role", "user"),
            "content": [{"type": "input_text", "text": m.get("content", "")}],
        }
        for m in messages
    ]


def _extract_responses_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = data.get("output")
    if isinstance(outputs, list):
        parts: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise FoundryError(f"Unexpected responses API payload: {data}")


class FoundryChatClient:
    def __init__(self, model_config: ModelConfig, timeout_seconds: int = 60) -> None:
        self.model_config = model_config
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if "/responses" in self.model_config.chat_path:
            return self._complete_responses_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._complete_chat_completions_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _complete_chat_completions_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = _build_url(self.model_config.endpoint, self.model_config.chat_path)
        params = {"api-version": self.model_config.api_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self.model_config.api_key,
        }
        base_payload: dict[str, Any] = {
            "model": self.model_config.model,
            "messages": messages,
        }
        payloads = [
            {**base_payload, "temperature": temperature, "max_completion_tokens": max_tokens},
            {**base_payload, "temperature": temperature, "max_tokens": max_tokens},
            {**base_payload, "temperature": temperature},
            {**base_payload, "max_completion_tokens": max_tokens},
            {**base_payload, "max_tokens": max_tokens},
            base_payload,
        ]

        data: dict[str, Any] | None = None
        last_error: str | None = None
        for payload in payloads:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code < 400:
                data = response.json()
                break
            last_error = f"({response.status_code}) {response.text}"
            if response.status_code >= 500:
                break

        if data is None:
            raise FoundryError(f"Chat request failed: {last_error or 'unknown error'}")

        choices = data.get("choices") or []
        if not choices:
            raise FoundryError(f"Unexpected chat response: {data}")

        message = choices[0].get("message", {})
        return _coerce_text(message.get("content", "")).strip()

    def _complete_responses_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = _build_url(self.model_config.endpoint, self.model_config.chat_path)
        params = {"api-version": self.model_config.api_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self.model_config.api_key,
        }
        response_input = _messages_to_response_input(messages)
        payloads = [
            {
                "model": self.model_config.model,
                "input": response_input,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
            {
                "model": self.model_config.model,
                "input": response_input,
                "temperature": temperature,
            },
            {
                "model": self.model_config.model,
                "input": response_input,
                "max_output_tokens": max_tokens,
            },
            {
                "model": self.model_config.model,
                "input": response_input,
            },
            {
                "model": self.model_config.model,
                "input": "\n\n".join(
                    [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
                ),
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
            {
                "model": self.model_config.model,
                "input": "\n\n".join(
                    [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
                ),
            },
        ]

        data: dict[str, Any] | None = None
        last_error: str | None = None
        for payload in payloads:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code < 400:
                data = response.json()
                break
            last_error = f"({response.status_code}) {response.text}"
            if response.status_code >= 500:
                break

        if data is None:
            raise FoundryError(
                f"Responses API request failed: {last_error or 'unknown error'}"
            )
        return _extract_responses_text(data)


class FoundryEmbeddingClient:
    def __init__(self, embed_config: EmbeddingConfig, timeout_seconds: int | None = None) -> None:
        if embed_config.provider not in {"azure_foundry", "foundry"}:
            raise FoundryError(
                f"FoundryEmbeddingClient requires provider=azure_foundry, got '{embed_config.provider}'."
            )
        if not embed_config.endpoint or not embed_config.api_key:
            raise FoundryError(
                "Foundry embedding config requires non-empty 'endpoint' and 'api_key'."
            )
        self.embed_config = embed_config
        self.timeout_seconds = timeout_seconds or int(embed_config.timeout_seconds)

    def embed(self, text: str) -> list[float]:
        endpoint = self.embed_config.endpoint or ""
        url = _build_url(endpoint, self.embed_config.embed_path)
        params = {"api-version": self.embed_config.api_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self.embed_config.api_key,
        }
        payloads = [
            {"model": self.embed_config.model, "input": [text]},
            {"model": self.embed_config.model, "input": text},
        ]
        data: dict[str, Any] | None = None
        last_error: str | None = None
        for payload in payloads:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code < 400:
                data = response.json()
                break
            last_error = f"({response.status_code}) {response.text}"
            if response.status_code >= 500:
                break

        if data is None:
            raise FoundryError(
                f"Embedding request failed: {last_error or 'unknown error'}"
            )

        items = data.get("data") or []
        if not items:
            raise FoundryError(f"Unexpected embeddings response: {data}")

        vector = items[0].get("embedding")
        if not isinstance(vector, list):
            raise FoundryError(f"Embedding vector missing in response: {data}")

        return [float(v) for v in vector]
