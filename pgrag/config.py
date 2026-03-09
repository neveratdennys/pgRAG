from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    default_model: str
    max_context_chunks: int
    system_prompt: str


@dataclass
class ModelConfig:
    provider: str
    endpoint: str
    api_key: str
    model: str
    api_version: str
    chat_path: str = "/chat/completions"


@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int | None = None
    timeout_seconds: int = 60

    # Ollama fields
    base_url: str = "http://127.0.0.1:11434"
    keep_alive: str | None = None

    # Azure Foundry fields
    endpoint: str | None = None
    api_key: str | None = None
    api_version: str = "2024-05-01-preview"
    embed_path: str = "/embeddings"


@dataclass
class DatabaseConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslmode: str = "disable"
    embedding_type: str = "halfvec"

    def connection_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode,
        }


@dataclass
class VectorSearchConfig:
    k: int = 40
    metric: str = "cosine"
    ef_search: int | None = 80
    min_score: float = 0.0


@dataclass
class LexicalSearchConfig:
    k: int = 40
    language: str = "english"
    min_score: float = 0.0


@dataclass
class HybridFusionConfig:
    mode: str = "rrf"
    vector_weight: float = 0.6
    lexical_weight: float = 0.4
    rrf_k: int = 60


@dataclass
class SemanticRerankerConfig:
    mode: str = "none"
    model_alias: str | None = None
    top_n: int = 40
    keep_n: int = 8
    min_score: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 800


@dataclass
class ContextAssemblyConfig:
    max_chunks: int = 8
    max_chars_per_chunk: int = 1200


@dataclass
class RetrievalProfileConfig:
    vector: VectorSearchConfig
    lexical: LexicalSearchConfig
    hybrid: HybridFusionConfig
    reranker: SemanticRerankerConfig
    context: ContextAssemblyConfig
    final_k: int = 8


@dataclass
class RetrievalConfig:
    default_profile: str
    profiles: dict[str, RetrievalProfileConfig]


@dataclass
class Settings:
    app: AppConfig
    models: dict[str, ModelConfig]
    embeddings: EmbeddingConfig | None
    database: DatabaseConfig
    retrieval: RetrievalConfig


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if _is_placeholder(value):
        raise ValueError(f"Missing '{key}' in '{section}' section.")
    return value


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not normalized:
        return True
    return normalized.startswith(("REPLACE_ME", "YOUR_"))


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_optional_int(value: Any) -> int | None:
    if value in (None, "", "auto", "AUTO", "Auto"):
        return None
    return int(value)


def _normalize_hybrid_weights(vector_weight: float, lexical_weight: float) -> tuple[float, float]:
    if vector_weight < 0:
        vector_weight = 0.0
    if lexical_weight < 0:
        lexical_weight = 0.0

    total = vector_weight + lexical_weight
    if total <= 0:
        return (0.6, 0.4)

    return (vector_weight / total, lexical_weight / total)


def _parse_retrieval_profile(
    app: AppConfig,
    name: str,
    raw_profile: dict[str, Any] | None,
) -> RetrievalProfileConfig:
    raw = raw_profile or {}

    final_k = _as_int(raw.get("final_k"), app.max_context_chunks)

    vector_raw = raw.get("vector", {})
    vector = VectorSearchConfig(
        k=_as_int(vector_raw.get("k"), max(24, final_k * 5)),
        metric=str(vector_raw.get("metric", "cosine")).lower(),
        ef_search=(
            None
            if vector_raw.get("ef_search") in (None, "", 0)
            else _as_int(vector_raw.get("ef_search"), 80)
        ),
        min_score=_as_float(vector_raw.get("min_score"), 0.0),
    )

    lexical_raw = raw.get("lexical", {})
    lexical = LexicalSearchConfig(
        k=_as_int(lexical_raw.get("k"), max(24, final_k * 5)),
        language=str(lexical_raw.get("language", "english")).lower(),
        min_score=_as_float(lexical_raw.get("min_score"), 0.0),
    )

    hybrid_raw = raw.get("hybrid", {})
    vector_weight, lexical_weight = _normalize_hybrid_weights(
        _as_float(hybrid_raw.get("vector_weight"), 0.6),
        _as_float(hybrid_raw.get("lexical_weight"), 0.4),
    )
    hybrid = HybridFusionConfig(
        mode=str(hybrid_raw.get("mode", "rrf")).lower(),
        vector_weight=vector_weight,
        lexical_weight=lexical_weight,
        rrf_k=_as_int(hybrid_raw.get("rrf_k"), 60),
    )

    reranker_raw = raw.get("reranker", {})
    reranker = SemanticRerankerConfig(
        mode=str(reranker_raw.get("mode", "none")).lower(),
        model_alias=reranker_raw.get("model_alias"),
        top_n=_as_int(reranker_raw.get("top_n"), max(32, final_k * 5)),
        keep_n=_as_int(reranker_raw.get("keep_n"), final_k),
        min_score=_as_float(reranker_raw.get("min_score"), 0.0),
        temperature=_as_float(reranker_raw.get("temperature"), 0.0),
        max_tokens=_as_int(reranker_raw.get("max_tokens"), 800),
    )

    context_raw = raw.get("context", {})
    context = ContextAssemblyConfig(
        max_chunks=_as_int(context_raw.get("max_chunks"), app.max_context_chunks),
        max_chars_per_chunk=_as_int(context_raw.get("max_chars_per_chunk"), 1200),
    )

    if vector.metric not in {"cosine", "l2", "ip"}:
        raise ValueError(
            f"Invalid retrieval profile '{name}': vector.metric must be one of cosine/l2/ip."
        )
    if hybrid.mode not in {"rrf", "weighted", "vector_only", "lexical_only"}:
        raise ValueError(
            f"Invalid retrieval profile '{name}': hybrid.mode must be rrf/weighted/vector_only/lexical_only."
        )
    if reranker.mode not in {"none", "llm"}:
        raise ValueError(
            f"Invalid retrieval profile '{name}': reranker.mode must be none/llm."
        )

    return RetrievalProfileConfig(
        vector=vector,
        lexical=lexical,
        hybrid=hybrid,
        reranker=reranker,
        context=context,
        final_k=final_k,
    )


def _parse_retrieval(app: AppConfig, raw_retrieval: dict[str, Any]) -> RetrievalConfig:
    raw_profiles = raw_retrieval.get("profiles")
    if not raw_profiles:
        top_k = _as_int(raw_retrieval.get("top_k"), app.max_context_chunks)
        metric = str(raw_retrieval.get("metric", "cosine")).lower()
        raw_profiles = {
            "balanced": {
                "vector": {"k": max(24, top_k * 5), "metric": metric, "ef_search": 80},
                "lexical": {"k": max(24, top_k * 5), "language": "english"},
                "hybrid": {"mode": "rrf", "vector_weight": 0.6, "lexical_weight": 0.4},
                "reranker": {"mode": "none", "top_n": max(32, top_k * 5), "keep_n": top_k},
                "context": {"max_chunks": app.max_context_chunks, "max_chars_per_chunk": 1200},
                "final_k": top_k,
            }
        }

    profiles: dict[str, RetrievalProfileConfig] = {}
    for name, raw_profile in raw_profiles.items():
        profiles[name] = _parse_retrieval_profile(app=app, name=name, raw_profile=raw_profile)

    default_profile = str(raw_retrieval.get("default_profile", "balanced"))
    if default_profile not in profiles:
        if "balanced" in profiles:
            default_profile = "balanced"
        else:
            first_profile = next(iter(profiles.keys()), None)
            if first_profile is None:
                raise ValueError("No retrieval profiles found in 'retrieval.profiles'.")
            default_profile = first_profile

    return RetrievalConfig(default_profile=default_profile, profiles=profiles)


def load_settings(
    path: str | Path = "config.local.yaml",
    require_model_profiles: bool = True,
) -> Settings:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file '{cfg_path}' not found. Start from config.local.example.yaml."
        )

    raw_text = cfg_path.read_text(encoding="utf-8")
    data = _expand_env(yaml.safe_load(raw_text) or {})

    app_section = data.get("app", {})
    app = AppConfig(
        default_model=_required(app_section, "default_model", "app"),
        max_context_chunks=int(app_section.get("max_context_chunks", 6)),
        system_prompt=app_section.get(
            "system_prompt",
            "You are a grounded assistant that cites retrieved evidence when available.",
        ),
    )

    raw_models = data.get("models", {})
    models: dict[str, ModelConfig] = {}
    default_model_alias = app.default_model
    for alias, raw_model in raw_models.items():
        endpoint = raw_model.get("endpoint")
        api_key = raw_model.get("api_key")
        model = raw_model.get("model")
        if _is_placeholder(endpoint) or _is_placeholder(api_key) or _is_placeholder(model):
            if alias == default_model_alias and require_model_profiles:
                raise ValueError(
                    f"Default model '{alias}' is not fully configured. "
                    "Provide endpoint, api_key, and model."
                )
            continue

        models[alias] = ModelConfig(
            provider=raw_model.get("provider", "azure_foundry"),
            endpoint=str(endpoint),
            api_key=str(api_key),
            model=str(model),
            api_version=raw_model.get("api_version", "2024-05-01-preview"),
            chat_path=raw_model.get("chat_path", "/chat/completions"),
        )

    if require_model_profiles:
        if not raw_models:
            raise ValueError("No model profiles found in 'models' section.")
        if default_model_alias not in models:
            raise ValueError(
                f"Configured default model '{default_model_alias}' is unavailable. "
                f"Valid model aliases: {', '.join(sorted(models.keys())) or 'none'}."
            )

    raw_embeddings = data.get("embeddings")
    embeddings: EmbeddingConfig | None = None
    if raw_embeddings:
        explicit_provider = raw_embeddings.get("provider")
        if explicit_provider:
            provider = str(explicit_provider).lower()
        else:
            # Backward compatibility: if endpoint/api_key are present, assume Foundry.
            if "endpoint" in raw_embeddings or "api_key" in raw_embeddings:
                provider = "azure_foundry"
            else:
                provider = "ollama"

        if provider in {"ollama"}:
            model = raw_embeddings.get("model", "qwen3-embedding:4b")
            if not _is_placeholder(model):
                embeddings = EmbeddingConfig(
                    provider="ollama",
                    model=str(model),
                    dimensions=_as_optional_int(raw_embeddings.get("dimensions")),
                    timeout_seconds=int(raw_embeddings.get("timeout_seconds", 60)),
                    base_url=str(raw_embeddings.get("base_url", "http://127.0.0.1:11434")),
                    keep_alive=(
                        None
                        if raw_embeddings.get("keep_alive") in (None, "")
                        else str(raw_embeddings.get("keep_alive"))
                    ),
                )
        elif provider in {"azure_foundry", "foundry"}:
            endpoint = raw_embeddings.get("endpoint")
            api_key = raw_embeddings.get("api_key")
            model = raw_embeddings.get("model")
            if not (
                _is_placeholder(endpoint)
                or _is_placeholder(api_key)
                or _is_placeholder(model)
            ):
                embeddings = EmbeddingConfig(
                    provider="azure_foundry",
                    endpoint=str(endpoint),
                    api_key=str(api_key),
                    model=str(model),
                    api_version=raw_embeddings.get("api_version", "2024-05-01-preview"),
                    embed_path=raw_embeddings.get("embed_path", "/embeddings"),
                    dimensions=_as_optional_int(raw_embeddings.get("dimensions")),
                    timeout_seconds=int(raw_embeddings.get("timeout_seconds", 60)),
                )
        else:
            raise ValueError(
                f"Unsupported embeddings.provider '{provider}'. "
                "Supported: ollama, azure_foundry."
            )

    db_raw = data.get("database", {})
    database = DatabaseConfig(
        host=_required(db_raw, "host", "database"),
        port=int(db_raw.get("port", 5432)),
        dbname=_required(db_raw, "dbname", "database"),
        user=_required(db_raw, "user", "database"),
        password=_required(db_raw, "password", "database"),
        sslmode=db_raw.get("sslmode", "disable"),
        embedding_type=str(db_raw.get("embedding_type", "halfvec")).lower(),
    )
    if database.embedding_type not in {"vector", "halfvec"}:
        raise ValueError(
            "database.embedding_type must be 'vector' or 'halfvec'."
        )

    retrieval = _parse_retrieval(app=app, raw_retrieval=data.get("retrieval", {}))

    return Settings(
        app=app,
        models=models,
        embeddings=embeddings,
        database=database,
        retrieval=retrieval,
    )
