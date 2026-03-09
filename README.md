# pgRAG

pgvector-backed RAG prototype with hybrid retrieval, optional semantic reranking, and default local embeddings via Ollama (`qwen3-embedding:4b`).

## What Is Implemented

- Local config system with secrets in `config.local.yaml` (gitignored).
- PostgreSQL + pgvector schema bootstrap (`chunks` table, HNSW vector index, GIN FTS index).
- Ingestion pipeline:
  - parses plain text and HTML-like sources
  - chunks documents
  - creates embeddings (default Ollama, optional Foundry, optional local hash fallback)
  - upserts into PostgreSQL
- Retrieval pipeline:
  - vector retrieval
  - lexical full-text retrieval
  - hybrid fusion (`rrf`, `weighted`, `vector_only`, `lexical_only`)
  - optional LLM semantic reranking
- RAG runtime:
  - `retrieve` for retrieval-only testing
  - `ask` and `chat` for full RAG answer generation
- Retrieval diagnostics:
  - candidate counts
  - stage timing breakdown
  - top-candidate score tracing

## Folder Layout

- `pgrag/config.py`: config models + profile parsing + validation.
- `pgrag/db.py`: database schema, vector/lexical queries, upsert/reindex helpers.
- `pgrag/embeddings.py`: Ollama/Foundry/local-hash embedding providers.
- `pgrag/ingest.py`: parsing/chunking/ingestion and reindex flows.
- `pgrag/ollama.py`: local Ollama embedding client (`/api/embed`).
- `pgrag/retrieval.py`: hybrid retrieval and semantic reranking pipeline.
- `pgrag/rag.py`: prompt/context assembly and model generation flow.
- `pgrag/cli.py`: CLI commands and runtime wiring.
- `config.local.example.yaml`: full config template including retrieval profiles.

## WSL Setup (Ubuntu)

Install PostgreSQL and pgvector:

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-client postgresql-16-pgvector
sudo service postgresql start
```

Create DB role/database and enable extension:

```bash
sudo -u postgres psql -c "CREATE ROLE pgrag WITH LOGIN PASSWORD 'change_me';"
sudo -u postgres psql -c "CREATE DATABASE pgrag OWNER pgrag;"
sudo -u postgres psql -d pgrag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Verify:

```bash
PGPASSWORD='change_me' psql -h localhost -U pgrag -d pgrag -c "SELECT extname FROM pg_extension WHERE extname='vector';"
```

## Project Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.local.example.yaml config.local.yaml
```

Make sure Ollama is running in WSL and the embedding model is available:

```bash
ollama pull qwen3-embedding:4b
ollama list
curl -s http://127.0.0.1:11434/api/version
```

Set DB password used by config env expansion:

```bash
export PGRAG_DB_PASSWORD='change_me'
```

## CLI Commands

Initialize schema:

```bash
python -m pgrag init-db
```

Ingest files:

```bash
python -m pgrag ingest ./docs
```

Reindex stored chunks:

```bash
python -m pgrag reindex
```

Usually not needed when adding new documents; `ingest` already embeds new chunks. Typical uses: after changing the embedding model/provider, switching between local and remote embeddings, or regenerating vectors for existing rows after a schema/config migration.

Retrieval-only test (no generation model required):

```bash
python -m pgrag retrieve "What is UDP stencil component library?" \
  --profile balanced \
  --hybrid-mode rrf \
  --vector-k 40 \
  --lexical-k 40 \
  --final-k 8 \
  --rerank-mode none \
  --debug-retrieval \
  --show-context
```

Single-shot RAG answer:

```bash
python -m pgrag ask "What is our SDLC release process?" \
  --profile balanced \
  --debug-retrieval \
  --show-context
```

Interactive chat:

```bash
python -m pgrag chat --profile high_precision --debug-retrieval
```

## Ollama Embedding Demo (Direct API)

Use these commands to demo that local Ollama is serving `qwen3-embedding:4b` embeddings directly, outside pgRAG.

Basic request:

```bash
curl -s http://127.0.0.1:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding:4b",
    "input": "Embed this sentence for semantic search demo."
  }'
```

Show only embedding vector length (quick sanity check):

```bash
curl -s http://127.0.0.1:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding:4b",
    "input": "Embed this sentence for semantic search demo."
  }' | python -c "import sys, json; d=json.load(sys.stdin); print(len(d['embeddings'][0]))"
```

Compare two texts and print cosine similarity:

```bash
python - <<'PY'
import math
import requests

URL = "http://127.0.0.1:11434/api/embed"
MODEL = "qwen3-embedding:4b"
texts = [
    "What is the SDLC process overview?",
    "Explain lifecycle phases in software delivery.",
]

vecs = []
for t in texts:
    r = requests.post(URL, json={"model": MODEL, "input": t}, timeout=120)
    r.raise_for_status()
    vecs.append(r.json()["embeddings"][0])

dot = sum(a*b for a, b in zip(vecs[0], vecs[1]))
n1 = math.sqrt(sum(a*a for a in vecs[0]))
n2 = math.sqrt(sum(b*b for b in vecs[1]))
print("dimensions:", len(vecs[0]))
print("cosine_similarity:", dot / (n1 * n2))
PY
```

## Retrieval Profiles and Tuning

`config.local.yaml` supports named profiles under `retrieval.profiles`. Each profile includes:

- `vector`: `k`, `metric`, `ef_search`, `min_score`.
- `lexical`: `k`, `language`, `min_score`.
- `hybrid`: `mode`, `vector_weight`, `lexical_weight`, `rrf_k`.
- `reranker`: `mode`, `model_alias`, `top_n`, `keep_n`, `min_score`, `temperature`, `max_tokens`.
- `context`: `max_chunks`, `max_chars_per_chunk`.
- `final_k`.

Runtime overrides:

- `--profile`
- `--hybrid-mode`
- `--vector-k`
- `--lexical-k`
- `--final-k`
- `--rerank-mode`
- `--rerank-top-n`
- `--rerank-model`
- `--source-filter`
- `--debug-retrieval`

## pgvector Index Note for 4B Embeddings

`qwen3-embedding:4b` currently returns 2560-dim vectors in this setup.

Use `database.embedding_type: halfvec` (default in current config) to support 2000+ dimensions with ANN indexing.

`init-db` will attempt HNSW index creation with `halfvec_cosine_ops`. If index creation fails in your environment, it automatically falls back to:

- `vector_index=none`
- `vector_index_mode=exact_scan`

The CLI output prints the selected mode after `init-db`.

## Preliminary Validation (Completed)

A local preliminary run was completed with current sample docs:

- DB schema init: success.
- Sample docs ingested: `3` files, `4` chunks.
- Hybrid retrieval sanity check: success with relevant top result from `udp-stencil-component-library.txt`.
- Retrieval trace output includes per-stage timings and candidate counts.
- Ollama embedding provider path validated (`qwen3-embedding:4b`, 2560 dims).

Note:

- Embeddings default to Ollama provider from `config.local.yaml`.
- Use `--use-local-embeddings` only when you explicitly want deterministic local-hash fallback.
- `ask`/`chat` require valid Foundry model credentials in `config.local.yaml`.
- `retrieve` works without generation credentials and is recommended for early tuning.

## Security

- `config.local.yaml` is ignored by git.
- `docs/` is ignored by git (prototype local corpus).
- Never commit API keys.
