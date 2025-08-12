# RAG Medical Diagnosis API

Hebrew medical assistance via Retrieval‑Augmented Generation (FastAPI + Sentence Transformers + FAISS + Groq LLM). Answers are grounded ONLY in the curated JSON knowledge base.

---

## Architecture

The RAG Medical Diagnosis API is built on a modular architecture that leverages several key components:

1. **FastAPI**: The web framework for building the API, providing a simple and efficient way to handle HTTP requests and responses.
2. **Sentence Transformers**: Used for generating embeddings from user queries and medical documents, enabling semantic search capabilities.
3. **FAISS**: A library for efficient similarity search and clustering of dense vectors, allowing for fast retrieval of relevant medical information.
4. **Groq LLM**: The language model used for generating natural language responses based on the retrieved information.

## UML + Sequence Diagrams

![UML](images/UML.png)
![Sequence Diagram](images/sequenceDiagram.png)

## ✨ Key Features

- Lazy load of embedder + FAISS index (fast cold deploy memory profile)
- Exact vector search (FAISS IndexFlatL2) for deterministic recall
- Simple `/diagnose` endpoint with optional `debug=true` metadata
- API key monitoring via `/key-stats` endpoint
- Round-robin API key rotation for load distribution
- Docker & docker‑compose ready (API + optional nginx)
- Lightweight test suite (pytest + monkeypatched heavy parts)

## ⚡ Quick Start

### Local

```bash
git clone <repo-url>
cd MedialRAGModelServer
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # Edit GROQ_API_KEY1
python app/main.py
```

Open: http://localhost:8000 (Docs at /docs)

### Docker (API only)

```bash
docker compose up -d rag-medical-api
```

### Docker (API + nginx proxy)

```bash
docker compose --profile production up -d
```

### Expose Publicly (ngrok)

Custom reserved domain example:

```bash
ngrok http --url=cicada-helpful-chipmunk.ngrok-free.app 8000
```

## 🔌 Test

Local:

```bash
pytest -q
```

Ephemeral container:

```bash
docker compose --profile test run --rm tests
```

## 🧪 Example Request

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"question": "יש לי כאב ראש"}'
```

Add `?debug=true` for retrieval diagnostics.

## ⚙️ Environment Variables (common)

| Variable            | Default                                   | Notes                                             |
| ------------------- | ----------------------------------------- | ------------------------------------------------- |
| GROQ_API_KEY1       | (required)                                | Primary Groq auth token                           |
| GROQ_API_KEY2       | (empty)                                   | Optional fallback (used only if key1 fails)       |
| GROQ_API_KEY3       | (empty)                                   | Optional fallback (used only if key1 & key2 fail) |
| GROQ_MODEL          | meta-llama/llama-4-scout-17b-16e-instruct | LLM model                                         |
| LLM_TEMPERATURE     | 0.3                                       | 0.0–1.0 creativity                                |
| EMBEDDER_MODEL      | distiluse-base-multilingual-cased         | Sentence embeddings                               |
| VECTOR_SEARCH_TOP_K | 3                                         | Retrieval depth                                   |
| DATA_PATH           | data/diseases.json                        | Knowledge base                                    |
| INDEX_PATH          | index/faiss.index                         | Persisted FAISS index                             |
| RAG_LAZY_INIT       | true                                      | Defer heavy load                                  |
| PORT                | 8000                                      | Service port                                      |

### API Key Cyclic Rotation

The LLM adapter now uses **round-robin (cyclic) rotation** across all available API keys for optimal load distribution:

**Key Features:**

- **True Rotation**: Each request uses the next key in sequence, cycling back to the first key
- **Thread-Safe**: Works correctly with FastAPI's concurrent request handling
- **Fallback on Failure**: If the selected key fails, tries remaining keys in order
- **Load Distribution**: Evenly distributes requests across all available keys
- **Monitoring**: New `/key-stats` endpoint shows rotation status

**Behavior:**

1. Request 1 → `GROQ_API_KEY1`, Request 2 → `GROQ_API_KEY2`, Request 3 → `GROQ_API_KEY3`, Request 4 → `GROQ_API_KEY1` (cycle repeats)
2. If `GROQ_API_KEY2` fails on Request 2, tries `GROQ_API_KEY3`, then `GROQ_API_KEY1`, but Request 3 still starts with `GROQ_API_KEY3`
3. Failed keys are logged with detailed error information (network, HTTP status, or parse errors)

**Monitoring:**

- `GET /key-stats` - Returns current rotation state and statistics
- Logs show which key was used for each request and any failures

Example `.env` snippet:

```
GROQ_API_KEY1=primary_key
GROQ_API_KEY2=backup_key
GROQ_API_KEY3=second_backup_key
```

## 🧱 Minimal Project Map

```
app/ (main.py, rag.py, model.py)
data/diseases.json
index/faiss.index (generated)
docker-compose.yml
Dockerfile
```

## 🐳 Common Docker Commands

| Goal      | Command                                      |
| --------- | -------------------------------------------- |
| API dev   | docker compose up -d rag-medical-api         |
| API+nginx | docker compose --profile production up -d    |
| Run tests | docker compose --profile test run --rm tests |
| Down      | docker compose down                          |

## 🔧 Rebuild Index

Delete `index/faiss.index` and issue any `/diagnose` request (auto‑rebuild + persist).

## 🛡️ Production Suggestions (Brief)

Add auth (API key/JWT), rate limiting (nginx + app), structured logs, request IDs, HTTPS termination, index versioning (hash dataset), multiple workers if CPU allows.

## 🤝 Contributing

Fork → branch → change → tests → PR with rationale.

## ⚠️ Medical Disclaimer

Not a substitute for professional medical diagnosis or treatment.

---

Educational / research use only.
