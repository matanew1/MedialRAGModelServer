"""RAG service with lazy initialization to reduce startup memory footprint.

On low-memory platforms (e.g., Render free tier 512Mi), loading the
SentenceTransformer and computing embeddings at import may OOM. This refactor
defers all heavy work until the first query.
"""

import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from model import call_llm
from dotenv import load_dotenv, find_dotenv
from threading import Lock

# Load environment variables from .env file (resolve regardless of CWD)
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "diseases.json"))
INDEX_PATH = os.getenv("INDEX_PATH", str(PROJECT_ROOT / "index" / "faiss.index"))
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "distiluse-base-multilingual-cased")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "3"))
LAZY_INIT = os.getenv("RAG_LAZY_INIT", "true").lower() == "true"

_state_lock = Lock()
_initialized = False
_embedder = None
_medical_data = None
_index = None
_embedding_dim = None
_index_built = False


def _initialize_if_needed():
    global _initialized, _embedder, _medical_data, _index, _embedding_dim, _index_built
    if _initialized and _embedder is not None:
        return
    with _state_lock:
        if _initialized and _embedder is not None:
            return
        # Load data
        with open(DATA_PATH, encoding="utf-8") as f:
            _medical_data = json.load(f)["data"]
        # Load model (lighter model can be chosen via EMBEDDER_MODEL env)
        _embedder = SentenceTransformer(EMBEDDER_MODEL)
        # Build / load index
        if os.path.exists(INDEX_PATH):
            _index = faiss.read_index(INDEX_PATH)
            _embedding_dim = _index.d
        else:
            texts = [item["disease"] + " " + item.get("symptoms", "") for item in _medical_data]
            embeddings = _embedder.encode(texts, convert_to_numpy=True)
            _embedding_dim = int(embeddings.shape[1])
            Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
            _index = faiss.IndexFlatL2(_embedding_dim)
            _index.add(embeddings)
            faiss.write_index(_index, INDEX_PATH)
            _index_built = True
            # Free embeddings array memory explicitly
            del embeddings
        _initialized = True


def _ensure_initialized():
    if not _initialized:
        _initialize_if_needed()


def _build_prompt(query: str, context_items):
    context_str = "\n\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in context_items)
    return f"""אתה עוזר רפואי חכם. על סמך הנתונים הבאים, ענה על השאלה בצורה ברורה וקצרה:
    \nשאלה: {query}\n\nמידע:\n{context_str}\n\nתשובה:"""


def get_answer(query: str):
    _ensure_initialized()
    query_embedding = _embedder.encode([query], convert_to_numpy=True)
    _, indices = _index.search(query_embedding, k=VECTOR_SEARCH_TOP_K)
    context_items = [_medical_data[i] for i in indices[0]]
    prompt = _build_prompt(query, context_items)
    return call_llm(prompt)


def get_answer_with_debug(query: str):
    """Returns (answer, debug_info) including embedding and FAISS search details."""
    _ensure_initialized()
    query_embedding = _embedder.encode([query], convert_to_numpy=True)
    distances, indices = _index.search(query_embedding, k=VECTOR_SEARCH_TOP_K)
    idxs = indices[0].tolist()
    dists = distances[0].tolist()
    context_items = [_medical_data[i] for i in idxs]
    prompt = _build_prompt(query, context_items)
    answer = call_llm(prompt)
    debug_info = {
        "embedder_model": EMBEDDER_MODEL,
        "embedding_dim": _embedding_dim,
        "vector_search_top_k": VECTOR_SEARCH_TOP_K,
        "index_path": INDEX_PATH,
        "index_built": _index_built,
        "indices": idxs,
        "distances": dists,
        "retrieved_items": [
            {"disease": item.get("disease"), "symptoms": item.get("symptoms")}
            for item in context_items
        ],
        "prompt_preview": prompt[:700],
        "lazy_init": LAZY_INIT,
        "initialized": _initialized,
    }
    return answer, debug_info

# Optionally eager initialize if lazy disabled
if not LAZY_INIT:
    _initialize_if_needed()
