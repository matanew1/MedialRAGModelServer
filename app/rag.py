import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from model import call_llm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Configuration from environment variables
DATA_PATH = os.getenv("DATA_PATH", "../data/diseases.json")
INDEX_PATH = os.getenv("INDEX_PATH", "../index/faiss.index")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "distiluse-base-multilingual-cased")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "3"))

# Load model + data once at startup
embedder = SentenceTransformer(EMBEDDER_MODEL)
with open(DATA_PATH, encoding="utf-8") as f:
    medical_data = json.load(f)["data"]

texts = [item["disease"] + " " + item.get("symptoms", "") for item in medical_data]
embeddings = embedder.encode(texts, convert_to_numpy=True)

# Build FAISS index if not exists
if not os.path.exists(INDEX_PATH):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
else:
    index = faiss.read_index(INDEX_PATH)

def get_answer(query: str):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k=VECTOR_SEARCH_TOP_K)

    context_items = [medical_data[i] for i in indices[0]]
    context_str = "\n\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in context_items)

    prompt = f"""אתה עוזר רפואי חכם. על סמך הנתונים הבאים, ענה על השאלה בצורה ברורה וקצרה:
    
שאלה: {query}

מידע:
{context_str}

תשובה:"""

    return call_llm(prompt)
