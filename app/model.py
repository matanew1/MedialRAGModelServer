import requests
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Any

# Load environment variables from .env file (resolve regardless of CWD)
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration from environment variables (ordered fallback)
GROQ_API_KEY1 = (os.getenv("GROQ_API_KEY1") or "").strip()
GROQ_API_KEY2 = (os.getenv("GROQ_API_KEY2") or "").strip()
GROQ_API_KEY3 = (os.getenv("GROQ_API_KEY3") or "").strip()

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "אתה עוזר רפואי מומחה בשפה העברית.")

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def _available_keys() -> List[str]:
    return [k for k in [GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3] if k]

def call_llm(prompt: str) -> str:
    """Call Groq API with ordered fallback: key1 -> key2 -> key3.

    No rotation: each request always prefers key1; only advances to next on failure.
    Failure = network error OR non-200 HTTP OR response parse error.
    Raises aggregated exception if all provided keys fail.
    """
    keys = _available_keys()
    if not keys:
        raise ValueError("No Groq API keys found. Set GROQ_API_KEY1 (and optionally 2/3).")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": LLM_TEMPERATURE
    }

    attempts: List[Tuple[int, Any]] = []  # (index, error info)
    for idx, key in enumerate(keys, start=1):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=60)
        except Exception as e:  # network / timeout
            attempts.append((idx, f"NETWORK_ERROR: {e}"))
            continue

        if resp.status_code != 200:
            body = resp.text
            if len(body) > 200:
                body = body[:200] + "..."
            attempts.append((idx, f"HTTP_{resp.status_code}: {body}"))
            continue

        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as parse_err:
            attempts.append((idx, f"PARSE_ERROR: {parse_err}"))
            continue

    # All failed
    raise Exception(f"All Groq API keys failed: {attempts}")
