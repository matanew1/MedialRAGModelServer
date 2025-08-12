import requests
import os
import threading
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

# Thread-safe cyclic key rotation state
class KeyRotator:
    def __init__(self):
        self._current_index = 0
        self._lock = threading.Lock()
        
    def get_next_key_index(self, available_keys: List[str]) -> int:
        """Get next key index in round-robin fashion (thread-safe)."""
        if not available_keys:
            return 0
            
        with self._lock:
            current = self._current_index
            self._current_index = (self._current_index + 1) % len(available_keys)
            return current
    
    def get_ordered_keys_from_index(self, available_keys: List[str], start_index: int) -> List[Tuple[int, str]]:
        """Get keys in order starting from start_index, with original indices."""
        if not available_keys:
            return []
            
        # Create ordered list starting from start_index
        ordered = []
        for i in range(len(available_keys)):
            actual_index = (start_index + i) % len(available_keys)
            ordered.append((actual_index, available_keys[actual_index]))
        return ordered

# Global key rotator instance
_key_rotator = KeyRotator()

def _available_keys() -> List[str]:
    return [k for k in [GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3] if k]

def call_llm(prompt: str) -> str:
    """Call Groq API with cyclic key rotation and fallback.

    Uses round-robin rotation to distribute load across all available keys.
    On each call, starts with the next key in cycle, but tries all keys if needed.
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

    # Get the next key in rotation and create ordered attempt list
    start_index = _key_rotator.get_next_key_index(keys)
    ordered_keys = _key_rotator.get_ordered_keys_from_index(keys, start_index)

    attempts: List[Tuple[int, Any]] = []  # (key_number, error info)
    for original_index, key in ordered_keys:
        key_number = original_index + 1  # Human-readable key number (1, 2, 3)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        try:
            resp = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=60)
        except Exception as e:  # network / timeout
            attempts.append((key_number, f"NETWORK_ERROR: {e}"))
            continue

        if resp.status_code != 200:
            body = resp.text
            if len(body) > 200:
                body = body[:200] + "..."
            attempts.append((key_number, f"HTTP_{resp.status_code}: {body}"))
            continue

        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as parse_err:
            attempts.append((key_number, f"PARSE_ERROR: {parse_err}"))
            continue

    # All failed
    raise Exception(f"All Groq API keys failed: {attempts}")


def get_current_key_stats() -> dict:
    """Get current API key rotation statistics (useful for debugging/monitoring)."""
    keys = _available_keys()
    if not keys:
        return {"available_keys": 0, "current_index": 0, "next_key": "None"}
    
    with _key_rotator._lock:
        current_idx = _key_rotator._current_index
        
    return {
        "available_keys": len(keys),
        "current_index": current_idx,
        "next_key": f"GROQ_API_KEY{current_idx + 1}",
        "total_keys_configured": sum(1 for k in [GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3] if k)
    }
