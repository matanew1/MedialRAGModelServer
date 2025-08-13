import requests
import os
import threading
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Any
import logging

# Configure logging with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration
GROQ_API_KEY1 = (os.getenv("GROQ_API_KEY1") or "").strip()
GROQ_API_KEY2 = (os.getenv("GROQ_API_KEY2") or "").strip()
GROQ_API_KEY3 = (os.getenv("GROQ_API_KEY3") or "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
אתה עוזר רפואי חכם ומומחה. על סמך הנתונים, ענה בצורה מאוזנת ומעשית:
1. ענה בעברית ברורה ומקצועית
2. התחל עם אפשרויות נפוצות ופשוטות
3. ציין שכיחות המחלה (נפוץ/שכיח/נדיר)
4. כלול מצבים קלים ונפוצים, לא רק חמורים
5. תן הקשר - דחוף או לא
6. ציין מתי לפנות לרופא
7. תן המלצות מעשיות לטיפול ראשוני
8. הרגע את המשתמש, אל תפחיד ללא סיבה
""")

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

class KeyRotator:
    def __init__(self):
        self._current_index = 0
        self._lock = threading.Lock()
        
    def get_next_key_index(self, available_keys: List[str]) -> int:
        if not available_keys:
            return 0
        with self._lock:
            current = self._current_index
            self._current_index = (self._current_index + 1) % len(available_keys)
            return current
    
    def get_ordered_keys_from_index(self, available_keys: List[str], start_index: int) -> List[Tuple[int, str]]:
        if not available_keys:
            return []
        ordered = []
        for i in range(len(available_keys)):
            actual_index = (start_index + i) % len(available_keys)
            ordered.append((actual_index, available_keys[actual_index]))
        return ordered

_key_rotator = KeyRotator()

def _available_keys() -> List[str]:
    return [k for k in [GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3] if k]

def validate_env_vars():
    if not any(_available_keys()):
        raise ValueError("At least one GROQ_API_KEY must be set in .env")
    logger.info(f"Validated {len(_available_keys())} Groq API keys")

def call_llm(prompt: str) -> str:
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

    start_index = _key_rotator.get_next_key_index(keys)
    ordered_keys = _key_rotator.get_ordered_keys_from_index(keys, start_index)

    attempts: List[Tuple[int, Any]] = []
    for original_index, key in ordered_keys:
        key_number = original_index + 1
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        try:
            resp = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=30)
            logger.info(f"API call with key {key_number} status: {resp.status_code}")
            if resp.status_code != 200:
                body = resp.text[:200] + "..." if len(resp.text) > 200 else resp.text
                logger.error(f"HTTP error with key {key_number}: {resp.status_code} - {body}")
                attempts.append((key_number, f"HTTP_{resp.status_code}: {body}"))
                continue
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            logger.error(f"Network error with key {key_number}: {e}", exc_info=True)
            attempts.append((key_number, f"NETWORK_ERROR: {e}"))
            continue
        except (KeyError, ValueError) as parse_err:
            logger.error(f"Parse error with key {key_number}: {parse_err}", exc_info=True)
            attempts.append((key_number, f"PARSE_ERROR: {parse_err}"))
            continue

    logger.error(f"All Groq API keys failed: {attempts}", exc_info=True)
    raise Exception(f"All Groq API keys failed: {attempts}")

def get_current_key_stats() -> dict:
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