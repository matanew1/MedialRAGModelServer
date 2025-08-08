import requests
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file (resolve regardless of CWD)
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "אתה עוזר רפואי מומחה בשפה העברית.")

def call_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Set it in your .env or environment variables.")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": LLM_TEMPERATURE
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code}, {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()
