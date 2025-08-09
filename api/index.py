# Serverless entry for Vercel
# Exposes the FastAPI app through Mangum (ASGI -> AWS Lambda style) which Vercel's Python runtime invokes.

from mangum import Mangum
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Ensure env loaded (Vercel sets env vars directly; .env for local dev fallback)
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Import the application (must be after env load to apply any settings)
from app.main import app  # noqa: E402

# Create handler for serverless
handler = Mangum(app)

# Allow local execution for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
