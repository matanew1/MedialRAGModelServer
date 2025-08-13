from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

try:
    from .rag import get_answer, get_answer_with_debug
    from .model import get_current_key_stats, validate_env_vars
except ImportError:
    from rag import get_answer, get_answer_with_debug
    from model import get_current_key_stats, validate_env_vars

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Ensure Windows console supports UTF-8
if os.name == 'nt':
    import sys
    import locale
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    try:
        locale.setlocale(locale.LC_ALL, 'he_IL.UTF-8')
    except locale.Error:
        logger.warning("Failed to set locale to he_IL.UTF-8, falling back to default UTF-8 handling")

# Load environment variables
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration
API_VERSION = os.getenv("API_VERSION", "1.0.0")
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://localhost:8080,http://localhost:3000").split(",")

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting RAG Medical API v{API_VERSION} in {ENVIRONMENT} mode")
    try:
        validate_env_vars()
    except Exception as e:
        logger.error(f"Startup validation failed: {e}", exc_info=True)
        raise
    yield
    logger.info("Shutting down RAG Medical API")

app = FastAPI(
    title="RAG Medical Diagnosis API",
    description="""
A Retrieval-Augmented Generation (RAG) system for medical diagnosis assistance in Hebrew.
Features:
- Hebrew Language Support: Native Hebrew medical terminology and responses
- Vector Search: FAISS-powered similarity search for relevant medical conditions
- LLM Integration: Groq API with Meta Llama models for intelligent responses
- Medical Knowledge Base: Comprehensive database of diseases and symptoms in Hebrew

⚠️ Medical Disclaimer: This system is for educational purposes only and should not replace professional medical advice.
    """,
    version=API_VERSION,
    contact={"name": "API Support", "email": "support@example.com"},
    license_info={"name": "Educational Use", "url": "https://opensource.org/licenses/MIT"},
    servers=[
        {"url": "http://localhost:8000", "description": "Local development server"},
        {"url": "https://cicada-helpful-chipmunk.ngrok-free.app", "description": "Public ngrok tunnel"}
    ],
    tags_metadata=[
        {
            "name": "Medical Diagnosis",
            "description": "Core medical diagnosis functionality using RAG."
        },
        {
            "name": "System",
            "description": "System monitoring and health check endpoints."
        }
    ],
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=MAX_QUERY_LENGTH,
        description="Medical query in Hebrew",
        examples=["יש לי כאב ראש"]
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "יש לי כאב ראש"},
                {"question": "יש לי חום וכאב גרון"},
                {"question": "כואב לי הבטן כבר שעתיים"}
            ]
        }
    )

class ResponseMetadata(BaseModel):
    response_time_ms: Optional[int]
    retrieved_conditions: Optional[int]
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    debug: Optional[dict]

class DiagnosisResponse(BaseModel):
    question: str
    answer: str
    metadata: Optional[ResponseMetadata]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class KeyStatsResponse(BaseModel):
    available_keys: int
    current_index: int
    next_key: str
    total_keys_configured: int

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str]
    timestamp: Optional[str]

# API Endpoints
@app.get("/", summary="Service Root", tags=["System"])
async def root():
    return {
        "name": "RAG Medical Diagnosis API",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "health": "/health",
        "key_stats": "/key-stats",
        "diagnose": "/diagnose",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check", tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        version=API_VERSION
    )

@app.get("/key-stats", response_model=KeyStatsResponse, summary="API Key Rotation Statistics", tags=["System"])
async def get_key_stats():
    try:
        stats = get_current_key_stats()
        return KeyStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting key stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to retrieve API key statistics")

@app.post("/diagnose", response_model=DiagnosisResponse, summary="Medical Diagnosis Query", tags=["Medical Diagnosis"])
async def diagnose(request: QueryRequest, debug: bool = Query(False)):
    start_time = time.time()
    logger.info(f"Processing query: {request.question[:50]}...")
    
    try:
        if debug:
            answer, debug_info, retrieved_count, avg_confidence = get_answer_with_debug(request.question)
        else:
            answer, retrieved_count, avg_confidence = get_answer(request.question)
            debug_info = None
        
        # Ensure confidence_score is within [0, 1]
        if avg_confidence is not None:
            avg_confidence = max(0.0, min(1.0, avg_confidence))
        
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Query processed in {response_time_ms}ms")
        
        return DiagnosisResponse(
            question=request.question,
            answer=answer,
            metadata=ResponseMetadata(
                response_time_ms=response_time_ms,
                retrieved_conditions=retrieved_count,
                confidence_score=avg_confidence,
                debug=debug_info
            )
        )
    except Exception as e:
        logger.error(f"Error processing query '{request.question}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail=f"Failed to process query: {str(e)}",
                error_code="INTERNAL_ERROR",
                timestamp=datetime.now(UTC).isoformat()
            ).model_dump()
        )

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc.detail}", exc_info=True)
    return JSONResponse(status_code=422, content=exc.detail if hasattr(exc, 'detail') else {"detail": "Validation error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)