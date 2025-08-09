from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from rag import get_answer, get_answer_with_debug
from datetime import datetime, timezone
try:  # Python 3.11+: datetime.UTC; fallback for 3.10
    from datetime import UTC  # type: ignore
except ImportError:  # pragma: no cover - compatibility shim
    UTC = timezone.utc  # type: ignore
import time
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file (resolve regardless of CWD)
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration from environment variables
API_VERSION = os.getenv("API_VERSION", "1.0.0")
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app with detailed metadata matching swagger.yaml
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting RAG Medical API v{API_VERSION} in {ENVIRONMENT} mode")
    logger.info(f"Server will run on {HOST}:{PORT}")
    yield
    # Shutdown
    logger.info("Shutting down RAG Medical API")


app = FastAPI(
    title="RAG Medical Diagnosis API",
    description="""
A Retrieval-Augmented Generation (RAG) system for medical diagnosis assistance in Hebrew.

## Features
- **Hebrew Language Support**: Native Hebrew medical terminology and responses
- **Vector Search**: FAISS-powered similarity search for relevant medical conditions
- **LLM Integration**: Groq API with Meta Llama models for intelligent responses
- **Medical Knowledge Base**: Comprehensive database of diseases and symptoms in Hebrew

## Architecture
The system combines:
1. **Semantic Search** using sentence transformers for query understanding
2. **Vector Database** with FAISS for fast similarity matching
3. **Language Model** via Groq API for natural language generation
4. **Medical Data** structured in JSON format with Hebrew content

## Usage
Send medical queries in Hebrew to get relevant diagnosis information and recommendations.

**⚠️ Medical Disclaimer**: This system is for educational purposes only and should not replace professional medical advice.
    """,
    version=API_VERSION,
    contact={
        "name": "API Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "Educational Use",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        },
        {
            "url": "https://cicada-helpful-chipmunk.ngrok-free.app",
            "description": "Public ngrok tunnel"
        }
    ],
    tags_metadata=[
        {
            "name": "Medical Diagnosis",
            "description": """
Core medical diagnosis functionality using RAG (Retrieval-Augmented Generation).

The system processes Hebrew medical queries through:
1. **Query Analysis**: Understanding the medical symptoms described
2. **Knowledge Retrieval**: Finding relevant medical conditions from the database
3. **AI Response**: Generating comprehensive diagnosis and recommendations
            """
        },
        {
            "name": "System",
            "description": """
System monitoring and health check endpoints.

Use these endpoints to monitor service availability and performance.
            """
        }
    ],
    lifespan=lifespan
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models matching swagger schemas
class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=MAX_QUERY_LENGTH,
        description="Medical query in Hebrew",
        examples=["יש לי כאב ראש"]
    )

    # Pydantic v2 style configuration (replaces deprecated inner Config class)
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
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    retrieved_conditions: Optional[int] = Field(None, description="Number of medical conditions retrieved from knowledge base")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score of the response (0-1)")
    debug: Optional[dict] = Field(None, description="Debug information about embedding and indexing flow")

class DiagnosisResponse(BaseModel):
    question: str = Field(..., description="The original query submitted by the user")
    answer: str = Field(..., description="AI-generated medical diagnosis and recommendations in Hebrew")
    metadata: Optional[ResponseMetadata] = Field(None, description="Additional metadata about the response")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", examples=["healthy"])
    timestamp: str = Field(..., description="Current timestamp in ISO format")
    version: str = Field(..., description="API version", examples=["1.0.0"])

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing what went wrong")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    timestamp: Optional[str] = Field(None, description="When the error occurred")

# API Endpoints

@app.get(
    "/",
    summary="Service Root",
    description="Basic service info and helpful links.",
    tags=["System"],
    responses={
        200: {
            "description": "Service root information"
        }
    }
)
async def root():
    return {
        "name": "RAG Medical Diagnosis API",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "health": "/health",
        "diagnose": "/diagnose",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service is running and healthy",
    tags=["System"],
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2025-08-08T15:30:00Z",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Health check endpoint to verify service availability.
    
    Returns the current status, timestamp, and version of the API.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        version=API_VERSION
    )

@app.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    summary="Medical Diagnosis Query",
    description="""
Submit a medical query in Hebrew and receive AI-powered diagnosis assistance.

The system will:
1. Convert your query to a vector representation
2. Search the medical knowledge base for similar conditions
3. Generate a comprehensive response using AI

**Example queries:**
- "יש לי כאב ראש" (I have a headache)
- "חום וכאב גרון" (Fever and sore throat)
- "כאב בטן" (Stomach pain)
    """,
    tags=["Medical Diagnosis"],
    responses={
        200: {
            "description": "Successful diagnosis response",
            "content": {
                "application/json": {
                    "examples": {
                        "headache_response": {
                            "summary": "Headache Response",
                            "description": "AI response for headache query",
                            "value": {
                                "question": "יש לי כאב ראש",
                                "answer": "בהתבסס על המידע שסופק, כאב הראש שלך יכול להיות קשור למספר אפשרויות...",
                                "metadata": {
                                    "response_time_ms": 1250,
                                    "retrieved_conditions": 3,
                                    "confidence_score": 0.85
                                }
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_question": {
                            "summary": "Missing Question Field",
                            "value": {
                                "detail": [
                                    {
                                        "type": "missing",
                                        "loc": ["body", "question"],
                                        "msg": "Field required",
                                        "input": {}
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {
                        "llm_error": {
                            "summary": "LLM API Error",
                            "value": {
                                "detail": "Error calling Groq API: Rate limit exceeded",
                                "error_code": "LLM_API_ERROR",
                                "timestamp": "2025-08-08T15:30:00Z"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def diagnose(request: QueryRequest, debug: bool = Query(False, description="Return embedding/indexing debug info")):
    """
    Process a medical diagnosis query in Hebrew.
    
    This endpoint accepts a medical question in Hebrew and returns
    an AI-powered diagnosis with recommendations based on the 
    medical knowledge base.
    """
    start_time = time.time()
    logger.info(f"Processing query: {request.question[:50]}...")
    
    try:
        # Get answer (optionally with debug info)
        if debug:
            answer, debug_info = get_answer_with_debug(request.question)
        else:
            answer = get_answer(request.question)
            debug_info = None
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Query processed successfully in {response_time_ms}ms")
        
        # Create response with metadata
        response = DiagnosisResponse(
            question=request.question,
            answer=answer,
            metadata=ResponseMetadata(
                response_time_ms=response_time_ms,
                retrieved_conditions=3,  # Fixed for now, could be dynamic
                confidence_score=0.85,   # Fixed for now, could be calculated
                debug=debug_info
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Create detailed error response
        error_response = ErrorResponse(
            detail=f"Internal server error: {str(e)}",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now(UTC).isoformat()
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )

# Add custom exception handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=exc.detail if hasattr(exc, 'detail') else {"detail": "Validation error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)