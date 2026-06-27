# Import system tools for path handling
import sys
from pathlib import Path

# Import FastAPI tools
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Pydantic for request body validation
from pydantic import BaseModel, Field

# Import optional typing helpers
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import RiskRadar project modules
from src.riskradar_retriever import RiskRadarRetriever
from src.riskradar_generator import RiskRadarGenerator
from src.riskradar_financials import RiskRadarFinancials


# Create FastAPI app
app = FastAPI(
    title="RiskRadar AI API",
    description="Local API for SEC filing retrieval, RAG answers, and structured financial context.",
    version="0.1.0"
)

# Add CORS middleware
# This allows a future Streamlit or frontend app to call the API locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    """
    Request body for evidence retrieval.
    """

    # User question
    question: str = Field(..., description="User question")

    # Optional ticker filter
    ticker: Optional[str] = Field(None, description="Optional ticker filter such as NVDA")

    # Optional SEC section filter
    section_name: Optional[str] = Field(
        None,
        description="Optional section filter such as item_1a_risk_factors"
    )

    # Number of final evidence chunks to return
    top_k: int = Field(3, description="Number of final evidence chunks")

    # Number of retrieval candidates before reranking
    candidate_k: int = Field(15, description="Number of retrieval candidates")

    # Whether to use the cross-encoder reranker
    use_reranker: bool = Field(True, description="Whether to use reranking")


class AnswerRequest(BaseModel):
    """
    Request body for grounded RAG answer generation.
    """

    # User question
    question: str = Field(..., description="User question")

    # Optional ticker filter
    ticker: Optional[str] = Field(None, description="Optional ticker filter such as NVDA")

    # Optional SEC section filter
    section_name: Optional[str] = Field(
        None,
        description="Optional section filter such as item_1a_risk_factors"
    )

    # Number of final evidence chunks
    top_k: int = Field(3, description="Number of final evidence chunks")

    # Number of retrieval candidates before reranking
    candidate_k: int = Field(15, description="Number of retrieval candidates")

    # Whether to use reranker
    use_reranker: bool = Field(True, description="Whether to use reranking")

    # Whether to include structured financial context from notebook 10
    include_financial_context: bool = Field(
        False,
        description="Whether to include structured financial context"
    )

    # Ollama model name
    ollama_model: str = Field(
        "llama3.2:3b",
        description="Local Ollama model name"
    )


# Create global objects
# These are loaded once when the API starts
retriever = None
financials = None


@app.on_event("startup")
def startup_event():
    """
    Load project assets once when the API starts.

    This avoids reloading embeddings, Chroma, and financial files on every request.
    """

    # Use global variables
    global retriever
    global financials

    # Load retrieval engine
    retriever = RiskRadarRetriever(
        project_root=PROJECT_ROOT,
        load_reranker=True
    )

    # Load structured financial layer
    financials = RiskRadarFinancials(
        project_root=PROJECT_ROOT
    )


@app.get("/")
def root():
    """
    Root endpoint.
    """

    # Return simple API message
    return {
        "project": "RiskRadar AI",
        "message": "API is running. Go to /docs to test endpoints."
    }


@app.get("/health")
def health_check():
    """
    Check API health and loaded system components.
    """

    # Return health status
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "retriever_loaded": retriever is not None,
        "financials_loaded": financials is not None,
        "retriever_health": retriever.health_check() if retriever else None,
        "financials_health": financials.health_check() if financials else None
    }


@app.get("/companies")
def get_companies():
    """
    Return companies available in the RAG system.
    """

    # Return company list
    return {
        "companies": retriever.get_available_companies()
    }


@app.get("/sections")
def get_sections():
    """
    Return SEC sections available in the RAG system.
    """

    # Return section list
    return {
        "sections": retriever.get_available_sections()
    }


@app.get("/financials/{ticker}")
def get_financials(ticker: str):
    """
    Return structured financial data for one ticker.
    """

    # Standardize ticker
    ticker = ticker.upper().strip()

    # Get financial package
    financial_package = financials.get_financial_package(ticker)

    # Return financial package
    return financial_package


@app.post("/retrieve")
def retrieve_evidence(request: RetrieveRequest):
    """
    Retrieve SEC filing evidence for a question.
    """

    # Standardize ticker if provided
    ticker = request.ticker.upper().strip() if request.ticker else None

    # Retrieve evidence records
    evidence = retriever.retrieve_evidence_records(
        query=request.question,
        ticker=ticker,
        section_name=request.section_name,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        use_reranker=request.use_reranker
    )

    # Return retrieval response
    return {
        "question": request.question,
        "ticker": ticker,
        "section_name": request.section_name,
        "top_k": request.top_k,
        "candidate_k": request.candidate_k,
        "use_reranker": request.use_reranker,
        "evidence_count": len(evidence),
        "evidence": evidence
    }


@app.post("/answer")
def answer_question(request: AnswerRequest):
    """
    Retrieve evidence and generate a grounded RAG answer.
    """

    # Standardize ticker if provided
    ticker = request.ticker.upper().strip() if request.ticker else None

    # Retrieve SEC evidence
    evidence = retriever.retrieve_evidence_records(
        query=request.question,
        ticker=ticker,
        section_name=request.section_name,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        use_reranker=request.use_reranker
    )

    # Default financial context to None
    financial_context = None

    # Add financial context only when requested and ticker is provided
    if request.include_financial_context and ticker is not None:
        financial_context = financials.get_financial_context(ticker)

    # Create generator with requested Ollama model
    generator = RiskRadarGenerator(
        ollama_model=request.ollama_model
    )

    # Generate grounded answer
    answer_package = generator.answer_question(
        question=request.question,
        evidence_records=evidence,
        financial_context=financial_context
    )

    # Return answer response
    return {
        "question": request.question,
        "ticker": ticker,
        "section_name": request.section_name,
        "top_k": request.top_k,
        "candidate_k": request.candidate_k,
        "use_reranker": request.use_reranker,
        "include_financial_context": request.include_financial_context,
        "ollama_model": request.ollama_model,
        "answer": answer_package["answer"],
        "sources": answer_package["sources"],
        "financial_context": answer_package["financial_context"],
        "financial_context_used": answer_package["financial_context_used"],
        "generation_time_seconds": answer_package["generation_time_seconds"],
        "evidence_count": len(evidence)
    }


@app.post("/answer-with-financials")
def answer_with_financials(request: AnswerRequest):
    """
    Convenience endpoint.

    This always includes structured financial context when ticker is provided.
    """

    # Force financial context to be included
    request.include_financial_context = True

    # Reuse the normal answer endpoint logic
    return answer_question(request)