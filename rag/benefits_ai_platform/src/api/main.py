from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.retrieve import BenefitsRetriever
from src.models.predict_claim_risk import ClaimRiskPredictor
from src.security.audit_logger import AuditLogger
from src.security.redact_pii import redact_sensitive_text


PROJECT_ROOT = Path.cwd()

app = FastAPI(
    title="BenefitsAI API",
    description="Secure AI/ML API for benefits retrieval and claims risk prediction.",
    version="0.1.0",
)

retriever = BenefitsRetriever(project_root=PROJECT_ROOT)
claim_predictor = ClaimRiskPredictor(project_root=PROJECT_ROOT)
audit_logger = AuditLogger(project_root=PROJECT_ROOT)


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="User benefits or plan question")
    top_k: int = Field(default=5, ge=1, le=10)


class ClaimRiskRequest(BaseModel):
    claim_type: Literal["inpatient", "outpatient"]
    claim_duration_days: float = Field(..., ge=1)
    has_provider_id: int = Field(..., ge=0, le=1)
    has_diagnosis_code: int = Field(..., ge=0, le=1)


@app.get("/")
def root():
    return {
        "app": "BenefitsAI API",
        "status": "running",
        "endpoints": [
            "/health",
            "/rag/query",
            "/claims/predict",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_index_loaded": True,
        "claim_model_loaded": True,
    }


@app.post("/rag/query")
def query_benefits(request: RAGQueryRequest):
    redacted_query = redact_sensitive_text(request.query)

    results = retriever.retrieve(
        query=redacted_query,
        top_k=request.top_k,
    )

    source_documents = [result["document_name"] for result in results]
    avg_score = sum(result["score"] for result in results) / len(results)

    audit_record = audit_logger.log_event(
        user_query=request.query,
        task_type="benefits_rag_retrieval",
        model_or_system="sentence-transformers/all-MiniLM-L6-v2 + FAISS",
        response_summary="Retrieved CMS plan chunks for benefits question.",
        source_documents=source_documents,
        confidence_score=avg_score,
        human_review_required=False,
    )

    return {
        "query": request.query,
        "redacted_query": redacted_query,
        "answer": "The most relevant CMS plan records are returned below. Review source chunks for exact plan details.",
        "results": results,
        "audit_id": audit_record["audit_id"],
    }


@app.post("/claims/predict")
def predict_claim_risk(request: ClaimRiskRequest):
    result = claim_predictor.predict_one(
        claim_type=request.claim_type,
        claim_duration_days=request.claim_duration_days,
        has_provider_id=request.has_provider_id,
        has_diagnosis_code=request.has_diagnosis_code,
    )

    audit_record = audit_logger.log_event(
        user_query=f"Predict claim risk for {request.claim_type} claim",
        task_type="claim_risk_prediction",
        model_or_system="Random Forest claim_risk_model.pkl",
        response_summary=(
            "Predicted high-cost claim probability: "
            f"{result['high_cost_claim_probability']:.4f}"
        ),
        source_documents=["models/claim_risk_model.pkl"],
        confidence_score=result["high_cost_claim_probability"],
        human_review_required=result["human_review_required"],
    )

    return {
        **result,
        "audit_id": audit_record["audit_id"],
    }