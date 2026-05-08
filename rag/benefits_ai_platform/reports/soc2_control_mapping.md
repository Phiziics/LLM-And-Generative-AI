# SOC 2-Style Control Mapping

## Purpose

This document maps BenefitsAI project features to SOC 2-style security, availability, confidentiality, and processing integrity practices.

## Control Areas

### 1. Security

Implemented or planned controls:

1. Audit logs for AI and ML system activity
2. Query hashing to avoid storing raw sensitive inputs
3. Redaction of emails, phone numbers, SSNs, and member identifiers
4. Planned role-based access control
5. Planned secure API authentication

### 2. Confidentiality

Implemented or planned controls:

1. Public and synthetic datasets only
2. No real protected health information used
3. Redacted user queries stored in audit logs
4. Source document tracking for RAG responses
5. Planned secrets management for API keys

### 3. Processing Integrity

Implemented or planned controls:

1. Model card documenting model purpose, target, features, and limitations
2. Target leakage prevention by excluding payment-derived features
3. Saved model evaluation results
4. Saved prediction outputs for reproducibility
5. Human review flag for high-risk predictions

### 4. Availability

Implemented or planned controls:

1. Saved model artifact for repeatable inference
2. Saved FAISS vector index for retrieval
3. Planned FastAPI service
4. Planned Streamlit dashboard
5. Planned Docker deployment

## Evidence Files

1. reports/model_card.md
2. reports/claim_model_results.csv
3. data/04_predictions/claim_risk_predictions.csv
4. data/04_predictions/audit_logs/ai_audit_log.csv
5. data/04_predictions/audit_logs/ai_audit_log.json
6. reports/governance_metrics.csv
7. vector_db/cms_plan_rag.index
8. vector_db/cms_plan_chunks_metadata.pkl