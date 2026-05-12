Copy and paste this full version into `README.md`.

````markdown
# BenefitsAI: Secure Claims, Eligibility, and Benefits Intelligence Platform

## Introduction

BenefitsAI is an AI/ML project designed for health insurance and benefits administration workflows. The project combines real public healthcare and insurance data with machine learning, retrieval-augmented generation, API development, dashboarding, and governance controls.

The system supports:

1. Benefits plan retrieval
2. Source-backed benefits question answering
3. High-cost claim risk prediction
4. Audit logging
5. PII/PHI-style redaction
6. Governance documentation
7. FastAPI and Streamlit deployment

This project is built to demonstrate production-style AI/ML engineering for regulated benefits operations.

---

## Problem

Benefits administrators, third-party administrators, employers, unions, and trust funds manage large volumes of plan documents, eligibility rules, claims records, and member communications.

These workflows are repetitive, time-consuming, and sensitive because they involve healthcare, insurance, and benefits data.

BenefitsAI helps internal teams:

1. Search and understand benefits plan information
2. Answer plan and eligibility questions with source-backed retrieval
3. Analyze claims and identify higher-risk cases
4. Predict high-cost claims using machine learning
5. Maintain audit-ready logs for AI and ML system activity
6. Demonstrate privacy, security, and governance awareness

---

## Project Goals

1. Use real public healthcare and insurance datasets
2. Clean and prepare CMS Exchange plan data
3. Clean and prepare CMS synthetic Medicare claims data
4. Build a high-cost claims risk model
5. Build a benefits document RAG retrieval system
6. Add PII/PHI-style redaction
7. Add audit logging
8. Create governance documentation
9. Build a FastAPI backend
10. Build a Streamlit dashboard
11. Add tests
12. Add Docker support
13. Document the project like a production AI/ML system

---

## Data Sources

This project uses public and privacy-safe healthcare and insurance datasets.

### 1. CMS Exchange Public Use Files

Used for:

1. Benefits plan analysis
2. Plan comparison
3. HSA and plan design fields
4. Out-of-pocket analysis
5. Generating plan summary documents for RAG
6. Benefits and coverage question answering

Raw files:

```text
data/01_raw/cms_exchange_pufs/
    plan_attributes_puf_2026.zip
    benefits_cost_sharing_puf_2026.zip
    rate_puf_2026.zip
````

Processed outputs:

```text
data/02_preprocessed/clean_plan_data.csv
data/02_preprocessed/rag_ready_plan_data_full.csv
data/02_preprocessed/rag_ready_plan_data_sample.csv
```

### 2. CMS DE-SynPUF Synthetic Medicare Claims Data

Used for:

1. Claims exploratory data analysis
2. Claim cost analysis
3. High-cost claim classification
4. Member utilization analysis
5. Claims intelligence modeling

Raw files:

```text
data/01_raw/cms_claims/
    DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
    DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv
    DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv
    DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
    DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv
```

Processed outputs:

```text
data/02_preprocessed/clean_claims_data.csv
data/03_features/claim_features.csv
data/03_features/member_claim_features.csv
```

---

## Project Architecture

```text
Real Public Healthcare / Insurance Data
    ↓
Data Cleaning and Feature Engineering
    ↓
Claims Risk Model + Benefits RAG System
    ↓
Reusable src/ Python Modules
    ↓
FastAPI Backend
    ↓
Streamlit Dashboard
    ↓
Audit Logging and Governance Reports
    ↓
Dockerized Deployment
```

---

## Repository Structure

```text
benefits_ai_platform/
    data/
        01_raw/
            cms_exchange_pufs/
            cms_claims/
            medicaid_eligibility/

        02_preprocessed/
            clean_plan_data.csv
            rag_ready_plan_data_full.csv
            rag_ready_plan_data_sample.csv
            clean_claims_data.csv
            rag_plan_chunks.csv
            rag_documents/
            sample_rag_retrieval_results.csv

        03_features/
            claim_features.csv
            member_claim_features.csv

        04_predictions/
            claim_risk_predictions.csv
            audit_logs/
                ai_audit_log.csv
                ai_audit_log.json
                ai_audit_log.jsonl

    notebooks/
        01_plan_data_eda.ipynb
        02_claims_eda.ipynb
        03_claim_risk_modeling.ipynb
        04_benefits_rag_prototype.ipynb
        05_ai_governance_audit_logs.ipynb

    src/
        data/
            load_claims.py
            load_plans.py
            clean_claims.py
            clean_plans.py

        features/
            claims_features.py
            plan_features.py

        models/
            train_claim_risk_model.py
            predict_claim_risk.py
            evaluate.py

        rag/
            build_plan_documents.py
            chunk_documents.py
            build_vector_store.py
            retrieve.py
            generate_answer.py

        security/
            redact_pii.py
            audit_logger.py
            access_control.py

        api/
            main.py
            schemas.py

        app/
            streamlit_app.py

    vector_db/
        cms_plan_rag.index
        cms_plan_chunks_metadata.pkl

    models/
        claim_risk_model.pkl

    reports/
        claim_model_results.csv
        model_card.md
        risk_assessment.md
        soc2_control_mapping.md
        hipaa_privacy_notes.md
        governance_metrics.csv

    tests/
        test_redaction.py
        test_claim_predictor.py
        test_rag_retriever.py

    Dockerfile
    docker-compose.yml
    .dockerignore
    requirements.txt
    README.md
    .gitignore
```

---

## Notebook Workflow

### 01_plan_data_eda.ipynb

Goal: Load and clean real CMS Exchange plan data.

Tasks completed:

1. Loaded CMS Exchange Plan Attributes PUF
2. Inspected columns, missing values, and data types
3. Cleaned plan ID, issuer, state, metal level, plan type, HSA, and out-of-pocket fields
4. Handled missing values carefully
5. Created a RAG-ready text field for each plan
6. Saved clean plan datasets

Outputs:

```text
data/02_preprocessed/clean_plan_data.csv
data/02_preprocessed/rag_ready_plan_data_full.csv
data/02_preprocessed/rag_ready_plan_data_sample.csv
```

---

### 02_claims_eda.ipynb

Goal: Load and explore CMS synthetic Medicare claims data.

Tasks completed:

1. Loaded beneficiary, inpatient, and outpatient claims data
2. Inspected columns, missing values, and data types
3. Standardized claim-level fields
4. Cleaned member IDs, claim IDs, dates, provider IDs, diagnosis codes, and payment amounts
5. Combined inpatient and outpatient claims
6. Created claim duration fields
7. Created a high-cost claim target
8. Built member-level utilization features
9. Built claim-level modeling features
10. Saved clean claims and feature datasets

Final claims summary:

```text
Clean claim records: 857,563
Unique members: 86,738
Claim types:
    outpatient: 790,790
    inpatient: 66,773
High-cost threshold: $2,100
```

Outputs:

```text
data/02_preprocessed/clean_claims_data.csv
data/03_features/claim_features.csv
data/03_features/member_claim_features.csv
```

---

### 03_claim_risk_modeling.ipynb

Goal: Train a machine learning model to predict high-cost claims.

Tasks completed:

1. Loaded claim feature data
2. Defined target variable: high_cost_claim
3. Split train and test data
4. Built preprocessing pipeline
5. Trained Logistic Regression baseline
6. Trained Random Forest model
7. Compared model performance
8. Checked for overfitting
9. Avoided target leakage by excluding payment-derived fields from model features
10. Saved predictions, model results, model artifact, and model card

Best model:

```text
Random Forest
```

Test performance:

```text
Accuracy: 0.94
High-cost claim precision: 0.66
High-cost claim recall: 0.81
High-cost claim F1-score: 0.72
Test ROC-AUC: 0.9056
```

Business interpretation:

The model is designed as a claims triage tool. It prioritizes claims that are more likely to be high-cost so internal teams can review them earlier. It should support review workflows, not automate claim decisions.

Outputs:

```text
reports/claim_model_results.csv
data/04_predictions/claim_risk_predictions.csv
models/claim_risk_model.pkl
reports/model_card.md
```

---

### 04_benefits_rag_prototype.ipynb

Goal: Build a retrieval system over real CMS plan data.

Tasks completed:

1. Loaded RAG-ready CMS plan data
2. Converted plan records into text documents
3. Chunked documents
4. Created embeddings with Sentence Transformers
5. Stored embeddings in a FAISS vector database
6. Retrieved relevant chunks for user questions
7. Returned source-backed plan information
8. Saved retrieval results for reporting

Outputs:

```text
data/02_preprocessed/rag_documents/
data/02_preprocessed/rag_plan_chunks.csv
vector_db/cms_plan_rag.index
vector_db/cms_plan_chunks_metadata.pkl
data/02_preprocessed/sample_rag_retrieval_results.csv
```

Example questions:

1. What is the out-of-pocket maximum for this plan?
2. Is this plan HSA eligible?
3. What metal level is this plan?
4. Which issuer offers this plan?
5. What state is this plan available in?
6. Are medical and drug deductibles integrated?

---

### 05_ai_governance_audit_logs.ipynb

Goal: Demonstrate AI governance, privacy, and compliance readiness.

Tasks completed:

1. Created sample audit logs
2. Tracked user query, retrieved sources, model version, response time, and confidence score
3. Added redaction status
4. Added human approval status
5. Created governance metrics
6. Created SOC 2-style control mapping
7. Created HIPAA privacy notes
8. Created project risk assessment

Outputs:

```text
data/04_predictions/audit_logs/ai_audit_log.csv
data/04_predictions/audit_logs/ai_audit_log.json
reports/governance_metrics.csv
reports/soc2_control_mapping.md
reports/hipaa_privacy_notes.md
reports/risk_assessment.md
```

---

## Key Features

### Claims Risk Model

The claims ML model predicts whether a claim is likely to be high-cost.

Current model features:

1. claim_type
2. claim_duration_days
3. has_provider_id
4. has_diagnosis_code

Important modeling decision:

Payment amount fields were excluded from training because the high-cost target was created from payment amount. This avoids target leakage and makes the model evaluation more realistic.

---

### Benefits RAG Assistant

The RAG assistant retrieves relevant CMS plan records and returns source-backed plan information.

Core features:

1. Plan document creation
2. Text chunking
3. Embeddings with Sentence Transformers
4. FAISS vector search
5. Source-backed retrieval
6. Retrieval-only baseline before LLM generation

---

### Security and Governance Layer

The project includes a governance layer to show responsible AI practices.

Implemented or planned controls:

1. PII/PHI-style redaction
2. Secure prompt handling
3. Audit logs
4. Model version tracking
5. Source document tracking
6. Human approval workflow
7. SOC 2-style control mapping
8. HIPAA privacy notes
9. Risk assessment

---

## Tech Stack

1. Python
2. pandas
3. NumPy
4. scikit-learn
5. Random Forest
6. XGBoost
7. FastAPI
8. Streamlit
9. FAISS
10. Sentence Transformers
11. Gemini API or Vertex AI Gemini
12. MLflow
13. Docker
14. BigQuery or PostgreSQL
15. GitHub Actions
16. pytest

---

## Environment Setup

### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To activate the environment later on Windows:

```bash
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To activate the environment later on macOS / Linux:

```bash
source .venv/bin/activate
```

---

## How to Run the Project

### 1. Run notebooks in order

Run the notebooks in this order:

```text
01_plan_data_eda.ipynb
02_claims_eda.ipynb
03_claim_risk_modeling.ipynb
04_benefits_rag_prototype.ipynb
05_ai_governance_audit_logs.ipynb
```

These notebooks create the required model, vector database, processed data, audit logs, and reports.

---

### 2. Required generated files

Before running the API or Streamlit app, these files should exist:

```text
models/claim_risk_model.pkl
vector_db/cms_plan_rag.index
vector_db/cms_plan_chunks_metadata.pkl
data/02_preprocessed/rag_ready_plan_data_sample.csv
data/02_preprocessed/sample_rag_retrieval_results.csv
reports/claim_model_results.csv
reports/model_card.md
```

---

### 3. Run tests

```bash
pytest tests -v
```

If imports fail on Windows PowerShell:

```bash
$env:PYTHONPATH = "."
pytest tests -v
```

Expected result:

```text
All tests passed
```

---

### 4. Run FastAPI backend

```bash
uvicorn src.api.main:app --reload
```

If imports fail on Windows PowerShell:

```bash
$env:PYTHONPATH = "."
uvicorn src.api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```text
GET  /
GET  /health
POST /rag/query
POST /claims/predict
```

Example RAG request:

```json
{
  "query": "What is the out-of-pocket maximum for this plan?",
  "top_k": 3
}
```

Example claim prediction request:

```json
{
  "claim_type": "inpatient",
  "claim_duration_days": 5,
  "has_provider_id": 1,
  "has_diagnosis_code": 1
}
```

---

### 5. Run Streamlit dashboard

```bash
streamlit run src/app/streamlit_app.py
```

If imports fail on Windows PowerShell:

```bash
$env:PYTHONPATH = "."
streamlit run src/app/streamlit_app.py
```

Open the dashboard:

```text
http://localhost:8501
```

Dashboard tabs:

```text
Benefits RAG Search
Claim Risk Prediction
Audit Logs
```

---

### 6. Run with Docker

Build containers:

```bash
docker compose build
```

Run services:

```bash
docker compose up
```

Open:

```text
FastAPI:   http://127.0.0.1:8000/docs
Streamlit: http://127.0.0.1:8501
```

Stop services:

```bash
docker compose down
```

---

## API Endpoints

### Health Check

```text
GET /health
```

Returns service status and confirms the RAG index and claim model are loaded.

---

### Benefits RAG Query

```text
POST /rag/query
```

Retrieves relevant CMS plan chunks using FAISS vector search.

Request body:

```json
{
  "query": "Is this plan HSA eligible?",
  "top_k": 5
}
```

Response includes:

```text
query
redacted_query
answer
retrieved source chunks
audit_id
```

---

### Claim Risk Prediction

```text
POST /claims/predict
```

Predicts whether a claim is likely to be high-cost.

Request body:

```json
{
  "claim_type": "outpatient",
  "claim_duration_days": 2,
  "has_provider_id": 1,
  "has_diagnosis_code": 1
}
```

Response includes:

```text
predicted_high_cost_claim
high_cost_claim_probability
human_review_required
audit_id
```

---

## Model Performance

The best claim risk model is a Random Forest classifier.

Test results:

```text
Accuracy: 0.94
High-cost claim precision: 0.66
High-cost claim recall: 0.81
High-cost claim F1-score: 0.72
Test ROC-AUC: 0.9056
```

The model is intended for triage and prioritization, not automatic claim decisions.

---

## Governance Outputs

The project creates governance and compliance-oriented files:

```text
reports/model_card.md
reports/risk_assessment.md
reports/soc2_control_mapping.md
reports/hipaa_privacy_notes.md
reports/governance_metrics.csv
data/04_predictions/audit_logs/ai_audit_log.csv
data/04_predictions/audit_logs/ai_audit_log.json
data/04_predictions/audit_logs/ai_audit_log.jsonl
```

These files demonstrate:

1. Audit logging
2. Redaction checks
3. Model version tracking
4. Source document tracking
5. Human review flags
6. SOC 2-style control mapping
7. HIPAA privacy awareness

---

## Current Progress

Completed:

1. Project setup
2. Notebook 1: Plan Data EDA
3. Notebook 2: Claims EDA
4. Notebook 3: Claim Risk Modeling
5. Notebook 4: Benefits RAG Prototype
6. Notebook 5: AI Governance and Audit Logs
7. RAG retrieval module
8. Claim risk prediction module
9. Redaction module
10. Audit logger module
11. FastAPI backend
12. Streamlit dashboard
13. pytest tests
14. Docker files

Next:

1. Final test pass
2. GitHub cleanup
3. Add screenshots
4. Add resume bullets
5. Add LinkedIn project description
6. Optional Gemini/Vertex AI answer generation layer

---

## Success Criteria

The project is successful if it can:

1. Use real public healthcare and insurance data
2. Clean and prepare real CMS plan data
3. Clean and prepare synthetic Medicare claims data
4. Train a claims risk prediction model
5. Answer benefits questions with source-backed retrieval
6. Log AI usage for auditability
7. Demonstrate privacy and security awareness
8. Run through FastAPI
9. Run through Streamlit
10. Pass tests
11. Be shown in interviews as a production-style AI/ML system

---

## Business Value

BenefitsAI can reduce administrative burden by helping internal teams answer repetitive plan and eligibility questions, identify higher-risk claims, and maintain audit-ready logs of AI system activity.

This type of system could support:

1. Third Party Administrators
2. Benefits brokers
3. Employer benefits teams
4. Union trust funds
5. Healthcare operations teams
6. Insurance support teams

---

## Future Improvements

1. Add Gemini or Vertex AI answer generation
2. Deploy API to Google Cloud Run
3. Store plan and claims data in BigQuery
4. Add Cloud Functions for lightweight workflows
5. Add Dataflow or Cloud Composer for scheduled ingestion
6. Add role-based access control
7. Add monitoring for model drift
8. Add RAG evaluation metrics
9. Add automated retraining pipeline
10. Add Kubernetes deployment option
11. Add CI/CD with GitHub Actions
12. Add MLflow experiment tracking
13. Add BigQuery warehouse layer
14. Add stronger PHI detection with Microsoft Presidio or similar tooling

---

## Disclaimer

This project uses public and synthetic data for educational and portfolio purposes. It does not use real protected health information.

The project is designed to demonstrate AI/ML engineering, security awareness, and responsible AI practices for healthcare and insurance workflows. It should not be used for real claim approval, denial, eligibility determination, or medical decision-making without proper compliance review, validation, access controls, and human oversight.

```
```