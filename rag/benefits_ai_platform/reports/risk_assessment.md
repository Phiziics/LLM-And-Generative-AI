# BenefitsAI Risk Assessment

## Purpose

This document identifies key risks in the BenefitsAI system and describes mitigation strategies.

## Key Risks

### 1. Incorrect RAG Retrieval

Risk:
The retrieval system may return irrelevant plan records.

Mitigation:
Use source-backed retrieval, confidence scores, and human review for sensitive use cases.

### 2. Hallucinated LLM Answers

Risk:
If an LLM is added, it may generate unsupported answers.

Mitigation:
Use retrieved context only, cite sources, and instruct the model to say when information is unavailable.

### 3. Target Leakage

Risk:
The model could use features that directly define the target.

Mitigation:
Payment amount fields were excluded from the claim risk model because the target is based on payment amount.

### 4. Overfitting

Risk:
The model may perform well on training data but poorly on new data.

Mitigation:
Train/test split, test metrics, overfitting gap checks, and controlled Random Forest depth.

### 5. Privacy Exposure

Risk:
User queries could contain sensitive identifiers.

Mitigation:
Redaction functions, query hashing, and audit logs that avoid storing raw sensitive text.

### 6. Automation Bias

Risk:
Users may overtrust model predictions.

Mitigation:
Use the model for triage and prioritization only. Human review remains required for high-risk cases.

## Overall Risk Rating

Medium.

The project uses public and synthetic data and includes basic governance controls, but production deployment would require stronger security, access control, monitoring, and compliance review.