# BenefitsAI: Secure Claims, Eligibility, and Benefits Intelligence Platform

## Introduction

BenefitsAI is an AI/ML project designed for health insurance and benefits administration workflows. The project combines real public healthcare and insurance data with machine learning, retrieval-augmented generation, and governance controls to support benefits document search, claims intelligence, eligibility analysis, and audit-ready AI usage.

This project is built to match real-world AI/ML Solutions Engineer responsibilities, including Generative AI, RAG, ML pipelines, secure prompt handling, data redaction, audit logging, and responsible AI practices in regulated environments.

## Problem

Benefits administrators, third-party administrators, employers, unions, and trust funds manage large volumes of plan documents, eligibility rules, claims information, and member communications.

These workflows are repetitive, time-consuming, and sensitive because they involve healthcare, insurance, and benefits data.

The goal of this project is to build a secure AI platform that helps internal teams:

1. Search and understand benefits plan information
2. Answer eligibility and coverage questions with source-backed responses
3. Analyze claims and identify higher-risk cases
4. Predict high-cost claims using machine learning
5. Generate structured claim summaries
6. Track AI usage through audit logs
7. Demonstrate privacy, security, and governance awareness

## Project Goals

1. Use real public healthcare and insurance datasets
2. Clean and prepare CMS Exchange plan data
3. Clean and prepare CMS synthetic Medicare claims data
4. Build a claims risk prediction model
5. Build a benefits document RAG assistant
6. Add secure prompt handling and PII/PHI redaction
7. Add audit logging for governance and compliance
8. Prepare the system for API and dashboard deployment
9. Document the project like a production AI/ML system

## Data Sources

This project uses public and privacy-safe healthcare and insurance datasets.

### 1. CMS Exchange Public Use Files

Used for:

1. Benefits plan analysis
2. Plan comparison
3. Deductible and out-of-pocket analysis
4. HSA and plan design fields
5. Generating plan summary documents for RAG
6. Eligibility and coverage question answering

Raw files:

```text
data/01_raw/cms_exchange_pufs/
    plan_attributes_puf_2026.zip
    benefits_cost_sharing_puf_2026.zip
    rate_puf_2026.zip