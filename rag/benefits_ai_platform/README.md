# BenefitsAI: Secure Claims, Eligibility, and Benefits Intelligence Platform

## Introduction

BenefitsAI is an AI/ML project designed for health insurance and benefits administration workflows. The project combines real public healthcare and insurance data with machine learning, retrieval-augmented generation, and governance controls to support benefits document search, claims intelligence, eligibility analysis, and audit-ready AI usage.

This project is built to demonstrate production-style AI/ML engineering for regulated benefits operations, including claims analytics, high-cost claim prediction, benefits plan retrieval, source-backed question answering, model documentation, and responsible AI governance.

## Problem

Benefits administrators, third-party administrators, employers, unions, and trust funds manage large volumes of plan documents, eligibility rules, claims records, and member communications.

These workflows are repetitive, time-consuming, and sensitive because they involve healthcare, insurance, and benefits data.

The goal of BenefitsAI is to help internal teams:

1. Search and understand benefits plan information
2. Answer plan and eligibility questions with source-backed retrieval
3. Analyze claims and identify higher-risk cases
4. Predict high-cost claims using machine learning
5. Maintain audit-ready logs for AI and ML system activity
6. Demonstrate privacy, security, and governance awareness

## Project Goals

1. Use real public healthcare and insurance datasets
2. Clean and prepare CMS Exchange plan data
3. Clean and prepare CMS synthetic Medicare claims data
4. Build a high-cost claims risk model
5. Build a benefits document RAG assistant
6. Add audit logging and governance controls
7. Prepare the system for FastAPI and Streamlit deployment
8. Document the project like a production AI/ML system

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