# DenialOps AI

## Insurance Claim Denial Prevention and Appeal Assistant

DenialOps AI is a healthcare analytics and GenAI project designed to identify insurance claim denial risk, explain likely denial reasons, and support appeal workflows using public CMS datasets and payer coverage policy documents.

## Problem

Hospitals and clinics lose time and revenue when insurance claims are denied because of missing documentation, coding issues, medical necessity problems, eligibility issues, or payer-policy requirements.

This project explores how data science and AI can help billing teams detect denial risk before submission and generate practical next-step recommendations.

## Project Goals

- Analyze Medicare provider service and payment data
- Identify high-risk billing patterns by provider, specialty, HCPCS code, place of service, charge amount, and payment amount
- Build a denial-risk scoring model
- Create a payer-policy RAG assistant using CMS coverage documents
- Generate appeal letter drafts and documentation checklists
- Build a dashboard for denial and reimbursement insights

## Data Sources

### CMS Medicare Physician & Other Practitioners by Provider and Service

Direct CSV:

```text
https://data.cms.gov/sites/default/files/2026-05/b5ebab5a-f490-418a-9bce-4b9f31419356/PHY_R26_P05_V10_D24_Prov_Svc.csv

Data Dictionary:

https://data.cms.gov/resources/medicare-physician-other-practitioners-by-provider-and-service-data-dictionary
CMS Medicare Coverage Database Downloads

Used for coverage policies, LCDs, NCDs, documentation rules, and medical necessity lookup.

https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx
CMS Review Reason Codes and Statements

Used for denial reason categories and appeal workflow logic.

https://www.cms.gov/data-research/monitoring-programs/medicare-fee-service-compliance-programs/review-reason-codes-and-statements
Planned Features
Data loading and cleaning
Exploratory data analysis
Feature engineering
Synthetic denial-risk labeling
Machine learning model
Explainability with feature importance / SHAP
Streamlit dashboard
FastAPI prediction endpoint
RAG assistant for CMS coverage policies
Appeal letter generator
Tech Stack
Python
pandas
scikit-learn
Streamlit
FastAPI
Chroma or FAISS
LangChain or LlamaIndex
Docker
Project Structure
DenialOps-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│
├── notebooks/
│   └── 01_data_load_and_eda.ipynb
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   └── preprocess.py
│
├── .gitignore
├── README.md
└── requirements.txt
Engineering Considerations
Architecture

The project separates data ingestion, preprocessing, modeling, API serving, and dashboard layers.

Scalability

Large CMS files should be processed in chunks or converted to Parquet for faster loading.

Security

Raw healthcare-related files are not committed to GitHub. No PHI should be used in this repository.

Observability

Future versions can log model predictions, confidence scores, input drift, and error rates.

Maintainability

Code is separated into reusable modules inside src/.

Reliability

The app should validate input columns before running predictions and return clear errors for missing fields.

Status

Project setup in progress.


## 7. Initialize Git

```powershell
git init
git add .
git commit -m "Initial project structure for DenialOps AI"