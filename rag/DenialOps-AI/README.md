# DenialOps AI

## Insurance Reimbursement-Risk Scoring Dashboard

DenialOps AI is a healthcare revenue-cycle analytics project that scores CMS-style provider-service records for reimbursement-friction risk.

The project uses public CMS Medicare provider-service data, machine learning, model comparison, threshold tuning, and a Streamlit dashboard to identify records that may deserve billing review before submission.

GitHub project folder:

```text
https://github.com/Phiziics/LLM-And-Generative-AI/tree/main/rag/DenialOps-AI
```

---

## Important Disclaimer

This project is **not trained on true insurance claim denial outcomes**.

The public CMS dataset used in this project does not include actual denied-claim labels. Because of that, the target variable is an engineered **proxy reimbursement-risk label** created from reimbursement-friction signals.

This means the model should be understood as a:

```text
proxy reimbursement-risk scoring model
```

not a:

```text
real-world denied-claim prediction system
```

The purpose of this project is to demonstrate an end-to-end healthcare revenue-cycle AI workflow and create a review-support dashboard.

---

## Business Problem

Hospitals, clinics, and billing teams lose time and money when claims require extra review, correction, appeal, or resubmission.

Common reimbursement pain points include:

- submitted charges that differ heavily from allowed amounts
- low payment relative to submitted charges
- high service intensity
- payer-policy complexity
- manual review queues
- documentation burden
- appeal and resubmission workflows

DenialOps AI simulates how a billing team could prioritize records for review using structured reimbursement and provider-service data.

---

## Project Goal

The goal of this project is to build a practical AI workflow that can:

1. Load and clean public CMS provider-service data
2. Engineer reimbursement-risk features
3. Create a proxy reimbursement-risk target
4. Compare multiple machine learning models
5. Select a production demo model
6. Tune the decision threshold
7. Build a Streamlit dashboard for scoring uploaded records
8. Allow users to download scored results

---

## Data Source

The main dataset used is the CMS Medicare Physician & Other Practitioners by Provider and Service dataset.

Direct CMS CSV source:

```text
https://data.cms.gov/sites/default/files/2026-05/b5ebab5a-f490-418a-9bce-4b9f31419356/PHY_R26_P05_V10_D24_Prov_Svc.csv
```

CMS data dictionary:

```text
https://data.cms.gov/resources/medicare-physician-other-practitioners-by-provider-and-service-data-dictionary
```

The raw CMS file is not included in this repository because it is large.

---

## Dataset Overview

The original CMS dataset contains Medicare provider-service payment and utilization information.

Key raw fields used include:

```text
provider_state
provider_type
medicare_participating
hcpcs_code
is_drug
place_of_service
total_beneficiaries
total_services
total_beneficiary_day_services
avg_submitted_charge
avg_medicare_allowed_amount
avg_medicare_payment_amount
avg_medicare_standardized_amount
```

These fields support analysis of provider type, service volume, submitted charges, allowed amounts, payment amounts, and place of service.

---

## Proxy Label Creation

The CMS dataset does not contain true denial outcomes, so a proxy target was created.

The proxy target is:

```text
denial_risk_proxy
```

A proxy label is a stand-in target used when the true outcome is not available. In this project, the true outcome would be whether a claim was actually denied by an insurer. Since public CMS data does not include that outcome, the proxy label was engineered from reimbursement-friction signals.

The proxy label was created from three reimbursement-friction signals:

1. High submitted-charge to Medicare-allowed ratio
2. Low Medicare-payment to submitted-charge ratio
3. High services per beneficiary

The engineered features were:

```text
charge_to_allowed_ratio =
avg_submitted_charge / avg_medicare_allowed_amount

payment_to_charge_ratio =
avg_medicare_payment_amount / avg_submitted_charge

payment_to_allowed_ratio =
avg_medicare_payment_amount / avg_medicare_allowed_amount

services_per_beneficiary =
total_services / total_beneficiaries
```

Then three risk flags were created:

```text
high_charge_allowed_ratio = 1
if charge_to_allowed_ratio is in the top 25%

low_payment_charge_ratio = 1
if payment_to_charge_ratio is in the bottom 25%

high_service_intensity = 1
if services_per_beneficiary is in the top 25%
```

The final proxy label was created as:

```text
risk_score =
    high_charge_allowed_ratio
  + low_payment_charge_ratio
  + high_service_intensity

denial_risk_proxy =
    1 if risk_score >= 2
    0 otherwise
```

In plain English:

```text
A record is labeled high-risk if it has at least 2 out of 3 reimbursement-friction signals.
```

This label represents reimbursement-risk patterns, not confirmed claim denials.

---

## Modeling Approach

The project used three modeling stages.

### Notebook 01: Data Loading and EDA

This notebook:

- loads the raw CMS dataset
- selects useful provider-service columns
- renames fields into readable snake_case names
- checks missing values
- creates reimbursement-friction features
- creates the proxy denial-risk label
- saves a 250,000-row modeling sample

Output:

```text
data/processed/denial_risk_modeling_sample_250k.csv
```

### Notebook 02: Baseline Modeling

This notebook trains two initial models.

#### Model A: Workflow Validation Model

This model included the direct reimbursement-ratio features used to create the proxy label.

It performed very well, but the performance was inflated because the model had access to features directly related to the target.

This model was useful for proving the workflow worked, but it was not selected as the final production demo model.

#### Model B: Less-Leaky Baseline Model

This model removed the direct ratio features:

```text
charge_to_allowed_ratio
payment_to_charge_ratio
payment_to_allowed_ratio
services_per_beneficiary
```

This made the model more honest because it had to learn from provider, service, volume, and payment fields instead of being handed the direct target-building features.

### Notebook 03: Improved Model Comparison

This notebook:

- loads the saved modeling dataset
- creates safer log-transformed numeric features
- groups rare HCPCS codes
- compares multiple models
- tunes the prediction threshold
- saves production model artifacts

---

## Models Compared

The following models were compared:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost

All models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

For this project, recall and F1 score are especially important because the dashboard is intended to create a review queue. Missing high-risk records may be more costly than flagging extra records for review.

---

## Final Production Demo Model

The selected production demo model was:

```text
HistGradientBoosting
```

Selected threshold:

```text
0.45
```

Final production metrics:

```text
Accuracy:  0.9870
Precision: 0.9695
Recall:    0.9760
F1 Score:  0.9727
ROC AUC:   0.9992
```

These scores are high because the model is learning an engineered reimbursement-risk proxy, not true denied-claim outcomes.

The final model should be interpreted as:

```text
a proxy reimbursement-risk scoring model
```

not:

```text
a real insurance denial prediction model
```

---

## Streamlit Dashboard

The Streamlit app allows users to upload a CMS-style CSV file and receive reimbursement-risk scores.

The uploaded file must include these 13 raw columns:

```text
provider_state
provider_type
medicare_participating
hcpcs_code
is_drug
place_of_service
total_beneficiaries
total_services
total_beneficiary_day_services
avg_submitted_charge
avg_medicare_allowed_amount
avg_medicare_payment_amount
avg_medicare_standardized_amount
```

The app automatically creates backend features:

```text
hcpcs_code_grouped
log_total_services
log_total_beneficiaries
log_total_beneficiary_day_services
log_avg_submitted_charge
log_avg_medicare_allowed_amount
log_avg_medicare_payment_amount
log_avg_medicare_standardized_amount
```

The app then:

- validates required columns
- converts datatypes
- checks invalid numeric values
- checks negative values
- recreates HCPCS grouping
- creates log features
- scores reimbursement-risk probability
- applies the saved threshold
- displays high-risk records
- shows risk summaries by provider type, place of service, and HCPCS code
- allows users to download scored results

---

## Dashboard Outputs

The dashboard shows:

- total uploaded records
- number of high-risk records
- high-risk rate
- average risk probability
- highest-risk records
- risk label distribution
- high-risk rate by provider type
- high-risk rate by place of service
- top high-risk HCPCS codes
- downloadable scored CSV

---

## Project Structure

```text
DenialOps-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│       └── streamlit_test_upload_1000_random_sample.csv
│
├── models/
│   ├── production_denial_risk_model.joblib
│   ├── production_threshold.joblib
│   └── top_hcpcs_codes.joblib
│
├── notebooks/
│   ├── 01_data_load_and_eda.ipynb
│   ├── 02_baseline_denial_risk_model.ipynb
│   └── 03_improve_model_b_model_comparison.ipynb
│
├── reports/
│   ├── model_comparison_results.csv
│   ├── production_model_metrics.csv
│   └── production_test_predictions.csv
│
├── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run Locally

This project lives inside a larger GitHub repository:

```text
LLM-And-Generative-AI/rag/DenialOps-AI
```

GitHub project folder:

```text
https://github.com/Phiziics/LLM-And-Generative-AI/tree/main/rag/DenialOps-AI
```

---

### Option 1: Clone the Full Repository

Use this option if you want the full `LLM-And-Generative-AI` repository.

```bash
git clone https://github.com/Phiziics/LLM-And-Generative-AI.git
cd LLM-And-Generative-AI/rag/DenialOps-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Upload the sample file:

```text
data/sample/streamlit_test_upload_1000_random_sample.csv
```

---

### Option 2: Clone Only the DenialOps-AI Folder Using Sparse Checkout

Use this option if you only want the `DenialOps-AI` project folder and not the entire repository.

```bash
git clone --filter=blob:none --sparse https://github.com/Phiziics/LLM-And-Generative-AI.git
cd LLM-And-Generative-AI
git sparse-checkout set rag/DenialOps-AI
cd rag/DenialOps-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Upload the sample file:

```text
data/sample/streamlit_test_upload_1000_random_sample.csv
```

---

## Required Upload Schema

Your upload file must contain these columns:

| Column | Expected Type | Example |
|---|---|---|
| provider_state | string | TX |
| provider_type | string | Nurse Practitioner |
| medicare_participating | string | Y |
| hcpcs_code | string | 99285 |
| is_drug | string | N |
| place_of_service | string | O |
| total_beneficiaries | integer | 51 |
| total_services | float | 51.0 |
| total_beneficiary_day_services | integer | 51 |
| avg_submitted_charge | float | 1377.98 |
| avg_medicare_allowed_amount | float | 172.98 |
| avg_medicare_payment_amount | float | 133.73 |
| avg_medicare_standardized_amount | float | 130.52 |

Extra columns are ignored by the app. The app only uses the required raw columns and creates backend model features automatically.

---

## Saved Artifacts

The Streamlit app expects the saved production artifacts to exist in the `models/` folder:

```text
models/production_denial_risk_model.joblib
models/production_threshold.joblib
models/top_hcpcs_codes.joblib
```

If these files are missing, rerun Notebook 03 to regenerate the production model artifacts.

The project also saves these report artifacts:

```text
reports/model_comparison_results.csv
reports/production_model_metrics.csv
reports/production_test_predictions.csv
```

---

## Screenshots

Add screenshots in the `screenshots/` folder and reference them here.

Example:

```markdown
![App Home](screenshots/app_home.png)
![Risk Summary](screenshots/risk_summary.png)
![High Risk Records](screenshots/high_risk_records.png)
```

Suggested screenshots:

```text
screenshots/app_home.png
screenshots/data_preview.png
screenshots/risk_summary.png
screenshots/high_risk_records.png
screenshots/risk_charts.png
```

---

## Limitations

This project has important limitations:

1. The model is trained on a proxy target, not true denied claims.
2. The CMS dataset is provider-service level, not hospital claim-line level.
3. The model does not include payer-specific prior authorization rules.
4. The model does not include clinical notes or documentation quality.
5. The model does not include patient eligibility information.
6. The model should not be used to make final denial, payment, or patient-care decisions.
7. The high model scores are partly expected because the proxy target is derived from reimbursement-related patterns in the same dataset.

---

## Future Improvements

Future versions could improve the project by adding:

- real denied-claim data from clinics or billing teams
- payer-policy RAG using CMS LCD/NCD documents
- CMS coverage-policy lookup
- appeal letter generation
- FastAPI prediction endpoint
- Pydantic schema validation
- SHAP explainability
- model monitoring
- Docker deployment
- cloud deployment on AWS, Azure, or GCP
- database storage with PostgreSQL
- scheduled batch scoring pipeline
- dashboard authentication

---

## RAG Extension Plan

A strong next improvement is to add a policy RAG assistant.

The RAG assistant would use CMS policy and coverage documents to help users answer questions such as:

- What documentation may be needed for this service?
- What does CMS say about medical necessity?
- Is this HCPCS code mentioned in coverage documents?
- What policy language could support an appeal?
- Why might this record need billing review?

Potential sources:

```text
CMS Local Coverage Determinations
CMS National Coverage Determinations
CMS Billing and Coding Articles
CMS Review Reason Codes
```

Useful CMS coverage source:

```text
https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx
```

Useful CMS review reason code source:

```text
https://www.cms.gov/data-research/monitoring-programs/medicare-fee-service-compliance-programs/review-reason-codes-and-statements
```

Possible RAG architecture:

```text
CMS policy documents
↓
document loader
↓
chunking
↓
embeddings
↓
Chroma or FAISS vector store
↓
retrieval
↓
LLM response with citations
```

This would turn DenialOps AI from a machine learning dashboard into a fuller healthcare AI decision-support system.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit
- joblib
- matplotlib
- seaborn
- Jupyter Notebook

---

## Business Use Case

DenialOps AI demonstrates how healthcare organizations could use analytics and AI to create a reimbursement-risk review queue.

A future commercial version could help:

- small clinics
- billing companies
- revenue-cycle teams
- specialty practices
- hospital outpatient billing departments

identify records that may need review before submission.

---

## Summary

DenialOps AI is an end-to-end healthcare analytics and machine learning project that turns public CMS provider-service data into a working reimbursement-risk scoring dashboard.

The project demonstrates:

- real-world healthcare data handling
- feature engineering
- proxy target design
- leakage analysis
- model comparison
- threshold tuning
- production artifact saving
- Streamlit dashboard deployment
- honest model limitation communication

The final product is a portfolio-ready healthcare AI project that shows how data science can support revenue-cycle operations while clearly explaining the limits of public data and proxy modeling.