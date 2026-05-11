from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.rag.retrieve import BenefitsRetriever
from src.models.predict_claim_risk import ClaimRiskPredictor
from src.security.audit_logger import AuditLogger
from src.security.redact_pii import redact_sensitive_text


st.set_page_config(
    page_title="BenefitsAI",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_retriever():
    return BenefitsRetriever(project_root=PROJECT_ROOT)


@st.cache_resource
def load_claim_predictor():
    return ClaimRiskPredictor(project_root=PROJECT_ROOT)


@st.cache_resource
def load_audit_logger():
    return AuditLogger(project_root=PROJECT_ROOT)


retriever = load_retriever()
claim_predictor = load_claim_predictor()
audit_logger = load_audit_logger()


st.title("BenefitsAI: Claims, Benefits, and Governance Intelligence")

st.write(
    "Secure AI/ML prototype for benefits retrieval, high-cost claim risk prediction, "
    "and audit-ready governance tracking."
)


tab_rag, tab_claims, tab_audit = st.tabs(
    [
        "Benefits RAG Search",
        "Claim Risk Prediction",
        "Audit Logs",
    ]
)


with tab_rag:
    st.header("Benefits Plan Retrieval")

    query = st.text_input(
        "Ask a benefits or plan question",
        value="What is the out-of-pocket maximum for this plan?",
    )

    top_k = st.slider(
        "Number of source chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button("Search Benefits Plans"):
        redacted_query = redact_sensitive_text(query)

        results = retriever.retrieve(
            query=redacted_query,
            top_k=top_k,
        )

        source_documents = [item["document_name"] for item in results]
        avg_score = sum(item["score"] for item in results) / len(results)

        audit_record = audit_logger.log_event(
            user_query=query,
            task_type="benefits_rag_retrieval",
            model_or_system="sentence-transformers/all-MiniLM-L6-v2 + FAISS",
            response_summary="Retrieved CMS plan chunks for benefits question.",
            source_documents=source_documents,
            confidence_score=avg_score,
            human_review_required=False,
        )

        st.subheader("Redacted Query")
        st.write(redacted_query)

        st.subheader("Audit ID")
        st.code(audit_record["audit_id"])

        st.subheader("Retrieved Sources")

        for item in results:
            with st.expander(
                f"Score: {item['score']:.4f} | {item['document_name']}"
            ):
                st.write(item["text"])


with tab_claims:
    st.header("High-Cost Claim Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        claim_type = st.selectbox(
            "Claim type",
            options=[
                "inpatient",
                "outpatient",
            ],
        )

        claim_duration_days = st.number_input(
            "Claim duration days",
            min_value=1.0,
            value=5.0,
            step=1.0,
        )

    with col2:
        has_provider_id = st.selectbox(
            "Has provider ID?",
            options=[
                1,
                0,
            ],
        )

        has_diagnosis_code = st.selectbox(
            "Has diagnosis code?",
            options=[
                1,
                0,
            ],
        )

    if st.button("Predict Claim Risk"):
        result = claim_predictor.predict_one(
            claim_type=claim_type,
            claim_duration_days=claim_duration_days,
            has_provider_id=has_provider_id,
            has_diagnosis_code=has_diagnosis_code,
        )

        audit_record = audit_logger.log_event(
            user_query=f"Predict claim risk for {claim_type} claim",
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

        st.subheader("Prediction Result")

        st.metric(
            "High-Cost Claim Probability",
            f"{result['high_cost_claim_probability']:.2%}",
        )

        st.write("Predicted high-cost claim:", result["predicted_high_cost_claim"])
        st.write("Human review required:", result["human_review_required"])

        st.subheader("Audit ID")
        st.code(audit_record["audit_id"])


with tab_audit:
    st.header("Audit Log Preview")

    audit_path = PROJECT_ROOT / "data" / "04_predictions" / "audit_logs" / "ai_audit_log.csv"

    if audit_path.exists():
        audit_df = pd.read_csv(audit_path)

        st.write("Audit events:", len(audit_df))

        display_cols = [
            "timestamp_utc",
            "task_type",
            "model_or_system",
            "redacted_query",
            "sensitive_pattern_detected",
            "confidence_score",
            "human_review_required",
        ]

        available_cols = [
            col for col in display_cols
            if col in audit_df.columns
        ]

        st.dataframe(
            audit_df[available_cols].tail(50),
            use_container_width=True,
        )
    else:
        st.info("No audit log found yet. Run a RAG search or claim prediction first.")