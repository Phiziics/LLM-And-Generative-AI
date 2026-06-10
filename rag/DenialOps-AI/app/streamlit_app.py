# Import libraries for building the dashboard, loading the model, scoring data, and handling features

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Set Streamlit page configuration

st.set_page_config(
    page_title="DenialOps AI",
    page_icon="🏥",
    layout="wide"
)


# Add title and project explanation

st.title("DenialOps AI")
st.subheader("Proxy Reimbursement-Risk Scoring Dashboard")

st.write(
    """
    DenialOps AI scores CMS-style provider-service records for reimbursement-friction risk.
    The model is designed to help billing and revenue-cycle teams create a review queue
    for records that may deserve closer attention.
    """
)


# Add honest model disclaimer

st.warning(
    """
    Disclaimer: This model is trained on an engineered proxy reimbursement-risk label,
    not true denied-claim outcomes. The public CMS dataset does not include actual
    insurance denial labels. This dashboard should be used as a review-support tool,
    not a final denial decision system.
    """
)


# Load the saved production model

@st.cache_resource
def load_model():
    # Load the production HistGradientBoosting pipeline saved from Notebook 03

    model_path = "models/production_denial_risk_model.joblib"
    model = joblib.load(model_path)
    return model


# Load the saved production threshold

@st.cache_resource
def load_threshold():
    # Load the selected probability threshold saved from Notebook 03

    threshold_path = "models/production_threshold.joblib"
    threshold = joblib.load(threshold_path)
    return threshold


# Load model and threshold

model = load_model()
threshold = load_threshold()


# Display model information

st.info(f"Production model: HistGradientBoosting | Current threshold: {threshold:.2f}")


# Define required columns from uploaded CSV

required_columns = [
    "provider_state",
    "provider_type",
    "medicare_participating",
    "hcpcs_code",
    "is_drug",
    "place_of_service",
    "total_beneficiaries",
    "total_services",
    "total_beneficiary_day_services",
    "avg_submitted_charge",
    "avg_medicare_allowed_amount",
    "avg_medicare_payment_amount",
    "avg_medicare_standardized_amount"
]


# Define feature columns expected by the production model

model_features = [
    "provider_state",
    "provider_type",
    "medicare_participating",
    "hcpcs_code_grouped",
    "is_drug",
    "place_of_service",
    "total_beneficiaries",
    "total_services",
    "total_beneficiary_day_services",
    "avg_submitted_charge",
    "avg_medicare_allowed_amount",
    "avg_medicare_payment_amount",
    "avg_medicare_standardized_amount",
    "log_total_services",
    "log_total_beneficiaries",
    "log_total_beneficiary_day_services",
    "log_avg_submitted_charge",
    "log_avg_medicare_allowed_amount",
    "log_avg_medicare_payment_amount",
    "log_avg_medicare_standardized_amount"
]


# Create app-side features to match Notebook 03 feature engineering

def create_model_features(input_df):
    # Copy uploaded data so original data remains unchanged

    app_df = input_df.copy()


    # Convert HCPCS code to string for consistent grouping

    app_df["hcpcs_code"] = app_df["hcpcs_code"].astype(str)


    # Create grouped HCPCS code feature
    # Note: In the notebook, top 500 HCPCS codes were kept and others were grouped.
    # For the app demo, unseen codes will still be handled by OneHotEncoder(handle_unknown="ignore").

    app_df["hcpcs_code_grouped"] = app_df["hcpcs_code"]


    # Create log-transformed numeric features to match Notebook 03

    app_df["log_total_services"] = np.log1p(app_df["total_services"])
    app_df["log_total_beneficiaries"] = np.log1p(app_df["total_beneficiaries"])
    app_df["log_total_beneficiary_day_services"] = np.log1p(app_df["total_beneficiary_day_services"])

    app_df["log_avg_submitted_charge"] = np.log1p(app_df["avg_submitted_charge"])
    app_df["log_avg_medicare_allowed_amount"] = np.log1p(app_df["avg_medicare_allowed_amount"])
    app_df["log_avg_medicare_payment_amount"] = np.log1p(app_df["avg_medicare_payment_amount"])
    app_df["log_avg_medicare_standardized_amount"] = np.log1p(app_df["avg_medicare_standardized_amount"])


    # Return only the columns expected by the production model

    return app_df[model_features]


# Create file uploader

uploaded_file = st.file_uploader(
    "Upload a CMS-style provider-service CSV file",
    type=["csv"]
)


# Run scoring workflow after upload

if uploaded_file is not None:

    # Load uploaded CSV

    input_df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(input_df.head())


    # Validate required columns

    missing_columns = [col for col in required_columns if col not in input_df.columns]

    if missing_columns:
        st.error("The uploaded file is missing required columns:")
        st.write(missing_columns)
        st.stop()


    # Create model-ready features

    model_input = create_model_features(input_df)


    # Generate reimbursement-risk probabilities

    risk_probabilities = model.predict_proba(model_input)[:, 1]


    # Apply production threshold

    risk_predictions = (risk_probabilities >= threshold).astype(int)


    # Add model outputs to the original uploaded data

    results_df = input_df.copy()
    results_df["predicted_high_risk_probability"] = risk_probabilities
    results_df["predicted_reimbursement_risk"] = risk_predictions
    results_df["risk_label"] = results_df["predicted_reimbursement_risk"].map({
        0: "Low Risk",
        1: "High Risk"
    })


    # Create dashboard metrics

    total_records = len(results_df)
    high_risk_records = int(results_df["predicted_reimbursement_risk"].sum())
    low_risk_records = total_records - high_risk_records
    high_risk_rate = high_risk_records / total_records
    average_risk_probability = results_df["predicted_high_risk_probability"].mean()


    # Display KPI cards

    st.write("### Risk Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("High-Risk Records", f"{high_risk_records:,}")
    col3.metric("High-Risk Rate", f"{high_risk_rate:.2%}")
    col4.metric("Avg Risk Probability", f"{average_risk_probability:.2%}")


    # Show highest-risk records

    st.write("### Highest-Risk Records")

    highest_risk_df = (
        results_df
        .sort_values("predicted_high_risk_probability", ascending=False)
        .head(25)
    )

    st.dataframe(highest_risk_df)


    # Show risk distribution by label

    st.write("### Risk Label Distribution")

    risk_distribution = (
        results_df["risk_label"]
        .value_counts()
        .reset_index()
    )

    risk_distribution.columns = ["risk_label", "count"]

    st.bar_chart(
        risk_distribution,
        x="risk_label",
        y="count"
    )


    # Show high-risk rate by provider type

    st.write("### High-Risk Rate by Provider Type")

    risk_by_provider_type = (
        results_df.groupby("provider_type")["predicted_reimbursement_risk"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    risk_by_provider_type["high_risk_rate_percent"] = (
        risk_by_provider_type["predicted_reimbursement_risk"] * 100
    )

    st.bar_chart(
        risk_by_provider_type,
        x="provider_type",
        y="high_risk_rate_percent"
    )


    # Show high-risk rate by place of service

    st.write("### High-Risk Rate by Place of Service")

    risk_by_place = (
        results_df.groupby("place_of_service")["predicted_reimbursement_risk"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    risk_by_place["high_risk_rate_percent"] = (
        risk_by_place["predicted_reimbursement_risk"] * 100
    )

    st.bar_chart(
        risk_by_place,
        x="place_of_service",
        y="high_risk_rate_percent"
    )


    # Show top high-risk HCPCS codes

    st.write("### Top 20 High-Risk HCPCS Codes")

    risk_by_hcpcs = (
        results_df.groupby("hcpcs_code")["predicted_reimbursement_risk"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    risk_by_hcpcs["high_risk_rate_percent"] = (
        risk_by_hcpcs["predicted_reimbursement_risk"] * 100
    )

    st.bar_chart(
        risk_by_hcpcs,
        x="hcpcs_code",
        y="high_risk_rate_percent"
    )


    # Add probability distribution

    st.write("### Predicted Risk Probability Distribution")

    st.line_chart(
        results_df["predicted_high_risk_probability"]
        .sort_values()
        .reset_index(drop=True)
    )


    # Add download button for scored results

    st.write("### Download Scored Results")

    csv_output = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download scored CSV",
        data=csv_output,
        file_name="denialops_scored_results.csv",
        mime="text/csv"
    )

else:

    # Show upload instructions before file is uploaded

    st.info("Upload a CSV file to generate reimbursement-risk scores.")