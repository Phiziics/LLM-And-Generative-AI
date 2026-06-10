# Import libraries for building the dashboard, loading artifacts, validating data, and scoring records

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
    # Load the production model pipeline saved from Notebook 03

    model_path = "models/production_denial_risk_model.joblib"
    return joblib.load(model_path)


# Load the saved production threshold

@st.cache_resource
def load_threshold():
    # Load the selected probability threshold saved from Notebook 03

    threshold_path = "models/production_threshold.joblib"
    return joblib.load(threshold_path)


# Load the saved top HCPCS codes used during training

@st.cache_resource
def load_top_hcpcs_codes():
    # Load the top HCPCS codes so the app can recreate the same grouping logic used in training

    top_hcpcs_path = "models/top_hcpcs_codes.joblib"
    return joblib.load(top_hcpcs_path)


# Load production artifacts

model = load_model()
threshold = load_threshold()
top_hcpcs_codes = load_top_hcpcs_codes()


# Display model information

st.info(f"Production model: HistGradientBoosting | Current threshold: {threshold:.2f}")


# Define raw columns required from uploaded CSV

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


# Define expected raw datatypes for validation

numeric_columns = [
    "total_beneficiaries",
    "total_services",
    "total_beneficiary_day_services",
    "avg_submitted_charge",
    "avg_medicare_allowed_amount",
    "avg_medicare_payment_amount",
    "avg_medicare_standardized_amount"
]

categorical_columns = [
    "provider_state",
    "provider_type",
    "medicare_participating",
    "hcpcs_code",
    "is_drug",
    "place_of_service"
]


# Define final model features expected by the saved pipeline

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


# Validate uploaded dataframe columns

def validate_required_columns(input_df):
    # Check whether all required raw columns are present in the uploaded CSV

    missing_columns = [col for col in required_columns if col not in input_df.columns]

    if missing_columns:
        return False, missing_columns

    return True, []


# Convert uploaded columns to expected datatypes

def convert_datatypes(input_df):
    # Convert uploaded columns to the expected model datatypes

    cleaned_df = input_df.copy()

    for col in categorical_columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    for col in numeric_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    return cleaned_df


# Validate numeric columns after conversion

def validate_numeric_values(input_df):
    # Check whether numeric fields contain missing values after conversion

    missing_numeric_counts = input_df[numeric_columns].isna().sum()

    invalid_numeric_columns = missing_numeric_counts[missing_numeric_counts > 0]

    if len(invalid_numeric_columns) > 0:
        return False, invalid_numeric_columns

    return True, invalid_numeric_columns


# Validate that numeric values are non-negative

def validate_non_negative_values(input_df):
    # Check whether numeric model fields contain negative values

    negative_counts = {}

    for col in numeric_columns:
        negative_count = (input_df[col] < 0).sum()

        if negative_count > 0:
            negative_counts[col] = int(negative_count)

    if negative_counts:
        return False, negative_counts

    return True, {}


# Create model features

def create_model_features(input_df, top_hcpcs_codes):
    # Create the same backend features used during model training

    app_df = input_df.copy()


    # Group HCPCS codes using the saved top 500 codes from training

    app_df["hcpcs_code_grouped"] = np.where(
        app_df["hcpcs_code"].isin(top_hcpcs_codes),
        app_df["hcpcs_code"],
        "OTHER"
    )


    # Create log-transformed numeric features

    app_df["log_total_services"] = np.log1p(app_df["total_services"])
    app_df["log_total_beneficiaries"] = np.log1p(app_df["total_beneficiaries"])
    app_df["log_total_beneficiary_day_services"] = np.log1p(app_df["total_beneficiary_day_services"])

    app_df["log_avg_submitted_charge"] = np.log1p(app_df["avg_submitted_charge"])
    app_df["log_avg_medicare_allowed_amount"] = np.log1p(app_df["avg_medicare_allowed_amount"])
    app_df["log_avg_medicare_payment_amount"] = np.log1p(app_df["avg_medicare_payment_amount"])
    app_df["log_avg_medicare_standardized_amount"] = np.log1p(app_df["avg_medicare_standardized_amount"])


    # Return only the features expected by the production pipeline

    return app_df[model_features]


# Create file uploader

uploaded_file = st.file_uploader(
    "Upload a CMS-style provider-service CSV file",
    type=["csv"]
)


# Show expected schema to the user

with st.expander("View required upload columns"):
    st.write("Your CSV must include these 13 raw columns:")
    st.code("\n".join(required_columns))


# Run scoring workflow after upload

if uploaded_file is not None:

    # Load uploaded CSV

    input_df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(input_df.head())


    # Check required columns

    columns_valid, missing_columns = validate_required_columns(input_df)

    if not columns_valid:
        st.error("The uploaded file is missing required columns:")
        st.write(missing_columns)
        st.stop()


    # Keep only required raw columns and convert datatypes

    cleaned_df = input_df[required_columns].copy()
    cleaned_df = convert_datatypes(cleaned_df)


    # Validate numeric conversion

    numeric_valid, invalid_numeric_columns = validate_numeric_values(cleaned_df)

    if not numeric_valid:
        st.error("Some numeric columns contain invalid or missing values after conversion.")
        st.write(invalid_numeric_columns)
        st.stop()


    # Validate non-negative values

    non_negative_valid, negative_counts = validate_non_negative_values(cleaned_df)

    if not non_negative_valid:
        st.error("Some numeric columns contain negative values, which are not valid for this scoring workflow.")
        st.write(negative_counts)
        st.stop()


    # Create final model features

    model_input = create_model_features(cleaned_df, top_hcpcs_codes)


    # Generate risk probabilities

    risk_probabilities = model.predict_proba(model_input)[:, 1]


    # Apply production threshold

    risk_predictions = (risk_probabilities >= threshold).astype(int)


    # Add results to original uploaded data

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


    # Show risk label distribution

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


    # Show top 20 high-risk HCPCS codes

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


    # Show sorted probability curve

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

    # Show instruction before a file is uploaded

    st.info("Upload a CSV file to generate reimbursement-risk scores.")