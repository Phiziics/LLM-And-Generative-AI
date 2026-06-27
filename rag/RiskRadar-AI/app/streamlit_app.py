# Import Streamlit for the user interface
import streamlit as st

# Import requests for calling the FastAPI backend
import requests

# Import pandas for displaying tables
import pandas as pd

# Import json for readable raw response display
import json


# Set Streamlit page configuration
st.set_page_config(
    page_title="RiskRadar AI",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Helper functions
# -----------------------------

def get_api_json(url, timeout=60):
    """
    Send a GET request to the FastAPI backend.
    """

    # Try calling the API
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return {
            "ok": True,
            "data": response.json(),
            "error": None
        }

    # Return error safely
    except Exception as error:
        return {
            "ok": False,
            "data": None,
            "error": str(error)
        }


def post_api_json(url, payload, timeout=700):
    """
    Send a POST request to the FastAPI backend.
    """

    # Try calling the API
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return {
            "ok": True,
            "data": response.json(),
            "error": None
        }

    # Return error safely
    except Exception as error:
        return {
            "ok": False,
            "data": None,
            "error": str(error)
        }


@st.cache_data(ttl=60)
def load_companies(api_base_url):
    """
    Load companies from the FastAPI backend.
    """

    # Call companies endpoint
    result = get_api_json(f"{api_base_url}/companies")

    # Return empty list if request failed
    if not result["ok"]:
        return []

    # Return companies
    return result["data"].get("companies", [])


@st.cache_data(ttl=60)
def load_sections(api_base_url):
    """
    Load available SEC sections from the FastAPI backend.
    """

    # Call sections endpoint
    result = get_api_json(f"{api_base_url}/sections")

    # Return empty list if request failed
    if not result["ok"]:
        return []

    # Return sections
    return result["data"].get("sections", [])


def display_sources(sources):
    """
    Display answer sources in a readable format.
    """

    # Handle no sources
    if not sources:
        st.info("No sources returned.")
        return

    # Loop through sources
    for source in sources:

        # Create expander title
        title = (
            f"Source {source.get('source_number')} — "
            f"{source.get('ticker')} — "
            f"{source.get('section_name')}"
        )

        # Display source metadata
        with st.expander(title):
            st.write("**Company:**", source.get("company_name", ""))
            st.write("**Citation:**", source.get("citation_label", ""))

            # Display filing URL if available
            filing_url = source.get("filing_url", "")
            if filing_url:
                st.write("**Filing URL:**", filing_url)


def display_evidence(evidence):
    """
    Display retrieved evidence chunks.
    """

    # Handle no evidence
    if not evidence:
        st.info("No evidence returned.")
        return

    # Loop through evidence chunks
    for index, row in enumerate(evidence, start=1):

        # Create evidence title
        title = (
            f"Evidence {index} — "
            f"{row.get('ticker')} — "
            f"{row.get('section_name')}"
        )

        # Display evidence details
        with st.expander(title):
            st.write("**Company:**", row.get("company_name", ""))
            st.write("**Citation:**", row.get("citation_label", ""))
            st.write("**Filing URL:**", row.get("filing_url", ""))

            # Show ranking scores if available
            score_cols = st.columns(4)

            with score_cols[0]:
                st.metric("Hybrid Rank", row.get("hybrid_rank", ""))

            with score_cols[1]:
                st.metric("Rerank Rank", row.get("rerank_rank", ""))

            with score_cols[2]:
                st.metric("BM25 Rank", row.get("bm25_rank", ""))

            with score_cols[3]:
                st.metric("Vector Rank", row.get("vector_rank", ""))

            # Display chunk text
            st.write("**Chunk Text:**")
            st.write(row.get("chunk_text", ""))


# -----------------------------
# Sidebar settings
# -----------------------------

st.sidebar.title("RiskRadar AI Settings")

# API base URL setting
api_base_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://127.0.0.1:8000"
)

# Ollama model setting
ollama_model = st.sidebar.text_input(
    "Ollama Model",
    value="llama3.2:3b"
)

# Retrieval settings
top_k = st.sidebar.slider(
    "Final evidence chunks",
    min_value=1,
    max_value=10,
    value=3
)

candidate_k = st.sidebar.slider(
    "Retrieval candidates",
    min_value=5,
    max_value=50,
    value=15
)

use_reranker = st.sidebar.checkbox(
    "Use reranker",
    value=True
)

# API health check
health_result = get_api_json(f"{api_base_url}/health", timeout=30)

if health_result["ok"]:
    st.sidebar.success("API connected")
else:
    st.sidebar.error("API not connected")
    st.sidebar.write(health_result["error"])


# -----------------------------
# Main page
# -----------------------------

st.title("RiskRadar AI")
st.caption("SEC filing RAG system with structured financial context")

# Load companies and sections
companies = load_companies(api_base_url)
sections = load_sections(api_base_url)

# Create ticker options
ticker_options = [""] + [
    company.get("ticker", "")
    for company in companies
]

# Create section options
section_options = [""] + sections


# Create app tabs
tab_ask, tab_retrieve, tab_financials, tab_health = st.tabs(
    [
        "Ask",
        "Retrieve Evidence",
        "Financials",
        "Health"
    ]
)


# -----------------------------
# Ask tab
# -----------------------------

with tab_ask:

    st.header("Ask a Risk Question")

    # Create input columns
    input_col_1, input_col_2 = st.columns(2)

    with input_col_1:
        selected_ticker = st.selectbox(
            "Ticker",
            options=ticker_options,
            index=ticker_options.index("NVDA") if "NVDA" in ticker_options else 0,
            key="ask_ticker"
        )

    with input_col_2:
        selected_section = st.selectbox(
            "SEC Section",
            options=section_options,
            index=section_options.index("item_1a_risk_factors")
            if "item_1a_risk_factors" in section_options
            else 0,
            key="ask_section"
        )

    # Question input
    question = st.text_area(
        "Question",
        value="What AI and competition risks does NVIDIA mention?",
        height=100
    )

    # Financial context option
    include_financial_context = st.checkbox(
        "Include structured financial context",
        value=False
    )

    # Ask button
    ask_clicked = st.button("Generate Answer", type="primary")

    # Run answer request
    if ask_clicked:

        # Build payload
        payload = {
            "question": question,
            "ticker": selected_ticker if selected_ticker else None,
            "section_name": selected_section if selected_section else None,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "use_reranker": use_reranker,
            "include_financial_context": include_financial_context,
            "ollama_model": ollama_model
        }

        # Show spinner while backend works
        with st.spinner("Retrieving evidence and generating answer..."):

            # Choose endpoint
            if include_financial_context:
                endpoint = f"{api_base_url}/answer-with-financials"
            else:
                endpoint = f"{api_base_url}/answer"

            # Call API
            result = post_api_json(
                endpoint,
                payload=payload,
                timeout=700
            )

        # Show result
        if result["ok"]:

            response_data = result["data"]

            st.subheader("Answer")
            st.write(response_data.get("answer", ""))

            # Display metadata
            meta_col_1, meta_col_2, meta_col_3 = st.columns(3)

            with meta_col_1:
                st.metric("Evidence Count", response_data.get("evidence_count", 0))

            with meta_col_2:
                st.metric(
                    "Generation Time",
                    response_data.get("generation_time_seconds", 0)
                )

            with meta_col_3:
                st.metric(
                    "Financial Context Used",
                    str(response_data.get("financial_context_used", False))
                )

            # Show financial context if used
            if response_data.get("financial_context_used"):
                st.subheader("Structured Financial Context")
                st.write(response_data.get("financial_context", ""))

            # Show sources
            st.subheader("Sources")
            display_sources(response_data.get("sources", []))

            # Raw response
            with st.expander("Raw API Response"):
                st.json(response_data)

        else:
            st.error("Answer request failed.")
            st.write(result["error"])


# -----------------------------
# Retrieve Evidence tab
# -----------------------------

with tab_retrieve:

    st.header("Retrieve SEC Evidence Only")

    # Create input columns
    retrieve_col_1, retrieve_col_2 = st.columns(2)

    with retrieve_col_1:
        retrieve_ticker = st.selectbox(
            "Ticker",
            options=ticker_options,
            index=ticker_options.index("NVDA") if "NVDA" in ticker_options else 0,
            key="retrieve_ticker"
        )

    with retrieve_col_2:
        retrieve_section = st.selectbox(
            "SEC Section",
            options=section_options,
            index=section_options.index("item_1a_risk_factors")
            if "item_1a_risk_factors" in section_options
            else 0,
            key="retrieve_section"
        )

    # Retrieval question
    retrieve_question = st.text_area(
        "Retrieval Question",
        value="What AI and competition risks does NVIDIA mention?",
        height=100
    )

    # Retrieval button
    retrieve_clicked = st.button("Retrieve Evidence", type="primary")

    # Run retrieval request
    if retrieve_clicked:

        # Build payload
        payload = {
            "question": retrieve_question,
            "ticker": retrieve_ticker if retrieve_ticker else None,
            "section_name": retrieve_section if retrieve_section else None,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "use_reranker": use_reranker
        }

        # Call retrieve endpoint
        with st.spinner("Retrieving evidence..."):
            result = post_api_json(
                f"{api_base_url}/retrieve",
                payload=payload,
                timeout=180
            )

        # Display result
        if result["ok"]:

            response_data = result["data"]

            st.metric("Evidence Count", response_data.get("evidence_count", 0))

            display_evidence(response_data.get("evidence", []))

            with st.expander("Raw API Response"):
                st.json(response_data)

        else:
            st.error("Retrieval request failed.")
            st.write(result["error"])


# -----------------------------
# Financials tab
# -----------------------------

with tab_financials:

    st.header("Structured Financial Context")

    # Select ticker
    financial_ticker = st.selectbox(
        "Ticker",
        options=ticker_options,
        index=ticker_options.index("NVDA") if "NVDA" in ticker_options else 0,
        key="financial_ticker"
    )

    # Button to load financials
    financial_clicked = st.button("Load Financials", type="primary")

    if financial_clicked:

        # Validate ticker
        if not financial_ticker:
            st.warning("Select a ticker first.")

        else:
            # Call financials endpoint
            with st.spinner("Loading financial data..."):
                result = get_api_json(
                    f"{api_base_url}/financials/{financial_ticker}",
                    timeout=60
                )

            # Display result
            if result["ok"]:

                financial_data = result["data"]

                # Show availability
                if financial_data.get("available"):
                    st.success("Financial data available")
                else:
                    st.warning("No financial data available for this ticker.")

                # Show context paragraph
                st.subheader("Financial Context")
                st.write(financial_data.get("financial_context", ""))

                # Show latest snapshot
                st.subheader("Latest Snapshot")
                latest_snapshot = financial_data.get("latest_snapshot", {})

                if latest_snapshot:
                    latest_snapshot_df = pd.DataFrame([latest_snapshot])
                    st.dataframe(latest_snapshot_df, use_container_width=True)
                else:
                    st.info("No latest snapshot returned.")

                # Show ratio history
                st.subheader("Ratio History")
                ratio_history = financial_data.get("ratio_history", [])

                if ratio_history:
                    ratio_history_df = pd.DataFrame(ratio_history)
                    st.dataframe(ratio_history_df, use_container_width=True)
                else:
                    st.info("No ratio history returned.")

                # Raw response
                with st.expander("Raw API Response"):
                    st.json(financial_data)

            else:
                st.error("Financial request failed.")
                st.write(result["error"])


# -----------------------------
# Health tab
# -----------------------------

with tab_health:

    st.header("API Health")

    # Refresh button
    refresh_health = st.button("Refresh Health")

    # Show health result
    if refresh_health or health_result["ok"]:

        # Reload health
        health_result = get_api_json(f"{api_base_url}/health", timeout=30)

        if health_result["ok"]:
            st.success("API is healthy")
            st.json(health_result["data"])
        else:
            st.error("API health check failed")
            st.write(health_result["error"])

    st.subheader("Run Commands")

    st.code(
        "uvicorn app.main:app --host 127.0.0.1 --port 8000",
        language="powershell"
    )

    st.code(
        "streamlit run app/streamlit_app.py",
        language="powershell"
    )