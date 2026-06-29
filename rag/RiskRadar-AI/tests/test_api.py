# Import FastAPI test client
from fastapi.testclient import TestClient

# Import app
from app.main import app


def test_api_health_endpoint():
    """
    Test API health endpoint.
    """

    # Use context manager so FastAPI startup runs
    with TestClient(app) as client:

        # Call health endpoint
        response = client.get("/health")

        # Validate response
        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "ok"
        assert data["retriever_loaded"] is True
        assert data["financials_loaded"] is True


def test_api_companies_endpoint():
    """
    Test companies endpoint.
    """

    # Use context manager so FastAPI startup runs
    with TestClient(app) as client:

        # Call companies endpoint
        response = client.get("/companies")

        # Validate response
        assert response.status_code == 200

        data = response.json()

        assert "companies" in data
        assert len(data["companies"]) > 0

        tickers = {
            company["ticker"]
            for company in data["companies"]
        }

        expected_tickers = {"AAPL", "AMD", "MSFT", "NVDA", "TSLA"}

        assert expected_tickers.issubset(tickers)


def test_api_retrieve_endpoint():
    """
    Test retrieve endpoint without Ollama.
    """

    # Use context manager so FastAPI startup runs
    with TestClient(app) as client:

        # Create request payload
        payload = {
            "question": "What AI and competition risks does NVIDIA mention?",
            "ticker": "NVDA",
            "section_name": "item_1a_risk_factors",
            "top_k": 3,
            "candidate_k": 15,
            "use_reranker": False
        }

        # Call retrieve endpoint
        response = client.post("/retrieve", json=payload)

        # Validate response
        assert response.status_code == 200

        data = response.json()

        assert data["ticker"] == "NVDA"
        assert data["evidence_count"] > 0
        assert len(data["evidence"]) > 0
        assert data["evidence"][0]["ticker"] == "NVDA"


def test_api_financials_endpoint():
    """
    Test financials endpoint.
    """

    # Use context manager so FastAPI startup runs
    with TestClient(app) as client:

        # Call financials endpoint
        response = client.get("/financials/NVDA")

        # Validate response
        assert response.status_code == 200

        data = response.json()

        assert data["ticker"] == "NVDA"
        assert data["available"] is True
        assert data["financial_context"] is not None
        assert isinstance(data["latest_snapshot"], dict)
        assert isinstance(data["ratio_history"], list)