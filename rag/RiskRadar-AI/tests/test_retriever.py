# Import retriever
from src.riskradar_retriever import RiskRadarRetriever


def test_retriever_health_check():
    """
    Test that retriever loads chunks, Chroma, embeddings, and company metadata.
    """

    # Initialize retriever without reranker for faster health test
    retriever = RiskRadarRetriever(load_reranker=False)

    # Run health check
    health = retriever.health_check()

    # Validate core files and vectorstore
    assert health["rag_chunks_file_exists"] is True
    assert health["embedding_summary_file_exists"] is True
    assert health["chroma_dir_exists"] is True

    # Validate counts
    assert health["chunk_rows"] > 0
    assert health["chroma_records"] == health["chunk_rows"]

    # Validate companies
    company_tickers = {
        company["ticker"]
        for company in health["available_companies"]
    }

    expected_tickers = {"AAPL", "AMD", "MSFT", "NVDA", "TSLA"}

    assert expected_tickers.issubset(company_tickers)


def test_retriever_returns_nvda_evidence():
    """
    Test that retriever returns evidence for an NVDA risk question.
    """

    # Initialize retriever without reranker for faster test
    retriever = RiskRadarRetriever(load_reranker=False)

    # Retrieve evidence
    evidence = retriever.retrieve_evidence_records(
        query="What AI and competition risks does NVIDIA mention?",
        ticker="NVDA",
        section_name="item_1a_risk_factors",
        top_k=3,
        candidate_k=15,
        use_reranker=False
    )

    # Validate evidence count
    assert len(evidence) > 0

    # Validate first evidence record
    first = evidence[0]

    assert first["ticker"] == "NVDA"
    assert first["section_name"] == "item_1a_risk_factors"
    assert "chunk_text" in first
    assert len(first["chunk_text"]) > 100
    assert "citation_label" in first