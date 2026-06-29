# Import generator
from src.riskradar_generator import RiskRadarGenerator


def test_generator_builds_prompt_with_evidence():
    """
    Test that the generator builds a grounded prompt from evidence.
    """

    # Create sample evidence record
    evidence_records = [
        {
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "section_name": "item_1a_risk_factors",
            "citation_label": "NVDA 2025 10-K Item 1A chunk 1",
            "filing_url": "https://www.sec.gov/example",
            "chunk_text": "NVIDIA faces competition risks and rapid technology changes."
        }
    ]

    # Initialize generator
    generator = RiskRadarGenerator()

    # Build prompt
    prompt = generator.build_rag_prompt(
        question="What competition risks does NVIDIA mention?",
        evidence_records=evidence_records
    )

    # Validate prompt contains key parts
    assert "RiskRadar AI" in prompt
    assert "What competition risks does NVIDIA mention?" in prompt
    assert "SOURCE 1" in prompt
    assert "NVIDIA faces competition risks" in prompt
    assert "[Source 1]" in prompt


def test_generator_formats_sources():
    """
    Test that source metadata is formatted correctly.
    """

    # Create sample evidence record
    evidence_records = [
        {
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "section_name": "item_1a_risk_factors",
            "citation_label": "NVDA 2025 10-K Item 1A chunk 1",
            "filing_url": "https://www.sec.gov/example",
            "chunk_text": "Sample evidence text."
        }
    ]

    # Initialize generator
    generator = RiskRadarGenerator()

    # Format sources
    sources = generator.format_sources(evidence_records)

    # Validate source output
    assert len(sources) == 1
    assert sources[0]["source_number"] == 1
    assert sources[0]["ticker"] == "NVDA"
    assert sources[0]["section_name"] == "item_1a_risk_factors"
    assert sources[0]["citation_label"] == "NVDA 2025 10-K Item 1A chunk 1"


def test_generator_fallback_when_no_evidence():
    """
    Test that generator returns a safe fallback when no evidence is provided.
    """

    # Initialize generator
    generator = RiskRadarGenerator()

    # Generate answer package without evidence
    result = generator.answer_question(
        question="What risks does NVIDIA mention?",
        evidence_records=[]
    )

    # Validate fallback
    assert result["answer"] == "No relevant SEC evidence was retrieved."
    assert result["sources"] == []
    assert result["prompt"] is None
    assert result["generation_time_seconds"] == 0