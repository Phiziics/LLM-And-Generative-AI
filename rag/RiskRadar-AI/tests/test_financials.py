# Import financial data layer
from src.riskradar_financials import RiskRadarFinancials


def test_financials_health_check():
    """
    Test that the structured financial layer loads successfully.
    """

    # Initialize financial layer
    financials = RiskRadarFinancials()

    # Run health check
    health = financials.health_check()

    # Check that core financial files exist
    assert health["financial_context_file_exists"] is True
    assert health["latest_snapshot_file_exists"] is True
    assert health["financial_ratios_file_exists"] is True

    # Check that rows were loaded
    assert health["financial_context_rows"] > 0
    assert health["latest_snapshot_rows"] > 0
    assert health["financial_ratios_rows"] > 0

    # Check expected tickers
    expected_tickers = {"AAPL", "AMD", "MSFT", "NVDA", "TSLA"}
    available_tickers = set(health["available_tickers"])

    assert expected_tickers.issubset(available_tickers)


def test_get_financial_package_for_nvda():
    """
    Test that NVDA financial package is available.
    """

    # Initialize financial layer
    financials = RiskRadarFinancials()

    # Get package
    package = financials.get_financial_package("NVDA")

    # Validate package
    assert package["ticker"] == "NVDA"
    assert package["available"] is True
    assert package["financial_context"] is not None
    assert isinstance(package["latest_snapshot"], dict)
    assert isinstance(package["ratio_history"], list)

    # Make sure latest snapshot is not empty
    assert len(package["latest_snapshot"]) > 0

    # Make sure ratio history is not empty
    assert len(package["ratio_history"]) > 0