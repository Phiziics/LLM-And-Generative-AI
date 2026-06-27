# Import tools for working with file paths
from pathlib import Path

# Import pandas for loading financial CSV files
import pandas as pd

# Import numpy for handling missing values and numeric types
import numpy as np


class RiskRadarFinancials:
    """
    Structured financial data layer for RiskRadar AI.

    This class loads financial outputs created in notebook 10:
    - latest financial snapshot
    - financial ratios
    - financial context

    The API will use this class later to add structured financial context
    to RAG answers.
    """

    def __init__(self, project_root=None):
        """
        Initialize the financial data layer.
        """

        # Resolve project root automatically if not provided
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)

        # Define important project folders
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"

        # Define financial output files from notebook 10
        self.financial_context_file = (
            self.processed_dir / "sec_company_financial_context.csv"
        )

        self.latest_snapshot_file = (
            self.processed_dir / "sec_company_latest_financial_snapshot.csv"
        )

        self.financial_ratios_file = (
            self.processed_dir / "sec_company_financial_ratios.csv"
        )

        self.financial_metrics_file = (
            self.processed_dir / "sec_company_financial_metrics.csv"
        )

        # Load available financial data
        self._load_financial_data()

    def _load_csv_if_exists(self, file_path):
        """
        Load a CSV file if it exists.

        Return an empty DataFrame if the file does not exist.
        """

        # Check whether the file exists
        if file_path.exists():

            # Load CSV and fill missing values with empty strings for safer API output
            return pd.read_csv(file_path).fillna("")

        # Return empty DataFrame if file is missing
        return pd.DataFrame()

    def _load_financial_data(self):
        """
        Load all structured financial datasets.
        """

        # Load financial context table
        self.financial_context_df = self._load_csv_if_exists(
            self.financial_context_file
        )

        # Load latest financial snapshot table
        self.latest_snapshot_df = self._load_csv_if_exists(
            self.latest_snapshot_file
        )

        # Load financial ratios table
        self.financial_ratios_df = self._load_csv_if_exists(
            self.financial_ratios_file
        )

        # Load financial metrics table
        self.financial_metrics_df = self._load_csv_if_exists(
            self.financial_metrics_file
        )

    def reload(self):
        """
        Reload financial data from disk.

        This is useful after rerunning notebook 10.
        """

        # Reload financial files
        self._load_financial_data()

        # Return status
        return {
            "status": "reloaded",
            "financial_context_rows": len(self.financial_context_df),
            "latest_snapshot_rows": len(self.latest_snapshot_df),
            "financial_ratios_rows": len(self.financial_ratios_df),
            "financial_metrics_rows": len(self.financial_metrics_df),
        }

    def get_available_tickers(self):
        """
        Return tickers available in the structured financial layer.
        """

        # Prefer latest snapshot because it represents final cleaned financial data
        if not self.latest_snapshot_df.empty and "ticker" in self.latest_snapshot_df.columns:
            tickers = sorted(self.latest_snapshot_df["ticker"].unique().tolist())
            return tickers

        # Fallback to financial context table
        if not self.financial_context_df.empty and "ticker" in self.financial_context_df.columns:
            tickers = sorted(self.financial_context_df["ticker"].unique().tolist())
            return tickers

        # Return empty list if no data is available
        return []

    def _standardize_ticker(self, ticker):
        """
        Standardize ticker input.
        """

        # Return None if ticker is missing
        if ticker is None:
            return None

        # Convert ticker to uppercase string
        return str(ticker).upper().strip()

    def _convert_record_to_json_safe(self, record):
        """
        Convert a pandas record into JSON-safe Python values.
        """

        # Create empty dictionary for clean output
        clean_record = {}

        # Loop through each key-value pair
        for key, value in record.items():

            # Convert numpy integers to normal Python integers
            if isinstance(value, np.integer):
                clean_record[key] = int(value)

            # Convert numpy floats to normal Python floats
            elif isinstance(value, np.floating):
                clean_record[key] = float(value)

            # Convert NaN values to None
            elif pd.isna(value):
                clean_record[key] = None

            # Keep normal values as-is
            else:
                clean_record[key] = value

        # Return JSON-safe record
        return clean_record

    def get_financial_context(self, ticker):
        """
        Return the short financial context paragraph for one ticker.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Return fallback if context data is missing
        if self.financial_context_df.empty:
            return None

        # Return fallback if required columns are missing
        if "ticker" not in self.financial_context_df.columns:
            return None

        if "financial_context" not in self.financial_context_df.columns:
            return None

        # Filter to ticker
        ticker_context = self.financial_context_df[
            self.financial_context_df["ticker"] == ticker
        ]

        # Return None if ticker was not found
        if ticker_context.empty:
            return None

        # Return context paragraph
        return str(ticker_context.iloc[0]["financial_context"])

    def get_latest_snapshot(self, ticker):
        """
        Return latest financial snapshot for one ticker.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Return empty dictionary if snapshot data is missing
        if self.latest_snapshot_df.empty:
            return {}

        # Return empty dictionary if ticker column is missing
        if "ticker" not in self.latest_snapshot_df.columns:
            return {}

        # Filter to ticker
        ticker_snapshot = self.latest_snapshot_df[
            self.latest_snapshot_df["ticker"] == ticker
        ]

        # Return empty dictionary if ticker is missing
        if ticker_snapshot.empty:
            return {}

        # Convert first row to dictionary
        record = ticker_snapshot.iloc[0].to_dict()

        # Return JSON-safe record
        return self._convert_record_to_json_safe(record)

    def get_ratio_history(self, ticker):
        """
        Return financial ratio history for one ticker.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Return empty list if ratio data is missing
        if self.financial_ratios_df.empty:
            return []

        # Return empty list if ticker column is missing
        if "ticker" not in self.financial_ratios_df.columns:
            return []

        # Filter to ticker
        ticker_ratios = self.financial_ratios_df[
            self.financial_ratios_df["ticker"] == ticker
        ].copy()

        # Return empty list if ticker is missing
        if ticker_ratios.empty:
            return []

        # Sort by fiscal year if available
        if "fy" in ticker_ratios.columns:
            ticker_ratios = ticker_ratios.sort_values("fy")

        # Convert records to JSON-safe dictionaries
        records = [
            self._convert_record_to_json_safe(record)
            for record in ticker_ratios.to_dict(orient="records")
        ]

        # Return records
        return records

    def get_financial_package(self, ticker):
        """
        Return all structured financial data for one ticker.

        This is the main method the API will use later.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Build financial package
        package = {
            "ticker": ticker,
            "financial_context": self.get_financial_context(ticker),
            "latest_snapshot": self.get_latest_snapshot(ticker),
            "ratio_history": self.get_ratio_history(ticker),
        }

        # Add availability flag
        package["available"] = (
            package["financial_context"] is not None
            or bool(package["latest_snapshot"])
            or len(package["ratio_history"]) > 0
        )

        # Return package
        return package

    def health_check(self):
        """
        Return status of the structured financial layer.
        """

        # Return status dictionary
        return {
            "financial_context_file_exists": self.financial_context_file.exists(),
            "latest_snapshot_file_exists": self.latest_snapshot_file.exists(),
            "financial_ratios_file_exists": self.financial_ratios_file.exists(),
            "financial_metrics_file_exists": self.financial_metrics_file.exists(),
            "financial_context_rows": len(self.financial_context_df),
            "latest_snapshot_rows": len(self.latest_snapshot_df),
            "financial_ratios_rows": len(self.financial_ratios_df),
            "financial_metrics_rows": len(self.financial_metrics_df),
            "available_tickers": self.get_available_tickers(),
        }