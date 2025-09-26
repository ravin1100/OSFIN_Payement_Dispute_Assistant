"""
data_loader.py
--------------
Data loading utilities for the CLI interface.

Handles loading and caching of:
- Disputes data (original)
- Classified disputes
- Transaction data
- Resolution data
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class DataLoader:
    """Centralized data loading and caching for CLI queries"""

    def __init__(self, data_dir="data", output_dir="output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self._cache = {}

    def _get_file_path(self, filename: str, is_output: bool = False) -> str:
        """Get full file path for data files"""
        base_dir = self.output_dir if is_output else self.data_dir
        return os.path.join(base_dir, filename)

    def _load_with_cache(self, key: str, file_path: str) -> pd.DataFrame:
        """Load data with caching to avoid repeated file reads"""
        if key not in self._cache:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            self._cache[key] = pd.read_csv(file_path)
        return self._cache[key].copy()

    def load_disputes(self) -> pd.DataFrame:
        """Load original disputes data"""
        file_path = self._get_file_path("disputes.csv")
        return self._load_with_cache("disputes", file_path)

    def load_classified_disputes(self) -> pd.DataFrame:
        """Load classified disputes data"""
        file_path = self._get_file_path("classified_disputes.csv", is_output=True)
        return self._load_with_cache("classified_disputes", file_path)

    def load_transactions(self) -> pd.DataFrame:
        """Load transactions data"""
        file_path = self._get_file_path("transactions.csv")
        return self._load_with_cache("transactions", file_path)

    def load_resolutions(self) -> pd.DataFrame:
        """Load resolutions data"""
        file_path = self._get_file_path("resolutions.csv", is_output=True)
        return self._load_with_cache("resolutions", file_path)

    def load_combined_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources and return as dictionary"""
        try:
            original_disputes = self.load_disputes()
            classified_disputes = self.load_classified_disputes()
            transactions = self.load_transactions()
            resolutions = self.load_resolutions()

            # Merge data for comprehensive analysis
            # Join original disputes with classifications
            disputes_with_class = original_disputes.merge(
                classified_disputes[["dispute_id", "predicted_category", "confidence"]],
                on="dispute_id",
                how="left",
            )

            # Join with resolutions
            full_data = disputes_with_class.merge(
                resolutions[["dispute_id", "suggested_action", "justification"]],
                on="dispute_id",
                how="left",
            )

            # Convert timestamps
            full_data["created_at"] = pd.to_datetime(full_data["created_at"])
            transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])

            return {
                "disputes": original_disputes,
                "classified_disputes": classified_disputes,
                "transactions": transactions,
                "resolutions": resolutions,
                "combined": full_data,
            }

        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("💡 Make sure to run the pipeline first: python src/pipeline.py")
            raise

    def get_data_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the loaded data"""
        try:
            data = self.load_combined_data()
            combined = data["combined"]
            transactions = data["transactions"]

            return {
                "total_disputes": len(combined),
                "dispute_categories": combined["predicted_category"]
                .value_counts()
                .to_dict(),
                "resolution_actions": combined["suggested_action"]
                .value_counts()
                .to_dict(),
                "total_transactions": len(transactions),
                "date_range": {
                    "disputes_from": combined["created_at"].min().strftime("%Y-%m-%d"),
                    "disputes_to": combined["created_at"].max().strftime("%Y-%m-%d"),
                    "transactions_from": transactions["timestamp"]
                    .min()
                    .strftime("%Y-%m-%d"),
                    "transactions_to": transactions["timestamp"]
                    .max()
                    .strftime("%Y-%m-%d"),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_cache(self):
        """Clear the data cache to force reload"""
        self._cache.clear()


# Utility functions for date filtering
def get_today_filter() -> datetime:
    """Get start of today for filtering"""
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def get_date_range_filter(days_back: int) -> datetime:
    """Get date filter for N days back"""
    return datetime.now() - timedelta(days=days_back)


def filter_by_date(
    df: pd.DataFrame, date_column: str, start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Filter DataFrame by date range"""
    if start_date is None:
        return df

    return df[df[date_column] >= start_date]
