"""
query_engine.py
---------------
Core query processing engine for the CLI interface.

Handles:
- Count queries (by category, date, status)
- List queries (filtered disputes)
- Analysis queries (breakdowns, summaries)
- Export functionality
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from data_loader import (
    DataLoader,
    get_today_filter,
    get_date_range_filter,
    filter_by_date,
)


class QueryEngine:
    """Core query processing engine for dispute analysis"""

    def __init__(self):
        self.data_loader = DataLoader()
        self._data = None

    def _load_data(self, force_reload: bool = False):
        """Load data if not already loaded"""
        if self._data is None or force_reload:
            self._data = self.data_loader.load_combined_data()

    def _get_combined_data(self) -> pd.DataFrame:
        """Get the combined dataset"""
        self._load_data()
        return self._data["combined"]

    def _format_results(
        self, results: Union[int, pd.DataFrame, Dict], query_type: str
    ) -> Dict[str, Any]:
        """Format query results for consistent output"""
        return {
            "query_type": query_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }

    # ===== COUNT QUERIES =====

    def count_disputes_by_category(
        self, category: Optional[str] = None, date_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Count disputes by category with optional date filtering"""
        df = self._get_combined_data()

        # Apply date filter
        if date_filter == "today":
            df = filter_by_date(df, "created_at", get_today_filter())
        elif date_filter == "week":
            df = filter_by_date(df, "created_at", get_date_range_filter(7))

        # Apply category filter
        if category:
            df = df[df["predicted_category"] == category.upper()]
            count = len(df)
            query_type = f"count_{category.lower()}_disputes"
        else:
            count = df["predicted_category"].value_counts().to_dict()
            query_type = "count_disputes_by_category"

        return self._format_results(count, query_type)

    def count_by_status(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Count disputes by resolution status"""
        df = self._get_combined_data()

        if status:
            count = len(df[df["suggested_action"] == status])
            query_type = f"count_{status.lower().replace(' ', '_')}_disputes"
        else:
            count = df["suggested_action"].value_counts().to_dict()
            query_type = "count_by_resolution_status"

        return self._format_results(count, query_type)

    def count_high_value_disputes(self, threshold: int = 5000) -> Dict[str, Any]:
        """Count disputes above amount threshold"""
        df = self._get_combined_data()
        high_value = df[df["amount"] >= threshold]

        result = {
            "total_high_value": len(high_value),
            "threshold": threshold,
            "by_category": high_value["predicted_category"].value_counts().to_dict(),
            "total_amount": high_value["amount"].sum(),
        }

        return self._format_results(result, "count_high_value_disputes")

    # ===== LIST QUERIES =====

    def list_disputes_by_category(
        self, category: str, limit: int = 10
    ) -> Dict[str, Any]:
        """List disputes of specific category"""
        df = self._get_combined_data()
        filtered = df[df["predicted_category"] == category.upper()].head(limit)

        result = filtered[
            [
                "dispute_id",
                "customer_id",
                "amount",
                "description",
                "suggested_action",
                "created_at",
            ]
        ].to_dict("records")

        return self._format_results(result, f"list_{category.lower()}_disputes")

    def list_unresolved_disputes(self, limit: int = 20) -> Dict[str, Any]:
        """List disputes that need attention (not auto-refund)"""
        df = self._get_combined_data()
        unresolved = df[df["suggested_action"] != "Auto-refund"].head(limit)

        result = unresolved[
            [
                "dispute_id",
                "predicted_category",
                "amount",
                "suggested_action",
                "description",
            ]
        ].to_dict("records")

        return self._format_results(result, "list_unresolved_disputes")

    def list_fraud_disputes(self, limit: int = 10) -> Dict[str, Any]:
        """List fraud disputes with details"""
        df = self._get_combined_data()
        fraud_cases = df[df["predicted_category"] == "FRAUD"].head(limit)

        result = fraud_cases[
            [
                "dispute_id",
                "customer_id",
                "amount",
                "description",
                "suggested_action",
                "justification",
            ]
        ].to_dict("records")

        return self._format_results(result, "list_fraud_disputes")

    # ===== ANALYSIS QUERIES =====

    def breakdown_by_type(self) -> Dict[str, Any]:
        """Comprehensive breakdown of disputes by type"""
        df = self._get_combined_data()

        breakdown = {}
        for category in df["predicted_category"].unique():
            cat_data = df[df["predicted_category"] == category]
            breakdown[category] = {
                "count": len(cat_data),
                "total_amount": cat_data["amount"].sum(),
                "avg_amount": round(cat_data["amount"].mean(), 2),
                "resolution_actions": cat_data["suggested_action"]
                .value_counts()
                .to_dict(),
            }

        return self._format_results(breakdown, "breakdown_by_type")

    def breakdown_by_channel(self) -> Dict[str, Any]:
        """Breakdown disputes by transaction channel"""
        df = self._get_combined_data()

        breakdown = {}
        for channel in df["channel"].unique():
            channel_data = df[df["channel"] == channel]
            breakdown[channel] = {
                "count": len(channel_data),
                "categories": channel_data["predicted_category"]
                .value_counts()
                .to_dict(),
                "avg_amount": round(channel_data["amount"].mean(), 2),
            }

        return self._format_results(breakdown, "breakdown_by_channel")

    def daily_summary(self, days: int = 7) -> Dict[str, Any]:
        """Daily summary of disputes over last N days"""
        df = self._get_combined_data()
        recent = filter_by_date(df, "created_at", get_date_range_filter(days))

        # Group by date
        recent["date"] = recent["created_at"].dt.date
        daily_stats = {}

        for date in recent["date"].unique():
            day_data = recent[recent["date"] == date]
            daily_stats[str(date)] = {
                "total_disputes": len(day_data),
                "categories": day_data["predicted_category"].value_counts().to_dict(),
                "total_amount": day_data["amount"].sum(),
            }

        return self._format_results(daily_stats, f"daily_summary_{days}_days")

    # ===== SPECIALIZED QUERIES =====

    def duplicate_charges_today(self) -> Dict[str, Any]:
        """Specific query: How many duplicate charges today?"""
        result = self.count_disputes_by_category("DUPLICATE_CHARGE", "today")
        result["query_type"] = "duplicate_charges_today"
        return result

    def pending_refunds(self) -> Dict[str, Any]:
        """List pending refund cases"""
        df = self._get_combined_data()
        pending = df[df["predicted_category"] == "REFUND_PENDING"]

        result = pending[
            [
                "dispute_id",
                "customer_id",
                "amount",
                "description",
                "suggested_action",
                "created_at",
            ]
        ].to_dict("records")

        return self._format_results(result, "pending_refunds")

    def top_merchants_with_disputes(self, limit: int = 10) -> Dict[str, Any]:
        """Find merchants with most disputes"""
        df = self._get_combined_data()

        # Extract merchant from txn_type or use a placeholder
        merchant_disputes = (
            df.groupby("txn_type")
            .agg({"dispute_id": "count", "amount": ["sum", "mean"]})
            .reset_index()
        )

        merchant_disputes.columns = [
            "merchant",
            "dispute_count",
            "total_amount",
            "avg_amount",
        ]
        merchant_disputes = merchant_disputes.sort_values(
            "dispute_count", ascending=False
        ).head(limit)

        result = merchant_disputes.to_dict("records")
        return self._format_results(result, "top_merchants_with_disputes")

    # ===== EXPORT FUNCTIONS =====

    def export_filtered_data(
        self, filters: Dict[str, Any], output_file: str = None
    ) -> Dict[str, Any]:
        """Export filtered dispute data"""
        df = self._get_combined_data()

        # Apply filters
        if "category" in filters:
            df = df[df["predicted_category"] == filters["category"].upper()]
        if "min_amount" in filters:
            df = df[df["amount"] >= filters["min_amount"]]
        if "action" in filters:
            df = df[df["suggested_action"] == filters["action"]]

        # Export or return
        if output_file:
            df.to_csv(output_file, index=False)
            result = {"exported_to": output_file, "record_count": len(df)}
        else:
            result = df.to_dict("records")

        return self._format_results(result, "export_filtered_data")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get overall summary statistics"""
        df = self._get_combined_data()

        stats = {
            "total_disputes": len(df),
            "total_amount": df["amount"].sum(),
            "avg_amount": round(df["amount"].mean(), 2),
            "categories": df["predicted_category"].value_counts().to_dict(),
            "resolution_actions": df["suggested_action"].value_counts().to_dict(),
            "channels": df["channel"].value_counts().to_dict(),
            "date_range": {
                "from": df["created_at"].min().strftime("%Y-%m-%d"),
                "to": df["created_at"].max().strftime("%Y-%m-%d"),
            },
        }

        return self._format_results(stats, "summary_statistics")
