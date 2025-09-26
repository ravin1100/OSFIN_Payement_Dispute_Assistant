"""
pipeline.py
-----------
Runs the full workflow:
1. Classify disputes (Task 1)
2. Suggest resolutions (Task 2)

Inputs :
- data/disputes.csv
- data/transactions.csv

Outputs:
- output/classified_disputes.csv
- output/resolutions.csv
"""

import os
import pandas as pd
from classify import classify_disputes, load_disputes, save_results as save_classified
from resolve import (
    generate_resolutions,
    load_data as load_classified_and_txns,
    save_results as save_resolutions,
)


def main():
    print("🚀 Starting AI-Powered Dispute Assistant Pipeline...")

    # -------------------------------
    # Step 1: Classification
    # -------------------------------
    print("🔹 Step 1: Classifying disputes...")
    disputes_df = load_disputes("data/disputes.csv")

    # Load transactions for enhanced classification
    try:
        from classify import load_transactions

        transactions_df = load_transactions("data/transactions.csv")
        print(
            f"✅ Loaded {len(transactions_df)} transactions for enhanced classification"
        )
    except FileNotFoundError:
        transactions_df = None
        print("⚠️ transactions.csv not found. Using basic classification.")

    classified_df = classify_disputes(disputes_df, transactions_df)
    save_classified(classified_df, "output/classified_disputes.csv")

    # -------------------------------
    # Step 2: Resolution Suggestion
    # -------------------------------
    print("🔹 Step 2: Suggesting resolutions...")
    classified_df, transactions_df = load_classified_and_txns(
        "output/classified_disputes.csv", "data/transactions.csv"
    )
    resolutions_df = generate_resolutions(classified_df, transactions_df)
    save_resolutions(resolutions_df, "output/resolutions.csv")

    print("✅ Pipeline completed successfully!")
    print("📂 Outputs saved in 'output/' folder:")
    print("   - classified_disputes.csv")
    print("   - resolutions.csv")


if __name__ == "__main__":
    main()
