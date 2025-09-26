"""
resolve.py
----------
Resolution suggestion script for disputes.

Input : output/classified_disputes.csv, data/transactions.csv
Output: output/resolutions.csv

Each dispute gets a suggested action and justification.
"""

import pandas as pd
import os
from datetime import datetime, timedelta


# ------------------------------
# Step 1: Load datasets
# ------------------------------
def load_data(
    disputes_file="output/classified_disputes.csv",
    transactions_file="data/transactions.csv",
):
    if not os.path.exists(disputes_file):
        raise FileNotFoundError(f"{disputes_file} not found. Run classify.py first.")
    if not os.path.exists(transactions_file):
        raise FileNotFoundError(f"{transactions_file} not found.")

    disputes = pd.read_csv(disputes_file)
    transactions = pd.read_csv(transactions_file)
    return disputes, transactions


# ------------------------------
# Step 2: Find duplicate transaction (helper)
# ------------------------------
def is_duplicate(txn_id, transactions_df, time_window=30):
    """
    Check if a transaction has a duplicate (same amount, same customer, within 30 sec).
    Returns True/False.
    """
    if txn_id not in transactions_df["txn_id"].values:
        return False

    txn = transactions_df[transactions_df["txn_id"] == txn_id].iloc[0]
    customer = txn["customer_id"]
    amount = txn["amount"]
    ts = pd.to_datetime(txn["timestamp"])

    # Find same customer & amount within +/- 30 sec
    nearby_txns = transactions_df[
        (transactions_df["customer_id"] == customer)
        & (transactions_df["amount"] == amount)
    ]
    for _, row in nearby_txns.iterrows():
        other_ts = pd.to_datetime(row["timestamp"])
        if (
            abs((ts - other_ts).total_seconds()) <= time_window
            and row["txn_id"] != txn_id
        ):
            return True
    return False


# ------------------------------
# Step 3: Suggest resolution
# ------------------------------
def suggest_resolution(row, transactions_df):
    category = row["predicted_category"]
    dispute_id = row["dispute_id"]
    txn_id = row["txn_id"]  # Now always available from classify.py
    amount = row.get("amount", 0)

    # Default action
    action, justification = "Ask for more info", "No strong rule applied."

    # Rule 1: Duplicate charge
    if category == "DUPLICATE_CHARGE":
        if txn_id and is_duplicate(txn_id, transactions_df):
            action = "Auto-refund"
            justification = "Duplicate transaction confirmed in system."
        else:
            action = "Manual review"
            justification = "Potential duplicate but not confirmed in system."

    # Rule 2: Failed transaction
    elif category == "FAILED_TRANSACTION":
        if txn_id in transactions_df["txn_id"].values:
            status = transactions_df.loc[
                transactions_df["txn_id"] == txn_id, "status"
            ].values[0]
            if status in ["FAILED", "CANCELLED"]:
                action = "Auto-refund"
                justification = (
                    f"Transaction {status.lower()} in records; refund applicable."
                )
            elif status == "PENDING":
                action = "Manual review"
                justification = "Transaction pending; needs manual verification."
            else:
                action = "Ask for more info"
                justification = (
                    "Transaction successful in records; needs clarification."
                )
        else:
            action = "Ask for more info"
            justification = "Transaction not found in system."

    # Rule 3: Fraud
    elif category == "FRAUD":
        if amount > 5000:
            action = "Escalate to bank"
            justification = "High-value fraud dispute requires bank escalation."
        elif amount > 1000:
            action = "Mark as potential fraud"
            justification = "Medium-value suspicious activity detected."
        else:
            action = "Manual review"
            justification = "Low-value fraud claim needs verification."

    # Rule 4: Refund pending
    elif category == "REFUND_PENDING":
        if txn_id in transactions_df["txn_id"].values:
            status = transactions_df.loc[
                transactions_df["txn_id"] == txn_id, "status"
            ].values[0]
            if status in ["CANCELLED", "FAILED"]:
                action = "Auto-refund"
                justification = "Transaction cancelled/failed; refund overdue."
            else:
                action = "Manual review"
                justification = "Refund process needs manual verification."
        else:
            action = "Manual review"
            justification = "Transaction not found; manual investigation needed."

    # Rule 5: Others
    elif category == "OTHERS":
        action = "Ask for more info"
        justification = "Dispute unclear, requires customer clarification."

    return {
        "dispute_id": dispute_id,
        "suggested_action": action,
        "justification": justification,
    }


# ------------------------------
# Step 4: Process all disputes
# ------------------------------
def generate_resolutions(disputes_df, transactions_df):
    results = []
    for _, row in disputes_df.iterrows():
        res = suggest_resolution(row, transactions_df)
        results.append(res)
    return pd.DataFrame(results)


# ------------------------------
# Step 5: Save results
# ------------------------------
def save_results(df, output_file="output/resolutions.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Resolutions saved to {output_file}")


# ------------------------------
# Step 6: Main pipeline
# ------------------------------
def main():
    disputes, transactions = load_data()
    resolutions_df = generate_resolutions(disputes, transactions)
    save_results(resolutions_df)


if __name__ == "__main__":
    main()
