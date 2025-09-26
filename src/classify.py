"""
classify.py
-----------
Rule-based dispute classification script.

Input : data/disputes.csv
Output: data/classified_disputes.csv

Each dispute is classified into one of:
- DUPLICATE_CHARGE
- FAILED_TRANSACTION
- FRAUD
- REFUND_PENDING
- OTHERS

Output columns:
dispute_id, predicted_category, confidence, explanation
"""

import pandas as pd
import os
from datetime import datetime, timedelta


# ------------------------------
# Step 1: Define classification rules
# ------------------------------
def classify_dispute_enhanced(
    dispute_row, transaction_data=None, all_transactions=None
):
    """
    dispute classification using both dispute description and transaction context.
    Returns: (category, confidence, explanation)
    """
    description = dispute_row.get("description", "")
    txn_id = dispute_row.get("txn_id", "")
    amount = dispute_row.get("amount", 0)

    desc = description.lower() if isinstance(description, str) else ""

    # Get transaction details if available
    txn_details = None
    if transaction_data is not None and txn_id:
        txn_match = transaction_data[transaction_data["txn_id"] == txn_id]
        if not txn_match.empty:
            txn_details = txn_match.iloc[0]

    # Enhanced Rule 1: Duplicate Charge Detection
    duplicate_keywords = [
        "charged twice",
        "duplicate charge",
        "double charge",
        "two debit messages",
        "duplicate transfer",
        "same merchant within minutes",
        "charged twice at",
        "duplicate upi",
        "same vpa",
        "minutes apart",
        "two upi debit",
        "got two",
        "same payment",
        "duplicate payment",
    ]

    for kw in duplicate_keywords:
        if kw in desc:
            confidence = 1.0
            explanation = f"Keyword match: '{kw}'"

            # Enhanced confidence with transaction context
            if txn_details is not None and all_transactions is not None:
                merchant = txn_details.get("merchant", "")
                txn_amount = txn_details.get("amount", 0)
                txn_time = txn_details.get("timestamp", "")

                # Check for actual duplicate transactions
                if merchant and txn_amount:
                    duplicates = find_duplicate_transactions(
                        txn_details, all_transactions
                    )
                    if len(duplicates) > 0:
                        confidence = 1.0
                        explanation += (
                            f" + Found {len(duplicates)} duplicate transaction(s)"
                        )

            return ("DUPLICATE_CHARGE", confidence, explanation)

    # Enhanced Rule 2: Failed Transaction
    failed_keywords = [
        "failed",
        "not refunded",
        "not received",
        "payment stuck",
        "pending",
    ]
    for kw in failed_keywords:
        if kw in desc:
            confidence = 0.9
            explanation = f"Keyword match: '{kw}'"

            # Enhanced with transaction status
            if txn_details is not None:
                txn_status = txn_details.get("status", "").upper()
                if txn_status in ["FAILED", "CANCELLED"]:
                    confidence = 1.0
                    explanation += f" + Transaction status: {txn_status}"

            return ("FAILED_TRANSACTION", confidence, explanation)

    # Enhanced Rule 3: Fraud Detection
    fraud_keywords = [
        "fraud",
        "unauthorized",
        "not made this payment",
        "scam",
        "did not make",
        "didn't authorize",
        "suspicious",
        "don't recognize",
    ]

    for kw in fraud_keywords:
        if kw in desc:
            confidence = 1.0
            explanation = f"Keyword match: '{kw}'"

            # Enhanced fraud detection with amount threshold
            if amount > 5000:
                confidence = 1.0
                explanation += f" + High amount: ₹{amount}"

            return ("FRAUD", confidence, explanation)

    # Enhanced Rule 4: Refund Pending
    refund_keywords = [
        "waiting for refund",
        "refund pending",
        "still not refunded",
        "refund not received",
        "still waiting",
        "refund for canceled",
    ]

    for kw in refund_keywords:
        if kw in desc:
            confidence = 0.8
            explanation = f"Keyword match: '{kw}'"

            # Enhanced with transaction status
            if txn_details is not None:
                txn_status = txn_details.get("status", "").upper()
                if txn_status == "CANCELLED":
                    confidence = 1.0
                    explanation += f" + Transaction status: {txn_status}"

            return ("REFUND_PENDING", confidence, explanation)

    # Default: Others (with contextual hints)
    confidence = 0.5
    explanation = "No strong keyword match"

    # Add contextual information for 'Others'
    if txn_details is not None:
        merchant = txn_details.get("merchant", "")
        channel = txn_details.get("channel", "")
        explanation += f" (Merchant: {merchant}, Channel: {channel})"

    return ("OTHERS", confidence, explanation)


def find_duplicate_transactions(txn_details, all_transactions):
    """
    Find potential duplicate transactions based on merchant, amount, and timestamp proximity
    """
    try:
        merchant = txn_details.get("merchant", "")
        amount = txn_details.get("amount", 0)
        timestamp_str = txn_details.get("timestamp", "")
        txn_id = txn_details.get("txn_id", "")

        if not all([merchant, amount, timestamp_str]):
            return []

        # Parse timestamp
        txn_time = pd.to_datetime(timestamp_str)

        # Find transactions with same merchant and amount
        potential_dups = all_transactions[
            (all_transactions["merchant"] == merchant)
            & (all_transactions["amount"] == amount)
            & (all_transactions["txn_id"] != txn_id)  # Exclude self
        ]

        # Check if within 5 minutes
        duplicates = []
        for _, dup_txn in potential_dups.iterrows():
            dup_time = pd.to_datetime(dup_txn["timestamp"])
            time_diff = abs((txn_time - dup_time).total_seconds())
            if time_diff <= 300:  # 5 minutes
                duplicates.append(dup_txn)

        return duplicates
    except Exception as e:
        return []


# Backward compatibility function
def classify_dispute(description: str):
    """
    Simple classification for backward compatibility
    """
    dispute_row = {"description": description, "amount": 0}
    category, confidence, explanation = classify_dispute_enhanced(dispute_row)
    return (category, confidence, explanation)


# ------------------------------
# Step 2: Load dataset
# ------------------------------
def load_disputes(file_path="data/disputes.csv"):
    """Load disputes dataset from CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please check the path.")
    return pd.read_csv(file_path)


def load_transactions(file_path="data/transactions.csv"):
    """Load transactions dataset from CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please check the path.")
    return pd.read_csv(file_path)


# ------------------------------
# Step 3: Apply classification
# ------------------------------
def classify_disputes(df, transactions_df=None):
    """Apply enhanced classification rules to all disputes"""
    results = []

    for _, row in df.iterrows():
        dispute_id = row["dispute_id"]
        description = row["description"]
        txn_id = row["txn_id"]
        amount = row.get("amount", 0)

        # Use enhanced classification with transaction context
        category, confidence, explanation = classify_dispute_enhanced(
            row, transactions_df, transactions_df
        )

        # Add merchant and channel info if available
        merchant = ""
        channel = ""
        if transactions_df is not None and txn_id:
            txn_match = transactions_df[transactions_df["txn_id"] == txn_id]
            if not txn_match.empty:
                merchant = txn_match.iloc[0].get("merchant", "")
                channel = txn_match.iloc[0].get("channel", "")

        results.append(
            {
                "dispute_id": dispute_id,
                "txn_id": txn_id,
                "amount": amount,
                "merchant": merchant,
                "channel": channel,
                "predicted_category": category,
                "confidence": confidence,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(results)


# ------------------------------
# Step 4: Save results
# ------------------------------
def save_results(df, output_path="output/classified_disputes.csv"):
    """Save classified results to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")


# ------------------------------
# Step 5: Main pipeline
# ------------------------------
def main():
    disputes_df = load_disputes()

    # Try to load transactions data
    transactions_df = None
    try:
        transactions_df = load_transactions()
        print(
            f"✅ Loaded {len(transactions_df)} transactions for enhanced classification"
        )
    except FileNotFoundError:
        print("⚠️ transactions.csv not found. Using basic classification.")

    classified_df = classify_disputes(disputes_df, transactions_df)
    save_results(classified_df)


if __name__ == "__main__":
    main()
