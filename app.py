"""
Streamlit Web Interface for OSFIN Dispute Classification System
==============================================================

A simple, user-friendly interface for uploading dispute data,
classifying disputes, viewing results, and querying the data.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import sys

# Add src to path for imports
sys.path.append("src")
from classify import classify_disputes, classify_dispute

# Page config
st.set_page_config(
    page_title="OSFIN Dispute Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "uploaded_transactions" not in st.session_state:
    st.session_state.uploaded_transactions = None
if "classified_data" not in st.session_state:
    st.session_state.classified_data = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def main():
    # Header
    st.title("🏦 OSFIN Dispute Classifier")
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Upload", "⚡ Classify", "📊 Results", "💬 Query"]
    )

    with tab1:
        upload_tab()

    with tab2:
        classify_tab()

    with tab3:
        results_tab()

    with tab4:
        query_tab()


def upload_tab():
    """Tab 1: Upload Data"""
    st.header("📤 Upload Data Files")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Disputes File (Required)")
        # Disputes file uploader
        uploaded_disputes = st.file_uploader(
            "Choose disputes CSV file",
            type=["csv"],
            help="Upload disputes CSV: dispute_id, description, txn_id, amount",
            key="disputes_file",
        )

    with col2:
        st.subheader("💳 Transactions File (Optional)")
        # Transactions file uploader
        uploaded_transactions = st.file_uploader(
            "Choose transactions CSV file",
            type=["csv"],
            help="Upload transactions CSV: txn_id, amount, merchant, timestamp, status",
            key="transactions_file",
        )

    # Process disputes file
    if uploaded_disputes is not None:
        try:
            # Read disputes file
            disputes_df = pd.read_csv(uploaded_disputes)

            # Validate required columns
            required_cols = ["dispute_id", "description", "txn_id"]
            missing_cols = [
                col for col in required_cols if col not in disputes_df.columns
            ]

            if missing_cols:
                st.error(
                    f"❌ Missing required columns in disputes file: {', '.join(missing_cols)}"
                )
                st.info(
                    "Required columns: dispute_id, description, txn_id, amount (optional)"
                )
            else:
                # Add amount column if missing
                if "amount" not in disputes_df.columns:
                    disputes_df["amount"] = 0
                    st.warning(
                        "⚠️ 'amount' column not found in disputes. Setting all amounts to 0."
                    )

                # Store in session state
                st.session_state.uploaded_data = disputes_df

                col1, col2 = st.columns(2)
                with col1:
                    st.success("✅ Disputes file uploaded!")
                    st.metric("Total Disputes", len(disputes_df))
                with col2:
                    st.metric("Disputes Columns", len(disputes_df.columns))

        except Exception as e:
            st.error(f"❌ Error reading disputes file: {str(e)}")

    # Process transactions file
    if uploaded_transactions is not None:
        try:
            # Read transactions file
            transactions_df = pd.read_csv(uploaded_transactions)

            # Validate transaction columns
            txn_required_cols = ["txn_id", "amount", "merchant", "timestamp"]
            missing_txn_cols = [
                col for col in txn_required_cols if col not in transactions_df.columns
            ]

            if missing_txn_cols:
                st.warning(
                    f"⚠️ Missing optional columns in transactions: {', '.join(missing_txn_cols)}"
                )
                st.info(
                    "Recommended columns: txn_id, amount, merchant, timestamp, status, channel"
                )

            # Store in session state
            st.session_state.uploaded_transactions = transactions_df

            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ Transactions file uploaded!")
                st.metric("Total Transactions", len(transactions_df))
            with col2:
                st.metric("Transaction Columns", len(transactions_df.columns))

        except Exception as e:
            st.error(f"❌ Error reading transactions file: {str(e)}")

    # Show preview if both files loaded
    if st.session_state.uploaded_data is not None:
        st.markdown("---")
        st.subheader("📋 Disputes Data Preview")
        st.dataframe(st.session_state.uploaded_data.head(), width="stretch")

        if st.session_state.uploaded_transactions is not None:
            st.subheader("💳 Transactions Data Preview")
            st.dataframe(st.session_state.uploaded_transactions.head(), width="stretch")
            st.info(
                "🎯 Enhanced classification will use both datasets for better accuracy!"
            )
        else:
            st.info(
                "💡 Upload transactions file for enhanced classification with merchant and timestamp analysis"
            )

    # Show upload instructions if no files
    if uploaded_disputes is None and uploaded_transactions is None:
        st.info("👆 Please upload at least the disputes CSV file to get started")

        # Show sample format
        st.subheader("📄 Expected Format")
        sample_data = pd.DataFrame(
            {
                "dispute_id": ["D001", "D002", "D003"],
                "description": [
                    "I was charged twice for the same transaction",
                    "Payment failed but amount was debited",
                    "Unauthorized transaction on my account",
                ],
                "txn_id": ["T001", "T002", "T003"],
                "amount": [1500, 2000, 5000],
            }
        )
        st.dataframe(sample_data, width="stretch")


def classify_tab():
    """Tab 2: Classify Disputes"""
    st.header("⚡ Classify Disputes")

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Upload tab")
        return

    df = st.session_state.uploaded_data

    # Show data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Disputes to Classify", len(df))
    with col2:
        if st.session_state.classified_data is not None:
            st.metric("Already Classified", len(st.session_state.classified_data))
        else:
            st.metric("Already Classified", 0)
    with col3:
        st.metric("Categories", 5)
    with col4:
        enhanced = "Yes" if st.session_state.uploaded_transactions is not None else "No"
        st.metric("Enhanced Mode", enhanced)

    st.markdown("---")

    # Classification button
    if st.button("🚀 Start Classification", type="primary", width="stretch"):

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Classify disputes using enhanced method
        try:
            # Use the enhanced classify_disputes function
            transactions_df = st.session_state.uploaded_transactions

            if transactions_df is not None:
                status_text.text(
                    "Using enhanced classification with transaction context..."
                )
            else:
                status_text.text("Using basic classification (no transaction data)...")

            # Use the enhanced classification function
            classified_df = classify_disputes(df, transactions_df)

            # Simulate progress for user feedback
            total = len(df)
            for i in range(total):
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"Classifying dispute {i+1} of {total}...")
                # Small delay for visual feedback
                import time

                time.sleep(0.1)

            # Store results
            st.session_state.classified_data = classified_df

            # Show completion
            progress_bar.progress(1.0)
            status_text.text("✅ Classification completed!")

            st.success("🎉 All disputes have been classified successfully!")

            # Show quick summary
            st.subheader("📊 Quick Summary")
            category_counts = classified_df["predicted_category"].value_counts()

            cols = st.columns(5)
            categories = [
                "DUPLICATE_CHARGE",
                "FAILED_TRANSACTION",
                "FRAUD",
                "REFUND_PENDING",
                "OTHERS",
            ]
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

            for i, cat in enumerate(categories):
                with cols[i]:
                    count = category_counts.get(cat, 0)
                    st.metric(cat.replace("_", " ").title(), count)

        except Exception as e:
            st.error(f"❌ Classification failed: {str(e)}")

    # Show existing results if available
    if st.session_state.classified_data is not None:
        st.markdown("---")
        st.info(
            "✅ Classification data is ready! Check the Results tab to view details."
        )


def results_tab():
    """Tab 3: View Results"""
    st.header("📊 Classification Results")

    if st.session_state.classified_data is None:
        st.warning(
            "⚠️ No classification results available. Please classify disputes first."
        )
        return

    df = st.session_state.classified_data

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Disputes", len(df))
    with col2:
        avg_confidence = df["confidence"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    with col3:
        high_conf = len(df[df["confidence"] >= 0.8])
        st.metric("High Confidence", high_conf)
    with col4:
        categories = df["predicted_category"].nunique()
        st.metric("Categories Found", categories)

    st.markdown("---")

    # Category distribution chart
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🥧 Category Distribution")
        category_counts = df["predicted_category"].value_counts()

        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution of Dispute Categories",
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col2:
        st.subheader("📈 Confidence Scores")
        fig_hist = px.histogram(
            df, x="confidence", nbins=20, title="Distribution of Confidence Scores"
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("---")

    # Results table
    st.subheader("📋 Detailed Results")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.selectbox(
            "Filter by Category", ["All"] + list(df["predicted_category"].unique())
        )
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.1
        )

    # Apply filters
    filtered_df = df.copy()
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["predicted_category"] == category_filter]
    filtered_df = filtered_df[filtered_df["confidence"] >= min_confidence]

    # Display filtered results
    display_columns = ["dispute_id", "predicted_category", "confidence", "amount"]

    # Add merchant and channel if available
    if "merchant" in filtered_df.columns and filtered_df["merchant"].notna().any():
        display_columns.append("merchant")
    if "channel" in filtered_df.columns and filtered_df["channel"].notna().any():
        display_columns.append("channel")

    display_columns.append("explanation")

    st.dataframe(
        filtered_df[display_columns],
        width="stretch",
    )

    # Download button
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Download full results
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_buffer,
            file_name="classified_disputes.csv",
            mime="text/csv",
            width="stretch",
        )

    with col2:
        # Download filtered results
        if len(filtered_df) < len(df):
            csv_buffer_filtered = BytesIO()
            filtered_df.to_csv(csv_buffer_filtered, index=False)
            csv_buffer_filtered.seek(0)

            st.download_button(
                label="📥 Download Filtered Results (CSV)",
                data=csv_buffer_filtered,
                file_name="filtered_disputes.csv",
                mime="text/csv",
                width="stretch",
            )


def query_tab():
    """Tab 4: Query Interface"""
    st.header("💬 Query Your Data")

    if st.session_state.classified_data is None:
        st.warning("⚠️ No data available for querying. Please classify disputes first.")
        return

    df = st.session_state.classified_data

    # Quick action buttons
    st.subheader("🚀 Quick Queries")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Show Fraud Cases", width="stretch"):
            process_query("Show all fraud disputes", df)

    with col2:
        if st.button("High Amount Disputes", width="stretch"):
            process_query("Show disputes with amount > 1000", df)

    with col3:
        if st.button("Low Confidence", width="stretch"):
            process_query("Show disputes with confidence < 0.7", df)

    with col4:
        if st.button("Duplicate Charges", width="stretch"):
            process_query("Show all duplicate charge disputes", df)

    st.markdown("---")

    # Custom query input
    st.subheader("✍️ Custom Query")

    # Query examples
    with st.expander("💡 Query Examples"):
        st.markdown(
            """
        **Try these example queries:**
        - Show me all fraud disputes
        - How many duplicate charges are there?
        - Find disputes with amount greater than 2000
        - Show REFUND_PENDING cases
        - List disputes with low confidence
        - Show me the highest amount dispute
        """
        )

    # Query input
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., Show me all fraud disputes with amount > 1000",
    )

    if st.button("🔍 Execute Query", type="primary"):
        if query.strip():
            process_query(query, df)
        else:
            st.warning("⚠️ Please enter a query")

    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📜 Recent Queries")

        for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:])):
            if st.button(f"🔄 {hist_query}", key=f"hist_{i}"):
                process_query(hist_query, df)


def process_query(query: str, df: pd.DataFrame):
    """Process natural language query and show results"""

    # Add to history
    if query not in st.session_state.query_history:
        st.session_state.query_history.append(query)

    # Display query
    st.markdown(f"**💬 Query:** {query}")

    try:
        # Simple query processing
        query_lower = query.lower()
        filtered_df = df.copy()

        # Category filters
        if "fraud" in query_lower:
            filtered_df = filtered_df[filtered_df["predicted_category"] == "FRAUD"]
        elif "duplicate" in query_lower:
            filtered_df = filtered_df[
                filtered_df["predicted_category"] == "DUPLICATE_CHARGE"
            ]
        elif "failed" in query_lower:
            filtered_df = filtered_df[
                filtered_df["predicted_category"] == "FAILED_TRANSACTION"
            ]
        elif "refund" in query_lower:
            filtered_df = filtered_df[
                filtered_df["predicted_category"] == "REFUND_PENDING"
            ]

        # Amount filters
        if "amount >" in query_lower or "amount greater than" in query_lower:
            import re

            numbers = re.findall(r"\d+", query)
            if numbers:
                threshold = int(numbers[0])
                filtered_df = filtered_df[filtered_df["amount"] > threshold]

        # Confidence filters
        if "low confidence" in query_lower or "confidence <" in query_lower:
            filtered_df = filtered_df[filtered_df["confidence"] < 0.7]
        elif "high confidence" in query_lower:
            filtered_df = filtered_df[filtered_df["confidence"] >= 0.8]

        # Show results
        if len(filtered_df) == 0:
            st.warning("🔍 No results found for your query")
        else:
            st.success(f"📊 Found {len(filtered_df)} results:")

            # Summary if it's a count query
            if "how many" in query_lower or "count" in query_lower:
                st.metric("Count", len(filtered_df))

            # Show results table
            display_columns = [
                "dispute_id",
                "predicted_category",
                "confidence",
                "amount",
            ]

            # Add merchant and channel if available
            if (
                "merchant" in filtered_df.columns
                and filtered_df["merchant"].notna().any()
            ):
                display_columns.append("merchant")
            if (
                "channel" in filtered_df.columns
                and filtered_df["channel"].notna().any()
            ):
                display_columns.append("channel")

            display_columns.append("explanation")

            st.dataframe(
                filtered_df[display_columns],
                width="stretch",
            )

            # Download filtered results
            if len(filtered_df) > 0:
                csv_buffer = BytesIO()
                filtered_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                st.download_button(
                    label="📥 Export Query Results",
                    data=csv_buffer,
                    file_name=f"query_results_{len(filtered_df)}_disputes.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        st.info("💡 Try simpler queries like 'show fraud disputes' or 'amount > 1000'")


if __name__ == "__main__":
    main()
