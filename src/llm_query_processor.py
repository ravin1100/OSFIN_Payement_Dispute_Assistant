import pandas as pd
from google import genai
import json
import os
import re
from typing import Tuple, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMQueryProcessor:
    """
    Process natural language queries using Google Gemini
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self._initialize_client()
        # Use a stable model name
        self.model_name = "gemini-2.0-flash"

    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            if self.api_key:
                # The new SDK uses a client-centric approach
                self.client = genai.Client(api_key=self.api_key)
                return True
            return False
        except Exception as e:
            print(f"Warning: Could not initialize Gemini client: {e}")
            return False

    def process_query(self, query: str, df: pd.DataFrame) -> Tuple[Any, str, str]:
        """
        Process natural language query and return results

        Args:
            query: Natural language query
            df: DataFrame to query

        Returns:
            (result, explanation, code_used)
        """
        if not self.client:
            return self._fallback_query(query, df)

        try:
            # Generate pandas code using LLM
            pandas_code, explanation = self._generate_pandas_code(query, df)

            # Execute the code safely
            result = self._execute_code_safely(pandas_code, df)

            return result, explanation, pandas_code

        except Exception as e:
            print(f"LLM query failed: {e}")
            return self._fallback_query(query, df)

    def _generate_pandas_code(self, query: str, df: pd.DataFrame) -> Tuple[str, str]:
        """Generate pandas code using Gemini"""

        # Prepare data context
        columns_info = list(df.columns)
        data_types = df.dtypes.to_dict()
        sample_data = df.head(3).to_string()

        # Get unique values for categorical columns
        categorical_info = {}
        if "predicted_category" in df.columns:
            categorical_info["predicted_category"] = (
                df["predicted_category"].unique().tolist()
            )
        if "merchant" in df.columns:
            categorical_info["merchant"] = (
                df["merchant"].unique()[:10].tolist()
            )  # Top 10 merchants
        if "channel" in df.columns:
            categorical_info["channel"] = df["channel"].unique().tolist()

        # Create comprehensive prompt
        prompt = f"""
You are an expert data analyst. Convert the natural language query to pandas DataFrame operations.

DATASET INFORMATION:
- Columns: {columns_info}
- Data Types: {data_types}
- Categories: {categorical_info}

SAMPLE DATA:
{sample_data}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Generate ONLY valid pandas code that works on DataFrame 'df'
2. Handle common queries like filtering, grouping, counting, sorting
3. For amount comparisons, use numeric operations
4. For category filters, use exact category names from the data
5. Return results that can be displayed (DataFrame, Series, or simple values)

EXAMPLES:
- "Show fraud disputes" → df[df['predicted_category'] == 'FRAUD']
- "High amount disputes" → df[df['amount'] > df['amount'].quantile(0.75)]
- "Count by category" → df['predicted_category'].value_counts()
- "Merchants with most disputes" → df['merchant'].value_counts().head(10)
- "Average confidence by category" → df.groupby('predicted_category')['confidence'].mean()

RESPONSE FORMAT:
Return a JSON with:
{{"pandas_code": "your_pandas_code_here", "explanation": "explanation of what the query does"}}

Generate code for: "{query}"
"""
        # Updated API call using the client
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        response_text = response.text.strip()

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                response_data = json.loads(json_str)
                pandas_code = response_data.get("pandas_code", "")
                explanation = response_data.get("explanation", "LLM-generated query")
            else:
                # Fallback: treat entire response as code
                pandas_code = response_text
                explanation = f"Generated query for: {query}"

            return pandas_code, explanation

        except json.JSONDecodeError:
            # If JSON parsing fails, use the response as code
            pandas_code = response_text
            explanation = f"Generated query for: {query}"
            return pandas_code, explanation

    def _execute_code_safely(self, pandas_code: str, df: pd.DataFrame) -> Any:
        """Safely execute pandas code"""
        try:
            # Clean the code
            code = pandas_code.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            # Create safe execution environment
            safe_globals = {"df": df, "pd": pd}

            # Execute the code using exec for statements, and eval for expressions
            if any(
                keyword in code for keyword in ["=", "import"]
            ):  # A simple check if it's a statement
                exec(code, safe_globals)
                return safe_globals.get(
                    "result", df
                )  # Assuming result is stored in a 'result' variable
            else:
                return eval(code, safe_globals)

        except Exception as e:
            raise Exception(f"Code execution error: {e}")

    def _fallback_query(self, query: str, df: pd.DataFrame) -> Tuple[Any, str, str]:
        """Fallback query processing when LLM is not available"""
        query_lower = query.lower()

        # Simple keyword matching as fallback
        if "fraud" in query_lower:
            result = df[df["predicted_category"] == "FRAUD"]
            return (
                result,
                "Showing fraud disputes (fallback)",
                "df[df['predicted_category'] == 'FRAUD']",
            )

        elif "duplicate" in query_lower:
            result = df[df["predicted_category"] == "DUPLICATE_CHARGE"]
            return (
                result,
                "Showing duplicate charge disputes (fallback)",
                "df[df['predicted_category'] == 'DUPLICATE_CHARGE']",
            )

        elif "failed" in query_lower:
            result = df[df["predicted_category"] == "FAILED_TRANSACTION"]
            return (
                result,
                "Showing failed transaction disputes (fallback)",
                "df[df['predicted_category'] == 'FAILED_TRANSACTION']",
            )

        elif "refund" in query_lower:
            result = df[df["predicted_category"] == "REFUND_PENDING"]
            return (
                result,
                "Showing refund pending disputes (fallback)",
                "df[df['predicted_category'] == 'REFUND_PENDING']",
            )

        elif "high amount" in query_lower or "amount >" in query_lower:
            threshold = 1000
            # Try to extract number from query
            numbers = re.findall(r"\d+", query)
            if numbers:
                threshold = int(numbers)
            result = df[df["amount"] > threshold]
            return (
                result,
                f"Showing disputes with amount > {threshold} (fallback)",
                f"df[df['amount'] > {threshold}]",
            )

        elif "count" in query_lower or "how many" in query_lower:
            result = df["predicted_category"].value_counts()
            return (
                result,
                "Showing category counts (fallback)",
                "df['predicted_category'].value_counts()",
            )

        else:
            return df, "Showing all disputes (fallback - query not recognized)", "df"


# Singleton instance for use in Streamlit
_llm_processor = None


def get_llm_processor(api_key: Optional[str] = None) -> LLMQueryProcessor:
    """Get singleton LLM processor instance"""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = LLMQueryProcessor(api_key)
    return _llm_processor


def process_natural_language_query(
    query: str, df: pd.DataFrame, api_key: Optional[str] = None
) -> Tuple[Any, str, str]:
    """
    Convenience function to process natural language queries

    Returns:
        (result, explanation, code_used)
    """
    processor = get_llm_processor(api_key)
    return processor.process_query(query, df)
