import json
import re
from typing import Dict, Any, Optional, List
import google.generativeai as genai
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_available_queries() -> Dict[str, str]:
    """Return available query types and their descriptions"""
    return {
        "duplicate_charges_today": "Count duplicate charges that occurred today",
        "fraud_disputes": "List fraud disputes with details",
        "breakdown_by_type": "Breakdown disputes by category (DUPLICATE_CHARGE, FRAUD, etc.)",
        "breakdown_by_channel": "Breakdown disputes by transaction channel (Mobile, Web, POS, etc.)",
        "unresolved_disputes": "List disputes that need manual attention (not auto-refund)",
        "pending_refunds": "List pending refund cases",
        "summary_stats": "Overall summary statistics of all disputes",
        "high_value_disputes": "Count disputes above a certain amount threshold",
        "daily_summary": "Daily summary of disputes over specified number of days",
        "count_by_category": "Count disputes by specific category",
        "count_by_status": "Count disputes by resolution status",
        "list_by_category": "List disputes of specific category",
    }


def create_query_analysis_prompt(user_query: str) -> str:
    """Create a prompt for Gemini to analyze the user query"""

    available_queries = get_available_queries()

    prompt = f"""
You are a dispute analysis assistant. Convert the user's natural language query into a structured command.

Available Query Types:
{json.dumps(available_queries, indent=2)}

Available Categories: DUPLICATE_CHARGE, FRAUD, FAILED_TRANSACTION, REFUND_PENDING, OTHERS
Available Channels: Mobile, Web, POS, QR
Available Actions: Auto-refund, Manual review, Escalate to bank, Mark as potential fraud, Ask for more info

User Query: "{user_query}"

Analyze the query and return ONLY a JSON response with this structure:
{{
    "query_type": "most_appropriate_query_from_available_list",
    "parameters": {{
        "category": "category_if_specified",
        "limit": number_if_specified_or_10,
        "threshold": amount_threshold_if_specified_or_5000,
        "days": number_of_days_if_specified_or_7,
        "date_filter": "today/week/null"
    }},
    "confidence": 0.0_to_1.0,
    "explanation": "brief_explanation_of_mapping"
}}

Examples:
- "How many duplicate charges today?" → query_type: "duplicate_charges_today"
- "Show me fraud cases" → query_type: "fraud_disputes"
- "Break down by type" → query_type: "breakdown_by_type"
- "List high value disputes above 10000" → query_type: "high_value_disputes", parameters: {{"threshold": 10000}}
- "Summary of all disputes" → query_type: "summary_stats"

Return ONLY the JSON, no additional text.
"""

    return prompt


def parse_natural_query(
    user_query: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse natural language query using Gemini 2.0 Flash

    Args:
        user_query: Natural language query from user
        api_key: Optional Gemini API key

    Returns:
        Dict with query_type, parameters, confidence, and explanation
    """

    try:

        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Create prompt
        prompt = create_query_analysis_prompt(user_query)

        # Generate response
        response = model.generate_content(prompt)

        # Parse JSON response
        response_text = response.text.strip()

        # Clean up response (remove any markdown formatting)
        if response_text.startswith("```json"):
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )

        parsed_response = json.loads(response_text)

        # Validate response structure
        if not validate_parsed_response(parsed_response):
            return create_fallback_response(user_query)

        return parsed_response

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse Gemini response as JSON: {e}")
        return create_fallback_response(user_query)

    except Exception as e:
        print(f"❌ Error processing natural language query: {e}")
        return create_fallback_response(user_query)


def validate_parsed_response(response: Dict[str, Any]) -> bool:
    """Validate that the parsed response has required fields"""
    required_fields = ["query_type", "parameters", "confidence", "explanation"]

    if not all(field in response for field in required_fields):
        return False

    # Check if query_type is valid
    available_queries = get_available_queries()
    if response["query_type"] not in available_queries:
        return False

    # Check confidence is a number between 0 and 1
    confidence = response.get("confidence", 0)
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        return False

    return True


def create_fallback_response(user_query: str) -> Dict[str, Any]:
    """Create a fallback response when NLP parsing fails"""

    # Simple keyword-based fallback
    query_lower = user_query.lower()

    if "duplicate" in query_lower and "today" in query_lower:
        return {
            "query_type": "duplicate_charges_today",
            "parameters": {},
            "confidence": 0.7,
            "explanation": "Fallback: detected duplicate + today keywords",
        }
    elif "fraud" in query_lower:
        return {
            "query_type": "fraud_disputes",
            "parameters": {"limit": 10},
            "confidence": 0.6,
            "explanation": "Fallback: detected fraud keyword",
        }
    elif "breakdown" in query_lower or "break down" in query_lower:
        return {
            "query_type": "breakdown_by_type",
            "parameters": {},
            "confidence": 0.6,
            "explanation": "Fallback: detected breakdown keyword",
        }
    elif "summary" in query_lower or "stats" in query_lower:
        return {
            "query_type": "summary_stats",
            "parameters": {},
            "confidence": 0.6,
            "explanation": "Fallback: detected summary/stats keyword",
        }
    else:
        return {
            "query_type": "summary_stats",
            "parameters": {},
            "confidence": 0.3,
            "explanation": "Fallback: no clear match, defaulting to summary",
        }


def map_nlp_to_query_engine(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map NLP-parsed query to query engine function call

    Args:
        parsed_query: Output from parse_natural_query

    Returns:
        Dict with function_name and kwargs for query engine
    """

    query_type = parsed_query["query_type"]
    parameters = parsed_query.get("parameters", {})

    # Map query types to function calls
    query_mapping = {
        "duplicate_charges_today": {
            "function": "duplicate_charges_today",
            "kwargs": {},
        },
        "fraud_disputes": {
            "function": "list_fraud_disputes",
            "kwargs": {"limit": parameters.get("limit", 10)},
        },
        "breakdown_by_type": {"function": "breakdown_by_type", "kwargs": {}},
        "breakdown_by_channel": {"function": "breakdown_by_channel", "kwargs": {}},
        "unresolved_disputes": {
            "function": "list_unresolved_disputes",
            "kwargs": {"limit": parameters.get("limit", 20)},
        },
        "pending_refunds": {"function": "pending_refunds", "kwargs": {}},
        "summary_stats": {"function": "get_summary_stats", "kwargs": {}},
        "high_value_disputes": {
            "function": "count_high_value_disputes",
            "kwargs": {"threshold": parameters.get("threshold", 5000)},
        },
        "daily_summary": {
            "function": "daily_summary",
            "kwargs": {"days": parameters.get("days", 7)},
        },
        "count_by_category": {
            "function": "count_disputes_by_category",
            "kwargs": {
                "category": parameters.get("category"),
                "date_filter": parameters.get("date_filter"),
            },
        },
        "count_by_status": {
            "function": "count_by_status",
            "kwargs": {"status": parameters.get("status")},
        },
        "list_by_category": {
            "function": "list_disputes_by_category",
            "kwargs": {
                "category": parameters.get("category", "FRAUD"),
                "limit": parameters.get("limit", 10),
            },
        },
    }

    if query_type in query_mapping:
        mapping = query_mapping[query_type]
        return {
            "function_name": mapping["function"],
            "kwargs": {k: v for k, v in mapping["kwargs"].items() if v is not None},
            "confidence": parsed_query.get("confidence", 0.5),
            "explanation": parsed_query.get("explanation", ""),
        }
    else:
        # Fallback to summary stats
        return {
            "function_name": "get_summary_stats",
            "kwargs": {},
            "confidence": 0.3,
            "explanation": f"Unknown query type: {query_type}, defaulting to summary",
        }


def process_natural_language_query(
    user_query: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline: Natural language → Gemini analysis → Query engine mapping

    Args:
        user_query: Natural language query from user
        api_key: Optional Gemini API key

    Returns:
        Dict ready for query engine execution
    """

    # Step 1: Parse natural language with Gemini
    parsed_query = parse_natural_query(user_query, api_key)

    # Step 2: Map to query engine function
    query_mapping = map_nlp_to_query_engine(parsed_query)

    # Step 3: Add metadata
    query_mapping.update(
        {
            "original_query": user_query,
            "parsed_query_type": parsed_query["query_type"],
            "processing_timestamp": datetime.now().isoformat(),
        }
    )

    return query_mapping


def extract_keywords_fallback(user_query: str) -> List[str]:
    """Extract keywords for fallback processing when Gemini is unavailable"""

    # Common dispute-related keywords
    keywords = []
    query_lower = user_query.lower()

    # Categories
    if re.search(r"\bduplicate|double|twice\b", query_lower):
        keywords.append("duplicate")
    if re.search(r"\bfraud|unauthorized|suspicious\b", query_lower):
        keywords.append("fraud")
    if re.search(r"\bfailed|failure|error\b", query_lower):
        keywords.append("failed")
    if re.search(r"\brefund|return|pending\b", query_lower):
        keywords.append("refund")

    # Actions
    if re.search(r"\bcount|how many|number\b", query_lower):
        keywords.append("count")
    if re.search(r"\blist|show|display\b", query_lower):
        keywords.append("list")
    if re.search(r"\bbreakdown|break down|analysis|analyze\b", query_lower):
        keywords.append("breakdown")
    if re.search(r"\bsummary|stats|statistics|overview\b", query_lower):
        keywords.append("summary")

    # Time
    if re.search(r"\btoday|now\b", query_lower):
        keywords.append("today")
    if re.search(r"\bweek|weekly\b", query_lower):
        keywords.append("week")
    if re.search(r"\bhigh|large|big\b", query_lower):
        keywords.append("high_value")

    return keywords


# Test function for development
def test_nlp_processor():
    """Test the NLP processor with sample queries"""

    test_queries = [
        "How many duplicate charges happened today?",
        "Show me all fraud disputes",
        "Break down disputes by type",
        "List high value disputes above 10000",
        "What's the summary of all disputes?",
        "Count fraud cases from last week",
        "Show me pending refunds",
    ]

    print("🧪 Testing NLP Processor...")

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        try:
            result = process_natural_language_query(query)
            print(f"✅ Function: {result['function_name']}")
            print(f"📊 Parameters: {result['kwargs']}")
            print(f"🎯 Confidence: {result['confidence']}")
            print(f"💡 Explanation: {result['explanation']}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_nlp_processor()
