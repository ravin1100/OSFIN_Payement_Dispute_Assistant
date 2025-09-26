"""
test_nlp.py
-----------
Simple test script for natural language processing functionality.
Tests the NLP processor with sample queries.
"""

import os
import sys

sys.path.append("src")

from nlp_processor import process_natural_language_query, test_nlp_processor


def test_without_api_key():
    """Test NLP processor without API key (fallback mode)"""
    print("🧪 Testing NLP Processor - Fallback Mode (No API Key)")
    print("=" * 60)

    test_queries = [
        "How many duplicate charges happened today?",
        "Show me all fraud disputes",
        "Break down disputes by type",
        "List high value disputes above 10000",
        "What's the summary of all disputes?",
        "Count fraud cases",
        "Show me pending refunds",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        try:
            result = process_natural_language_query(query)
            print(f"   → Function: {result['function_name']}")
            print(f"   → Parameters: {result['kwargs']}")
            print(f"   → Confidence: {result['confidence']}")
            print(f"   → Explanation: {result['explanation']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_with_api_key():
    """Test NLP processor with API key (full Gemini mode)"""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("\n🔑 Gemini API Key not found in environment variables")
        print(
            "💡 Set GEMINI_API_KEY environment variable to test full NLP functionality"
        )
        return

    print("\n🧪 Testing NLP Processor - Full Mode (With API Key)")
    print("=" * 60)

    test_queries = [
        "How many duplicate charges happened today?",
        "Show me fraud cases from this week",
        "I need a breakdown of all dispute types",
        "Can you list disputes above 5000 rupees?",
        "What's the overall summary?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        try:
            result = process_natural_language_query(query, api_key)
            print(f"   → Function: {result['function_name']}")
            print(f"   → Parameters: {result['kwargs']}")
            print(f"   → Confidence: {result['confidence']}")
            print(f"   → Explanation: {result['explanation']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Main test function"""
    print("🤖 Natural Language Processing Test Suite")
    print("Testing dispute analysis NLP capabilities...")

    # Test fallback mode (without API key)
    test_without_api_key()

    # Test with API key if available
    test_with_api_key()

    print("\n" + "=" * 60)
    print("✅ NLP Testing Complete!")
    print("\n💡 Usage Examples:")
    print("   # Fallback mode (keyword matching)")
    print("   python src/cli.py --natural 'show me duplicate charges'")
    print("\n   # Full NLP mode (requires API key)")
    print("   export GEMINI_API_KEY='your-api-key'")
    print("   python src/cli.py --natural 'How many fraud disputes happened today?'")


if __name__ == "__main__":
    main()
