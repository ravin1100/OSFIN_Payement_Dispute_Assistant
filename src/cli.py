"""
cli.py
------
Main CLI interface for the AI-Powered Dispute Assistant.

Usage:
    python cli.py --query "duplicate_charges_today"
    python cli.py --query "fraud_disputes" --limit 5
    python cli.py --interactive
    python cli.py --help
"""

import argparse
import json
import sys
from typing import Dict, Any
from query_engine import QueryEngine
from data_loader import DataLoader
from nlp_processor import process_natural_language_query


class DisputeCLI:
    """Command-line interface for dispute analysis"""

    def __init__(self):
        self.query_engine = QueryEngine()
        self.data_loader = DataLoader()

    def format_output(
        self, result: Dict[str, Any], output_format: str = "table"
    ) -> str:
        """Format query results for display"""
        if "error" in result:
            return f"❌ Error: {result['error']}"

        output = []
        output.append(f"🔍 Query: {result['query_type']}")
        output.append(f"⏰ Timestamp: {result['timestamp']}")
        output.append("-" * 50)

        results = result["results"]

        if output_format == "json":
            output.append(json.dumps(results, indent=2, default=str))
        else:
            output.append(self._format_table_output(results, result["query_type"]))

        return "\n".join(output)

    def _format_table_output(self, results: Any, query_type: str) -> str:
        """Format results as table"""
        if isinstance(results, dict):
            if query_type.startswith("count_"):
                return self._format_count_results(results)
            elif query_type.startswith("breakdown_"):
                return self._format_breakdown_results(results)
            elif query_type == "summary_statistics":
                return self._format_summary_results(results)
            else:
                return self._format_dict_results(results)
        elif isinstance(results, list):
            return self._format_list_results(results)
        else:
            return str(results)

    def _format_count_results(self, results: Dict) -> str:
        """Format count query results"""
        if isinstance(results, int):
            return f"📊 Count: {results}"

        output = ["📊 Count Results:"]
        for key, value in results.items():
            output.append(f"  {key}: {value}")
        return "\n".join(output)

    def _format_breakdown_results(self, results: Dict) -> str:
        """Format breakdown analysis results"""
        output = ["📈 Breakdown Analysis:"]
        for category, stats in results.items():
            output.append(f"\n🏷️  {category}:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    output.append(f"    {key}:")
                    for sub_key, sub_value in value.items():
                        output.append(f"      {sub_key}: {sub_value}")
                else:
                    output.append(f"    {key}: {value}")
        return "\n".join(output)

    def _format_list_results(self, results: list) -> str:
        """Format list query results"""
        if not results:
            return "📝 No results found."

        output = [f"📝 Found {len(results)} results:"]
        output.append("")

        for i, item in enumerate(results, 1):
            output.append(f"{i}. Dispute ID: {item.get('dispute_id', 'N/A')}")
            if "customer_id" in item:
                output.append(f"   Customer: {item['customer_id']}")
            if "amount" in item:
                output.append(f"   Amount: ₹{item['amount']}")
            if "predicted_category" in item:
                output.append(f"   Category: {item['predicted_category']}")
            if "suggested_action" in item:
                output.append(f"   Action: {item['suggested_action']}")
            if "description" in item:
                desc = (
                    item["description"][:100] + "..."
                    if len(item["description"]) > 100
                    else item["description"]
                )
                output.append(f"   Description: {desc}")
            output.append("")

        return "\n".join(output)

    def _format_dict_results(self, results: Dict) -> str:
        """Format general dictionary results"""
        output = []
        for key, value in results.items():
            if isinstance(value, dict):
                output.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    output.append(f"  {sub_key}: {sub_value}")
            else:
                output.append(f"{key}: {value}")
        return "\n".join(output)

    def _format_summary_results(self, results: Dict) -> str:
        """Format summary statistics"""
        output = ["📊 Summary Statistics:"]
        output.append(f"Total Disputes: {results.get('total_disputes', 0)}")
        output.append(f"Total Amount: ₹{results.get('total_amount', 0):,}")
        output.append(f"Average Amount: ₹{results.get('avg_amount', 0)}")

        if "categories" in results:
            output.append("\n🏷️ By Category:")
            for cat, count in results["categories"].items():
                output.append(f"  {cat}: {count}")

        if "resolution_actions" in results:
            output.append("\n⚡ Resolution Actions:")
            for action, count in results["resolution_actions"].items():
                output.append(f"  {action}: {count}")

        return "\n".join(output)

    def handle_natural_language_query(
        self, user_query: str, api_key: str = None
    ) -> Dict[str, Any]:
        """Handle natural language queries using NLP processing"""
        try:
            # Process natural language query
            nlp_result = process_natural_language_query(user_query, api_key)

            # Extract function details
            function_name = nlp_result["function_name"]
            kwargs = nlp_result.get("kwargs", {})

            # Execute the mapped query
            if hasattr(self.query_engine, function_name):
                query_function = getattr(self.query_engine, function_name)
                result = query_function(**kwargs)

                # Add NLP metadata to result
                result["nlp_info"] = {
                    "original_query": nlp_result["original_query"],
                    "confidence": nlp_result["confidence"],
                    "explanation": nlp_result["explanation"],
                    "parsed_as": nlp_result["parsed_query_type"],
                }

                return result
            else:
                return {"error": f"Query function not found: {function_name}"}

        except Exception as e:
            return {"error": f"Natural language processing failed: {str(e)}"}

    def handle_predefined_query(self, query_name: str, **kwargs) -> Dict[str, Any]:
        """Handle predefined query commands"""
        try:
            # Map query names to methods
            query_map = {
                "duplicate_charges_today": self.query_engine.duplicate_charges_today,
                "fraud_disputes": lambda: self.query_engine.list_fraud_disputes(
                    kwargs.get("limit", 10)
                ),
                "breakdown_by_type": self.query_engine.breakdown_by_type,
                "breakdown_by_channel": self.query_engine.breakdown_by_channel,
                "unresolved_disputes": lambda: self.query_engine.list_unresolved_disputes(
                    kwargs.get("limit", 20)
                ),
                "pending_refunds": self.query_engine.pending_refunds,
                "summary_stats": self.query_engine.get_summary_stats,
                "high_value_disputes": lambda: self.query_engine.count_high_value_disputes(
                    kwargs.get("threshold", 5000)
                ),
                "daily_summary": lambda: self.query_engine.daily_summary(
                    kwargs.get("days", 7)
                ),
            }

            if query_name not in query_map:
                available_queries = list(query_map.keys())
                return {
                    "error": f"Unknown query: {query_name}. Available: {', '.join(available_queries)}"
                }

            return query_map[query_name]()

        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}"}

    def interactive_mode(self):
        """Run interactive CLI mode"""
        print("🤖 AI-Powered Dispute Assistant - Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n💬 Query> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "help":
                    self.show_help()
                elif user_input.lower() == "stats":
                    result = self.query_engine.get_summary_stats()
                    print(self.format_output(result))
                elif user_input.lower().startswith("duplicate"):
                    result = self.query_engine.duplicate_charges_today()
                    print(self.format_output(result))
                elif user_input.lower().startswith("fraud"):
                    result = self.query_engine.list_fraud_disputes()
                    print(self.format_output(result))
                elif user_input.lower().startswith("breakdown"):
                    result = self.query_engine.breakdown_by_type()
                    print(self.format_output(result))
                elif user_input.lower().startswith("unresolved"):
                    result = self.query_engine.list_unresolved_disputes()
                    print(self.format_output(result))
                else:
                    # Try natural language processing
                    print("🧠 Processing natural language query...")
                    result = self.handle_natural_language_query(user_input)
                    print(self.format_output(result))

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_help(self):
        """Show help information"""
        help_text = """
🤖 AI-Powered Dispute Assistant Commands:

🧠 NATURAL LANGUAGE QUERIES:
  "How many duplicate charges today?"
  "Show me fraud disputes"
  "Break down disputes by type" 
  "List high value disputes above 10000"
  "What's the summary?"
  "Count fraud cases from last week"

📊 PREDEFINED QUERIES:
  duplicate_charges_today    - Count duplicate charges today
  fraud_disputes            - List fraud disputes  
  breakdown_by_type         - Breakdown disputes by category
  breakdown_by_channel      - Breakdown by transaction channel
  unresolved_disputes       - List disputes needing attention
  pending_refunds          - List pending refund cases
  summary_stats            - Overall statistics
  high_value_disputes      - High-value dispute count
  daily_summary            - Daily summary (last 7 days)

💬 INTERACTIVE COMMANDS:
  duplicate                - Duplicate charges today
  fraud                    - List fraud disputes
  breakdown                - Breakdown by type  
  unresolved               - Unresolved disputes
  stats                    - Summary statistics
  help                     - Show this help
  quit/exit/q              - Exit interactive mode

📝 COMMAND LINE USAGE:
  # Natural Language (requires Gemini API key)
  python cli.py --natural "How many fraud cases today?"
  python cli.py --natural "Show me duplicate charges" --api-key YOUR_API_KEY
  
  # Predefined Queries
  python cli.py --query "duplicate_charges_today"
  python cli.py --query "fraud_disputes" --limit 5
  python cli.py --query "high_value_disputes" --threshold 10000
  python cli.py --interactive
  python cli.py --format json --query "summary_stats"
"""
        print(help_text)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Dispute Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--query", "-q", help="Predefined query to execute")
    parser.add_argument(
        "--natural",
        "-n",
        help='Natural language query (e.g., "How many fraud cases today?")',
    )
    parser.add_argument(
        "--api-key", help="Gemini API key for natural language processing"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive mode"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Limit number of results (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=5000,
        help="Amount threshold for high-value queries (default: 5000)",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        help="Number of days for daily summary (default: 7)",
    )

    args = parser.parse_args()

    # Check if data files exist
    cli = DisputeCLI()
    try:
        cli.data_loader.get_data_stats()
    except FileNotFoundError:
        print("❌ Data files not found!")
        print("💡 Please run the pipeline first: python src/pipeline.py")
        sys.exit(1)

    # Handle different modes
    if args.interactive:
        cli.interactive_mode()
    elif args.query:
        kwargs = {"limit": args.limit, "threshold": args.threshold, "days": args.days}
        result = cli.handle_predefined_query(args.query, **kwargs)
        print(cli.format_output(result, args.format))
    elif args.natural:
        print(f"🧠 Processing: '{args.natural}'")
        result = cli.handle_natural_language_query(args.natural, args.api_key)
        print(cli.format_output(result, args.format))
    else:
        print("🤖 AI-Powered Dispute Assistant")
        print(
            "Use --query for predefined queries, --natural for natural language, or --interactive for interactive mode"
        )
        print("Examples:")
        print('  python cli.py --natural "How many duplicate charges today?"')
        print('  python cli.py --query "summary_stats"')
        print("Use --help for more information")

        # Show quick stats
        result = cli.query_engine.get_summary_stats()
        print("\n" + cli.format_output(result))


if __name__ == "__main__":
    main()
