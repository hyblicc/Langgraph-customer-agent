"""
LangGraph Multi-Agent Customer Service System Demo
"""

import asyncio
import logging
import sys
import argparse
from typing import Dict, Any

from src.workflow.state_schema import AgentState, IntentType, create_empty_state
from src.workflow.graph_builder import get_compiled_graph
from src.knowledge.rag_retriever import get_retriever
from langchain.schema import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_knowledge_base():
    """Initialize knowledge base with sample documents"""
    logger.info("Setting up knowledge base...")
    
    retriever = get_retriever()
    
    # Sample documents for customer service knowledge base
    sample_docs = [
        Document(
            page_content="""Return Policy: We accept returns within 30 days of purchase for most items.
Items must be in original condition with all packaging and documentation. Refunds are processed 
within 5-7 business days after we receive the returned item.""",
            metadata={"source": "Return Policy", "category": "policies"}
        ),
        Document(
            page_content="""Shipping Information: Standard shipping takes 5-7 business days. 
Express shipping is available for 2-3 business days delivery. Free shipping on orders over $50. 
International shipping available to select countries.""",
            metadata={"source": "Shipping Info", "category": "policies"}
        ),
        Document(
            page_content="""Product Warranty: All our products come with a 1-year manufacturer's warranty 
covering defects in materials and workmanship. Extended warranties are available for purchase at time of ordering.""",
            metadata={"source": "Warranty Info", "category": "policies"}
        ),
        Document(
            page_content="""Customer Support Hours: Our customer support team is available Monday-Friday 
9AM-6PM EST. Email support is available 24/7 with responses within 24 hours. Live chat available during business hours.""",
            metadata={"source": "Support Hours", "category": "contact"}
        ),
        Document(
            page_content="""Payment Methods: We accept all major credit cards (Visa, Mastercard, American Express),
PayPal, and Apple Pay. Payment is processed securely using industry-standard encryption.""",
            metadata={"source": "Payment Methods", "category": "billing"}
        ),
    ]
    
    retriever.add_knowledge(sample_docs)
    logger.info(f"Knowledge base initialized with {len(sample_docs)} documents")


async def process_query(query: str, user_id: str = "demo_user") -> Dict[str, Any]:
    """
    Process a customer query through the multi-agent system
    
    Args:
        query: User query
        user_id: User identifier
    
    Returns:
        Agent response and metadata
    """
    logger.info(f"Processing query: {query}")
    
    try:
        # Get compiled graph
        graph = get_compiled_graph()
        
        # Create initial state
        initial_state = create_empty_state(user_id)
        initial_state["query"] = query
        
        # Invoke graph with configuration
        config = {"configurable": {"thread_id": f"{user_id}_{query[:20]}"}}
        result = graph.invoke(initial_state, config)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            "final_answer": f"Error processing your request: {str(e)}",
            "error_message": str(e),
            "confidence_score": 0.0,
        }


def demo_mode():
    """Run demo with predefined test queries"""
    print("\n" + "="*70)
    print("LangGraph Multi-Agent Customer Service System - Demo Mode")
    print("="*70 + "\n")
    
    # Setup knowledge base
    setup_knowledge_base()
    
    # Test queries
    test_queries = [
        "What is your return policy?",
        "How long does standard shipping take?",
        "Can you check my order status?",
        "Do you have a warranty on products?",
        "What payment methods do you accept?",
    ]
    
    # Process each query
    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {idx}: {query}")
        print(f"{'─'*70}")
        
        try:
            # Process query (note: using sync wrapper)
            # In production, use proper async handling
            import asyncio
            result = asyncio.run(process_query(query))
            
            # Display results
            print(f"\n✓ Intent Recognized:")
            print(f"   {result.get('intent', 'Unknown').value if hasattr(result.get('intent', 'Unknown'), 'value') else result.get('intent', 'Unknown')}")
            print(f"   Confidence: {result.get('intent_confidence', 0):.0%}")
            
            if result.get('knowledge_sources'):
                print(f"\n✓ Knowledge Retrieved:")
                for source in result['knowledge_sources']:
                    print(f"   - {source.get('source', 'Unknown')} (relevance: {source.get('score', 0):.0%})")
            
            print(f"\n✓ Response:")
            print(f"   {result.get('final_answer', 'No response generated')}")
            
            print(f"\n✓ Confidence: {result.get('confidence_score', 0):.0%}")
            
            if result.get('answer_sources'):
                print(f"   Sources: {', '.join(result['answer_sources'])}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            logger.error(f"Error processing query: {e}", exc_info=True)
    
    print(f"\n{'='*70}")
    print("Demo completed successfully!")
    print(f"{'='*70}\n")


def interactive_mode():
    """Interactive mode for real-time queries"""
    print("\n" + "="*70)
    print("LangGraph Multi-Agent Customer Service System - Interactive Mode")
    print("="*70)
    print("\nEnter your queries (type 'quit', 'exit', or 'q' to exit):\n")
    
    # Setup knowledge base
    setup_knowledge_base()
    
    try:
        while True:
            user_query = input("You: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using our customer service system. Goodbye!\n")
                break
            
            if not user_query:
                continue
            
            try:
                result = asyncio.run(process_query(user_query))
                
                print(f"\nAssistant: {result.get('final_answer', 'No response generated')}")
                print(f"(Confidence: {result.get('confidence_score', 0):.0%})\n")
            
            except Exception as e:
                print(f"\nError: {e}\n")
                logger.error(f"Error in interactive mode: {e}", exc_info=True)
    
    except KeyboardInterrupt:
        print("\n\nSession ended by user.\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent Customer Service System"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="Run mode: 'demo' (batch processing) or 'interactive' (user input)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_mode()
    else:
        demo_mode()


if __name__ == "__main__":
    main()
