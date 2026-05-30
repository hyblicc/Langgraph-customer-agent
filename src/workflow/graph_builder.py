"""
LangGraph workflow graph construction
"""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state_schema import AgentState, IntentType
from .nodes import (
    intent_recognition_node,
    knowledge_retrieval_node,
    tool_calling_node,
    answer_generation_node,
    route_by_intent,
)

logger = logging.getLogger(__name__)


def build_workflow_graph() -> StateGraph:
    """
    Build the multi-agent workflow graph
    
    Structure:
    1. Start → Intent Recognition
    2. Intent Recognition → Router (route by intent type)
    3. Knowledge-only path: Knowledge Retrieval → Answer Generation → End
    4. Tool-only path: Tool Calling → Answer Generation → End
    5. Hybrid path: Knowledge Retrieval + Tool Calling (parallel) → Answer Generation → End
    """
    
    logger.info("Building workflow graph...")
    
    # Initialize graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    logger.debug("Adding nodes...")
    graph.add_node("intent_recognition", intent_recognition_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("answer_generation", answer_generation_node)
    
    # Add edges
    logger.debug("Adding edges...")
    
    # Entry point
    graph.set_entry_point("intent_recognition")
    
    # Intent recognition → Router
    graph.add_conditional_edges(
        "intent_recognition",
        route_by_intent,
        {
            "knowledge_only": "knowledge_retrieval",
            "tool_only": "tool_calling",
            "hybrid": "knowledge_retrieval",  # Start with knowledge retrieval in hybrid mode
        }
    )
    
    # Knowledge retrieval paths
    graph.add_edge("knowledge_retrieval", "answer_generation")
    
    # Tool calling paths
    graph.add_edge("tool_calling", "answer_generation")
    
    # Answer generation → End
    graph.set_finish_point("answer_generation")
    
    logger.info("Workflow graph built successfully")
    
    return graph


def get_compiled_graph():
    """
    Get a compiled and ready-to-use graph
    
    Returns:
        Compiled graph with memory checkpointing enabled
    """
    
    logger.info("Compiling workflow graph...")
    
    # Build graph
    graph = build_workflow_graph()
    
    # Add memory checkpointing for persistence
    memory = MemorySaver()
    
    # Compile graph
    compiled = graph.compile(checkpointer=memory)
    
    logger.info("Workflow graph compiled successfully with memory checkpointing")
    
    return compiled


def visualize_graph():
    """
    Visualize the workflow graph structure
    
    Returns:
        Graph visualization as string
    """
    
    graph = build_workflow_graph()
    
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {e}")
        return None


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Get compiled graph
    compiled_graph = get_compiled_graph()
    
    # Visualize
    viz = visualize_graph()
    if viz:
        print("Graph visualization:")
        print(viz)
    
    print("\nGraph compiled successfully!")
    print(f"Nodes: {list(compiled_graph.nodes.keys())}")
