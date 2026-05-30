"""
Supervisor Agent - Central coordinator for multi-agent system
"""

import logging
import json
from typing import Dict, Any, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.config import get_settings
from src.workflow.state_schema import AgentState

logger = logging.getLogger(__name__)
settings = get_settings()


class SupervisorAgent:
    """
    Central supervisor agent that:
    1. Coordinates multiple specialized agents
    2. Routes tasks to appropriate agents
    3. Aggregates results
    4. Manages error handling and fallback strategies
    """
    
    def __init__(self):
        """Initialize supervisor agent"""
        self.llm = ChatOpenAI(
            model_name=settings.supervisor_model,
            temperature=settings.supervisor_temperature,
            openai_api_key=settings.openai_api_key,
        )
        logger.info("SupervisorAgent initialized")
    
    async def coordinate_agents(self, state: AgentState) -> Dict[str, Any]:
        """
        Coordinate multiple agents for complex queries
        
        Args:
            state: Current agent state
        
        Returns:
            Coordination results
        """
        logger.info(f"Coordinating agents for query: {state['query'][:100]}")
        
        try:
            # Determine coordination strategy
            if state["intent"].value == "hybrid_query":
                return await self._coordinate_hybrid_query(state)
            else:
                # Single agent handling is sufficient
                return {"coordination_required": False}
        
        except Exception as e:
            logger.error(f"Error coordinating agents: {e}")
            return {
                "coordination_required": False,
                "error": str(e),
            }
    
    async def _coordinate_hybrid_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Coordinate RAG Agent and Tool Agent for hybrid queries
        
        Strategy:
        1. RAG Agent retrieves knowledge in parallel
        2. Tool Agent executes API calls in parallel
        3. Aggregate results and generate combined response
        """
        logger.info("Coordinating hybrid query")
        
        # In production, these would run in parallel
        # For now, they run sequentially through the workflow
        
        return {
            "coordination_required": True,
            "strategy": "parallel_rag_tool",
            "rag_agent_active": True,
            "tool_agent_active": True,
            "priority": "rag_first",  # RAG results take priority if conflicts arise
        }
    
    async def aggregate_results(
        self,
        rag_results: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents
        
        Args:
            rag_results: Knowledge retrieval results
            tool_results: Tool execution results
        
        Returns:
            Aggregated results
        """
        logger.info("Aggregating results from multiple agents")
        
        try:
            # Combine contexts
            combined_context = ""
            
            if rag_results.get("context"):
                combined_context += f"Knowledge Base Information:\n{rag_results['context']}\n\n"
            
            if tool_results:
                combined_context += f"Real-time Data:\n{json.dumps(tool_results, ensure_ascii=False)}\n"
            
            return {
                "combined_context": combined_context,
                "rag_score": rag_results.get("avg_score", 0),
                "has_tool_results": bool(tool_results),
                "aggregation_method": "sequential",
            }
        
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return {
                "combined_context": "",
                "error": str(e),
            }
    
    async def handle_fallback(
        self,
        original_error: str,
        state: AgentState,
    ) -> Dict[str, Any]:
        """
        Execute fallback strategy when primary approach fails
        
        Fallback hierarchy:
        1. If Tool Agent fails → try RAG Agent only
        2. If RAG Agent fails → try Tool Agent only
        3. If both fail → generate generic response
        
        Args:
            original_error: Original error message
            state: Current agent state
        
        Returns:
            Fallback strategy results
        """
        logger.warning(f"Executing fallback strategy: {original_error}")
        
        try:
            if state["retry_count"] >= settings.max_retries:
                logger.warning("Max retries reached, using generic response")
                return {
                    "fallback_strategy": "generic_response",
                    "message": "I apologize, but I'm unable to fully answer your question right now. Please try again later or contact customer support.",
                }
            
            # Fallback logic
            if state["tool_errors"] and not state["knowledge_context"]:
                # Tools failed, try RAG only
                logger.info("Falling back to RAG-only approach")
                return {"fallback_strategy": "rag_only"}
            
            elif state["knowledge_context"] and state["tool_errors"]:
                # Both available, prefer knowledge
                logger.info("Falling back to knowledge preference")
                return {"fallback_strategy": "knowledge_priority"}
            
            else:
                # Generic fallback
                logger.info("Using generic fallback")
                return {"fallback_strategy": "generic_response"}
        
        except Exception as e:
            logger.error(f"Error in fallback handling: {e}")
            return {"fallback_strategy": "error", "error": str(e)}
    
    async def evaluate_response_quality(
        self,
        response: str,
        state: AgentState,
    ) -> Dict[str, Any]:
        """
        Evaluate quality of generated response
        
        Args:
            response: Generated response text
            state: Current agent state
        
        Returns:
            Quality evaluation results
        """
        logger.info("Evaluating response quality")
        
        try:
            # Simple quality metrics
            quality_score = 0.0
            issues = []
            
            # Check if response is too short
            if len(response) < 50:
                issues.append("Response too short")
                quality_score -= 0.1
            
            # Check if it contains useful information
            if any(keyword in response.lower() for keyword in ["error", "failed", "unable"]):
                issues.append("Response indicates error or limitation")
                quality_score -= 0.05
            
            # Base score from intent confidence
            quality_score += state.get("intent_confidence", 0.5)
            
            # Adjust based on sources
            if state.get("answer_sources"):
                quality_score += 0.1 * min(len(state["answer_sources"]), 2)
            
            # Normalize to [0, 1]
            quality_score = max(0.0, min(1.0, quality_score))
            
            return {
                "quality_score": quality_score,
                "is_high_quality": quality_score >= 0.7,
                "issues": issues,
            }
        
        except Exception as e:
            logger.error(f"Error evaluating response quality: {e}")
            return {
                "quality_score": 0.0,
                "is_high_quality": False,
                "error": str(e),
            }
    
    async def log_interaction(self, state: AgentState, response: str) -> None:
        """Log interaction for monitoring and analysis"""
        try:
            interaction_log = {
                "user_id": state.get("user_id"),
                "query": state.get("query"),
                "intent": state.get("intent", {}).value if hasattr(state.get("intent"), "value") else str(state.get("intent")),
                "response_length": len(response),
                "sources_used": state.get("answer_sources", []),
                "confidence": state.get("confidence_score", 0),
            }
            logger.info(f"Interaction: {json.dumps(interaction_log, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"Error logging interaction: {e}")
