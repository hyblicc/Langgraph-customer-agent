"""
RAG retriever implementation
"""

import logging
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from .vector_store import VectorStore
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGRetriever:
    """
    Retrieval-Augmented Generation (RAG) retriever
    Handles knowledge base retrieval, ranking, and formatting
    """
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: VectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        logger.info("RAGRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents from knowledge base
        
        Args:
            query: Search query
            top_k: Number of results
            threshold: Similarity threshold
        
        Returns:
            Retrieval results with formatted context
        """
        top_k = top_k or settings.top_k_retrieval
        threshold = threshold or settings.retrieval_score_threshold
        
        logger.info(f"Retrieving documents for query: {query[:100]}")
        
        # Search vector store
        results = self.vector_store.search(query, top_k, threshold)
        
        if not results:
            logger.warning("No documents retrieved")
            return {
                "context": "",
                "sources": [],
                "scores": [],
                "count": 0,
            }
        
        # Extract information
        contexts = [r["content"] for r in results]
        sources = [r["metadata"] for r in results]
        scores = [r["score"] for r in results]
        
        # Format context
        context_text = self._format_context(contexts, sources, scores)
        
        logger.info(f"Retrieved {len(results)} documents (avg score: {sum(scores)/len(scores):.3f})")
        
        return {
            "context": context_text,
            "sources": sources,
            "scores": scores,
            "count": len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
        }
    
    def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents considering conversation context
        
        Args:
            query: Current query
            conversation_history: Previous messages for context
            top_k: Number of results
        
        Returns:
            Retrieval results with context-aware ranking
        """
        # Augment query with recent context if available
        augmented_query = query
        
        if conversation_history:
            # Take last 2 messages as context
            recent_context = " ".join([
                msg.get("content", "")
                for msg in conversation_history[-2:]
                if msg.get("role") == "user"
            ])
            if recent_context:
                augmented_query = f"{recent_context} {query}"
        
        return self.retrieve(augmented_query, top_k)
    
    def rerank_results(
        self,
        results: Dict[str, Any],
        query: str,
        method: str = "score",
    ) -> Dict[str, Any]:
        """
        Re-rank retrieval results
        
        Args:
            results: Original retrieval results
            query: Original query
            method: Ranking method ('score', 'relevance', 'recency')
        
        Returns:
            Re-ranked results
        """
        logger.info(f"Re-ranking results using method: {method}")
        
        if method == "score":
            # Already sorted by score
            return results
        
        elif method == "relevance":
            # Re-rank by semantic relevance (in production, use cross-encoder)
            sources = results.get("sources", [])
            scores = results.get("scores", [])
            
            # Simple mock re-ranking: boost sources marked as most_relevant
            boosted_scores = [
                s * 1.1 if src.get("most_relevant") else s
                for s, src in zip(scores, sources)
            ]
            
            # Resort
            sorted_indices = sorted(range(len(boosted_scores)), key=lambda i: boosted_scores[i], reverse=True)
            
            results["scores"] = [boosted_scores[i] for i in sorted_indices]
            results["sources"] = [sources[i] for i in sorted_indices]
            results["context"] = self._format_context(
                [results["context"].split("\n---\n")[i] for i in sorted_indices],
                [sources[i] for i in sorted_indices],
                [boosted_scores[i] for i in sorted_indices],
            )
        
        return results
    
    def _format_context(
        self,
        contexts: List[str],
        sources: List[Dict[str, str]],
        scores: List[float],
    ) -> str:
        """
        Format retrieved context for LLM consumption
        
        Args:
            contexts: Document contents
            sources: Source metadata
            scores: Relevance scores
        
        Returns:
            Formatted context string
        """
        formatted_parts = []
        
        for i, (context, source, score) in enumerate(zip(contexts, sources, scores), 1):
            source_name = source.get("source", "Unknown")
            category = source.get("category", "general")
            
            part = f"""
[Source {i}: {source_name} (Category: {category}, Relevance: {score:.2%})]
{context}
"""
            formatted_parts.append(part)
        
        return "\n---\n".join(formatted_parts)
    
    def add_knowledge(self, documents: List[Document]) -> bool:
        """
        Add new documents to knowledge base
        
        Args:
            documents: List of Document objects
        
        Returns:
            Success status
        """
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self.vector_store.get_collection_stats()


# Global retriever instance
_retriever_instance = None


def get_retriever() -> RAGRetriever:
    """Get or create global retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance
