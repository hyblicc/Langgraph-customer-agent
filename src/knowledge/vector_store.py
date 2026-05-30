"""
ChromaDB vector store integration
"""

import logging
from typing import List, Optional, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    raise ImportError("chromadb is not installed. Install it with: pip install chromadb")

from langchain.schema import Document
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """
    ChromaDB-based vector store for knowledge management
    """
    
    def __init__(self, collection_name: str = None, persist_dir: str = None):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_dir: Directory for persistent storage
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        
        logger.info(f"Initializing VectorStore (collection: {self.collection_name})")
        
        try:
            # Initialize ChromaDB client
            chroma_settings = ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
            )
            
            self.client = chromadb.Client(chroma_settings)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"VectorStore initialized (collection count: {self.collection.count()})")
        
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector store
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            for i, doc in enumerate(documents):
                doc_id = f"doc_{i}_{hash(doc.page_content) % 10000}"
                
                self.collection.add(
                    ids=[doc_id],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata or {}],
                )
            
            logger.info(f"Successfully added {len(documents)} documents")
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Search vector store for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
        
        Returns:
            List of search results with scores
        """
        top_k = top_k or settings.top_k_retrieval
        threshold = threshold or settings.retrieval_score_threshold
        
        logger.debug(f"Searching vector store (query: {query[:100]}, top_k: {top_k})")
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2,  # Fetch more to filter by threshold
            )
            
            if not results or not results["documents"]:
                logger.info("No documents found")
                return []
            
            # Process results
            documents = results["documents"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            
            # Convert distances to similarity scores (for cosine similarity)
            # ChromaDB returns distances, we convert to similarity [0, 1]
            similarities = [1 - (d / 2) for d in distances]
            
            # Filter by threshold and format results
            filtered_results = []
            for doc, sim, meta in zip(documents, similarities, metadatas):
                if sim >= threshold:
                    filtered_results.append({
                        "content": doc,
                        "score": sim,
                        "metadata": meta,
                    })
            
            logger.info(f"Found {len(filtered_results)} documents above threshold")
            
            return filtered_results[:top_k]
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from vector store"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Document deleted: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents in collection"""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs["ids"]:
                self.collection.delete(ids=all_docs["ids"])
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": "text-embedding-3-small",
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
