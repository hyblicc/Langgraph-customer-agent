"""
Configuration settings management
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    openai_temperature: float = 0.7
    
    # ChromaDB Configuration
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "customer_knowledge_base")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    
    # Application Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # RAG Configuration
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 3
    retrieval_score_threshold: float = 0.6
    
    # Agent Configuration
    max_retries: int = 2
    retry_delay: float = 1.0
    agent_timeout: float = 30.0
    
    # Supervisor Configuration
    supervisor_model: str = "gpt-4"
    supervisor_temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    load_dotenv()
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print(f"OpenAI Model: {settings.openai_model}")
    print(f"ChromaDB Collection: {settings.chroma_collection_name}")
    print(f"Debug Mode: {settings.debug}")
