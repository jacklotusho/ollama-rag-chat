"""Configuration management for the RAG application."""
import os
from typing import Optional


class Config:
    """Application configuration."""
    
    def __init__(self):
        # Ollama configuration
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama2")
        
        # ChromaDB configuration
        self.chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name: str = os.getenv("COLLECTION_NAME", "documents")
        
        # Document processing configuration
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Retrieval configuration
        self.top_k_results: int = int(os.getenv("TOP_K_RESULTS", "4"))
        
    def update_ollama_url(self, url: str) -> None:
        """Update the Ollama base URL."""
        self.ollama_base_url = url
        
    def update_ollama_model(self, model: str) -> None:
        """Update the Ollama model."""
        self.ollama_model = model


# Global config instance
config = Config()

# Made with Bob
