"""Vector store management using ChromaDB."""
from typing import List, Optional
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store for document embeddings."""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize Ollama embeddings."""
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=config.ollama_base_url,
                model=config.ollama_model
            )
            logger.info(f"Initialized embeddings with model: {config.ollama_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents."""
        try:
            # Ensure persist directory exists
            Path(config.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=config.collection_name,
                persist_directory=config.chroma_persist_directory
            )
            
            logger.info(f"Created vector store with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing vector store from disk."""
        try:
            persist_path = Path(config.chroma_persist_directory)
            
            if not persist_path.exists():
                logger.warning(f"Vector store directory not found: {config.chroma_persist_directory}")
                return None
            
            self.vector_store = Chroma(
                collection_name=config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=config.chroma_persist_directory
            )
            
            logger.info("Loaded existing vector store")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store."""
        if self.vector_store is None:
            logger.warning("No vector store loaded, creating new one")
            self.create_vector_store(documents)
        else:
            try:
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
            except Exception as e:
                logger.error(f"Error adding documents: {str(e)}")
                raise
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or config.top_k_results
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def get_retriever(self, k: int = None):
        """Get a retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or config.top_k_results
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def delete_collection(self) -> None:
        """Delete the vector store collection."""
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                self.vector_store = None
                logger.info("Deleted vector store collection")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def update_embeddings_model(self, model: str) -> None:
        """Update the embeddings model."""
        config.update_ollama_model(model)
        self._initialize_embeddings()
        logger.info(f"Updated embeddings model to: {model}")

# Made with Bob
