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
            
            # Check if the collection has documents
            try:
                collection = self.vector_store._collection
                count = collection.count()
                logger.info(f"Loaded existing vector store with {count} documents")
                
                if count == 0:
                    logger.warning("Vector store loaded but contains no documents")
                    return None
            except Exception as e:
                logger.warning(f"Could not verify document count: {str(e)}")
            
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
    
    def similarity_search_with_threshold(self, query: str, k: int = None, threshold: float = None) -> List[Document]:
        """Search for similar documents and filter by threshold."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or config.top_k_results
        threshold = threshold or config.similarity_threshold
        
        try:
            # Chroma returns (document, score) where score is distance (lower is better for distance, higher is better for similarity)
            # LangChain Chroma wrapper usually returns L2 distance.
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by threshold. Note: For distance metrics, "similarity" usually means low distance.
            # We'll assume the threshold is a minimum similarity or maximum distance depending on the metric.
            # Standardizing this can be tricky, but for now let's filter based on a simple distance threshold.
            # Many LangChain retrievers treat 'score_threshold' in search_kwargs for 'similarity_score_threshold' search type.
            
            filtered_results = []
            for doc, score in results_with_scores:
                logger.info(f"Document score: {score}")
                # For Chroma with L2 distance, smaller is better.
                # If we use cosine similarity, larger is better.
                # Let's check what Chroma uses by default in LangChain. It's usually L2 distance.
                if score <= (1 - threshold) * 2: # Very rough heuristic for distance vs similarity
                    filtered_results.append(doc)
            
            logger.info(f"Found {len(filtered_results)} documents passing threshold {threshold}")
            return filtered_results
        except Exception as e:
            logger.error(f"Error during similarity search with threshold: {str(e)}")
            raise
    
    def get_retriever(self, k: int = None, threshold: float = None):
        """Get a retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or config.top_k_results
        threshold = threshold or config.similarity_threshold
        
        # Use similarity_score_threshold search type if threshold is provided
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": threshold
            }
        )
    
    def delete_collection(self) -> None:
        """Delete the vector store collection and remove persist directory."""
        import shutil
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                self.vector_store = None
            
            # Physically remove the directory
            persist_path = Path(config.chroma_persist_directory)
            if persist_path.exists():
                shutil.rmtree(persist_path)
                logger.info(f"Removed persist directory: {config.chroma_persist_directory}")
                
            logger.info("Deleted vector store collection and cleared disk storage")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def update_embeddings_model(self, model: str) -> None:
        """Update the embeddings model."""
        config.update_ollama_model(model)
        self._initialize_embeddings()
        logger.info(f"Updated embeddings model to: {model}")

# Made with Bob
