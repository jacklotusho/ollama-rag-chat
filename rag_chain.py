"""RAG chain implementation using LangChain."""
from typing import Dict, Any, Optional
import logging

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from config import config
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """Retrieval-Augmented Generation chain with fallback to direct LLM."""
    
    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        self.vector_store_manager = vector_store_manager
        self.llm = None
        self.rag_chain = None
        self.direct_chain = None
        self.retriever = None
        self._initialize_llm()
        self._create_direct_chain()
        if vector_store_manager and vector_store_manager.vector_store:
            self._create_rag_chain()
    
    def _initialize_llm(self) -> None:
        """Initialize the Ollama LLM."""
        try:
            self.llm = ChatOllama(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                temperature=0.7,
            )
            logger.info(f"Initialized LLM with model: {config.ollama_model}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def _create_direct_chain(self) -> None:
        """Create a direct LLM chain (no RAG)."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer questions to the best of your ability."),
                ("human", "{question}")
            ])
            
            self.direct_chain = prompt | self.llm | StrOutputParser()
            logger.info("Direct LLM chain created successfully")
        except Exception as e:
            logger.error(f"Error creating direct chain: {str(e)}")
            raise
    
    def _format_docs(self, docs):
        """Format documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_rag_chain(self) -> None:
        """Create the RAG chain using LCEL."""
        if self.vector_store_manager is None or self.vector_store_manager.vector_store is None:
            logger.warning("Vector store not initialized, RAG chain creation skipped")
            return
        
        try:
            # Custom prompt template for RAG
            prompt_template = """You are a helpful AI assistant. Answer the question using the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the context
- If the question asks for code but the context doesn't contain code, explain what the context does contain and offer to help based on that information
- Include all relevant details, examples, and technical information from the context
- If the context is related to the question but doesn't have the exact answer requested, provide the relevant information you do have
- Be helpful and informative rather than simply saying you don't know
- Format your response clearly with proper structure

Answer: """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Get retriever
            self.retriever = self.vector_store_manager.get_retriever()
            
            # Create RAG chain using LCEL
            self.rag_chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("RAG chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {str(e)}")
            raise
    
    def _is_relevant_context(self, sources: list, question: str) -> bool:
        """Check if retrieved sources are relevant to the question."""
        if not sources:
            logger.info("No sources retrieved")
            return False
        
        # Simple heuristic: check if any source has reasonable length
        # In production, you might want more sophisticated relevance checking
        has_relevant = False
        for source in sources:
            content_length = len(source.page_content.strip())
            logger.debug(f"Source content length: {content_length}")
            if content_length > 50:
                has_relevant = True
                break
        
        logger.info(f"Relevant context found: {has_relevant} (checked {len(sources)} sources)")
        return has_relevant
    
    def query(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Query the chain with hybrid RAG + LLM fallback.
        
        Args:
            question: The question to answer
            use_rag: Whether to try RAG first (if available)
        
        Returns:
            Dict with answer, sources, question, and method used
        """
        try:
            logger.info(f"Processing query: {question}")
            logger.info(f"RAG chain available: {self.rag_chain is not None}")
            logger.info(f"Retriever available: {self.retriever is not None}")
            logger.info(f"Use RAG: {use_rag}")
            
            # Try RAG first if available and requested
            if use_rag and self.rag_chain is not None and self.retriever is not None:
                try:
                    logger.info("Attempting to retrieve documents...")
                    # Get source documents
                    source_documents = self.retriever.invoke(question)
                    logger.info(f"Retrieved {len(source_documents)} documents")
                    
                    # Check if we have relevant context
                    if self._is_relevant_context(source_documents, question):
                        logger.info("Using RAG to generate answer...")
                        # Use RAG
                        answer = self.rag_chain.invoke(question)
                        
                        # Format sources
                        sources = []
                        for i, doc in enumerate(source_documents, 1):
                            source_info = {
                                "index": i,
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            }
                            sources.append(source_info)
                        
                        logger.info(f"RAG query processed successfully with {len(sources)} sources")
                        
                        return {
                            "answer": answer,
                            "sources": sources,
                            "question": question,
                            "method": "rag"
                        }
                    else:
                        logger.info("No relevant context found, falling back to direct LLM")
                
                except Exception as e:
                    logger.warning(f"RAG query failed, falling back to direct LLM: {str(e)}", exc_info=True)
            else:
                logger.info("RAG not available or not requested, using direct LLM")
            
            # Fallback to direct LLM
            logger.info("Using direct LLM to generate answer...")
            answer = self.direct_chain.invoke({"question": question})
            logger.info("Direct LLM query processed successfully")
            
            return {
                "answer": answer,
                "sources": [],
                "question": question,
                "method": "direct_llm"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
    
    def update_model(self, model: str) -> None:
        """Update the LLM model."""
        config.update_ollama_model(model)
        self._initialize_llm()
        self._create_direct_chain()
        if self.vector_store_manager and self.vector_store_manager.vector_store:
            self._create_rag_chain()
        logger.info(f"Updated model to: {model}")
    
    def update_ollama_url(self, url: str) -> None:
        """Update the Ollama base URL."""
        config.update_ollama_url(url)
        self._initialize_llm()
        self._create_direct_chain()
        if self.vector_store_manager and self.vector_store_manager.vector_store:
            self._create_rag_chain()
        logger.info(f"Updated Ollama URL to: {url}")
    
    def reinitialize_chain(self) -> None:
        """Reinitialize the chain with current vector store."""
        if self.vector_store_manager and self.vector_store_manager.vector_store:
            self._create_rag_chain()

# Made with Bob
