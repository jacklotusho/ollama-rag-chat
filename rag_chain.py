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
            prompt_template = """You are a helpful AI assistant. Use ONLY the provided context to answer the question. 

Context:
{context}

Question: {question}

Instructions:
1. If the answer is contained within the context, provide a concise and accurate response.
2. If the answer cannot be found in the provided context, OR if the context is empty or irrelevant, strictly respond with: "I do not have enough information to answer this question based on the provided documents."
3. Do NOT use any external or internal knowledge not present in the context.
4. Do NOT make up facts or fabricate information.
5. If the context is empty, simply state that no information was found.

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
    
    def _is_relevant_context(self, sources: list) -> bool:
        """Check if retrieved sources are inherently empty or missing."""
        if not sources:
            logger.info("No sources retrieved from vector store")
            return False
        
        # Check if all sources are just whitespace or extremely short
        has_content = False
        for source in sources:
            content = source.page_content.strip()
            if len(content) > 10:  # Minimum character count to be considered useful
                has_content = True
                break
        
        if not has_content:
            logger.info("Retrieved sources contain no meaningful content")
            return False
            
        logger.info(f"Found {len(sources)} potentially relevant sources")
        return True
    
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
            # RAG flow
            if use_rag:
                if self.rag_chain is None or self.retriever is None:
                    return {
                        "answer": "RAG system is not initialized. Please upload documents first.",
                        "sources": [],
                        "question": question,
                        "method": "error"
                    }

                try:
                    logger.info("Attempting to retrieve documents...")
                    # Get source documents
                    # The retriever already filters by similarity threshold if configured
                    source_documents = self.retriever.invoke(question)
                    logger.info(f"Retrieved {len(source_documents)} documents")
                    
                    # Check if we have relevant context ("Stop-First" check)
                    if self._is_relevant_context(source_documents):
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
                        logger.info("Stop-First intervention: No relevant context found. Falling back to direct LLM.")
                        # Fallback to direct LLM
                        answer = self.direct_chain.invoke({"question": question})
                        return {
                            "answer": answer,
                            "sources": [],
                            "question": question,
                            "method": "retrieval_fallback"
                        }
                
                except Exception as e:
                    logger.warning(f"RAG query failed: {str(e)}", exc_info=True)
                    return {
                        "answer": f"An error occurred during retrieval: {str(e)}",
                        "sources": [],
                        "question": question,
                        "method": "error"
                    }
            
            # Direct LLM flow (only if use_rag is False)
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
