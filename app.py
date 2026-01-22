"""Streamlit UI for the RAG Chat Application."""
import streamlit as st
from pathlib import Path
import logging
from typing import List

from config import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain
from ollama_utils import get_available_models, test_ollama_connection, format_model_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'vector_store_manager' not in st.session_state:
        # Try to load existing vector store on startup
        try:
            vector_store_manager = VectorStoreManager()
            loaded_store = vector_store_manager.load_vector_store()
            if loaded_store is not None:
                st.session_state.vector_store_manager = vector_store_manager
                st.session_state.documents_loaded = True
                logger.info("Loaded existing vector store on startup")
            else:
                st.session_state.vector_store_manager = None
                st.session_state.documents_loaded = False
        except Exception as e:
            logger.warning(f"Could not load vector store on startup: {str(e)}")
            st.session_state.vector_store_manager = None
            st.session_state.documents_loaded = False
    
    if 'rag_chain' not in st.session_state:
        # Initialize RAG chain with vector store if available
        if st.session_state.get('documents_loaded', False) and st.session_state.vector_store_manager:
            rag_chain = RAGChain(st.session_state.vector_store_manager)
            # Ensure the RAG chain is properly initialized with the loaded vector store
            if rag_chain.rag_chain is None:
                rag_chain.reinitialize_chain()
            st.session_state.rag_chain = rag_chain
            logger.info("RAG chain initialized with loaded vector store")
        else:
            # Initialize RAG chain without documents for direct LLM access
            st.session_state.rag_chain = RAGChain()
            logger.info("RAG chain initialized in direct LLM mode")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Ollama Configuration
        st.subheader("Ollama Settings")
        
        # Ollama URL input
        ollama_url = st.text_input(
            "Ollama Base URL",
            value=config.ollama_base_url,
            help="URL where Ollama is running (e.g., http://localhost:11434)"
        )
        
        # Test connection and fetch models
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Refresh Models", use_container_width=True):
                with st.spinner("Fetching models..."):
                    if test_ollama_connection(ollama_url):
                        st.success("Connected!")
                        st.session_state.available_models = get_available_models(ollama_url)
                    else:
                        st.error("Connection failed!")
                        st.session_state.available_models = []
        
        with col2:
            connection_status = "🟢" if test_ollama_connection(ollama_url) else "🔴"
            st.markdown(f"### {connection_status}")
        
        # Initialize available models if not already done
        if 'available_models' not in st.session_state:
            st.session_state.available_models = get_available_models(ollama_url)
        
        # Model selection
        if st.session_state.available_models:
            model_names = [m["name"] for m in st.session_state.available_models]
            
            # Find current model index
            try:
                current_index = model_names.index(config.ollama_model)
            except ValueError:
                current_index = 0
            
            selected_model = st.selectbox(
                "Select Ollama Model",
                options=model_names,
                index=current_index,
                help="Choose from available models on your Ollama server"
            )
            
            # Show model info
            selected_model_info = next(
                (m for m in st.session_state.available_models if m["name"] == selected_model),
                None
            )
            if selected_model_info:
                st.caption(f"Size: {format_model_size(selected_model_info['size'])}")
        else:
            # Fallback to text input if no models found
            st.warning("⚠️ No models found. Using manual input.")
            selected_model = st.text_input(
                "Ollama Model",
                value=config.ollama_model,
                help="Name of the Ollama model to use (e.g., llama2, mistral)"
            )
        
        # Update settings button
        if st.button("✅ Update Ollama Settings", use_container_width=True):
            config.update_ollama_url(ollama_url)
            config.update_ollama_model(selected_model)
            
            # Reinitialize components if they exist
            if st.session_state.vector_store_manager:
                st.session_state.vector_store_manager.update_embeddings_model(selected_model)
            if st.session_state.rag_chain:
                st.session_state.rag_chain.update_ollama_url(ollama_url)
                st.session_state.rag_chain.update_model(selected_model)
            
            st.success(f"Settings updated! Using model: {selected_model}")
            st.rerun()
        
        st.divider()
        
        # Document Processing Settings
        st.subheader("Document Settings")
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=config.chunk_size,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=config.chunk_overlap,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=config.top_k_results,
            help="Number of relevant documents to retrieve"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.similarity_threshold,
            step=0.05,
            help="Minimum similarity score for a document to be considered relevant (higher is stricter)"
        )
        
        if st.button("Update Document Settings"):
            config.chunk_size = chunk_size
            config.chunk_overlap = chunk_overlap
            config.top_k_results = top_k
            config.similarity_threshold = similarity_threshold
            
            # Recreate RAG chain to apply new settings to retriever
            if st.session_state.vector_store_manager:
                st.session_state.rag_chain = RAGChain(st.session_state.vector_store_manager)
            
            st.success("Document settings updated!")
        
        st.divider()
        
        # Vector Store Management
        st.subheader("Vector Store")
        if st.button("Clear Vector Store"):
            if st.session_state.vector_store_manager:
                st.session_state.vector_store_manager.delete_collection()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.success("Vector store cleared!")
            else:
                st.warning("No vector store to clear")


def upload_documents():
    """Handle document upload and processing."""
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Initialize components if needed
                if st.session_state.vector_store_manager is None:
                    st.session_state.vector_store_manager = VectorStoreManager()
                
                # Save uploaded files temporarily
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))
                
                # Process documents
                processor = DocumentProcessor()
                chunks = processor.process_multiple_files(file_paths)
                
                # Add to vector store
                if st.session_state.documents_loaded:
                    st.session_state.vector_store_manager.add_documents(chunks)
                else:
                    st.session_state.vector_store_manager.create_vector_store(chunks)
                    st.session_state.documents_loaded = True
                
                # Initialize RAG chain
                st.session_state.rag_chain = RAGChain(st.session_state.vector_store_manager)
                
                # Clean up temp files
                for file_path in file_paths:
                    Path(file_path).unlink()
                
                st.success(f"✅ Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                logger.error(f"Document processing error: {str(e)}")


def chat_interface():
    """Main chat interface."""
    st.header("💬 Chat with AI")
    
    # Show info about current mode
    if not st.session_state.documents_loaded:
        st.info("💡 No documents loaded. Chatting directly with the LLM. Upload documents to enable RAG mode!")
    else:
        st.success("📚 RAG mode active! Answers will be based on your documents when relevant.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show method badge
            if message["role"] == "assistant" and "method" in message:
                if message["method"] == "rag":
                    st.caption("🔍 Answer from documents")
                elif message["method"] == "retrieval_fallback":
                    st.caption("⚠️ Fallback: No relevant documents found. Answer generated from general knowledge.")
                elif message["method"] == "direct_llm":
                    st.caption("🤖 Direct LLM answer")
                else:
                    st.caption(f"ℹ️ Method: {message['method']}")
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**Source {source['index']}:**")
                        st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                        if source['metadata']:
                            st.caption(f"Metadata: {source['metadata']}")
                        st.divider()
    
    # Chat input
    prompt_text = "Ask a question..." if not st.session_state.documents_loaded else "Ask a question about your documents..."
    if question := st.chat_input(prompt_text):
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        with st.chat_message("user"):
            st.write(question)
        
        # Get response from RAG chain (with automatic fallback)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.query(question)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    method = result.get("method", "unknown")
                    
                    st.write(answer)
                    
                    # Show method badge
                    if method == "rag":
                        st.caption("🔍 Answer from documents")
                    elif method == "retrieval_fallback":
                        st.caption("⚠️ Fallback: No relevant documents found. Answer generated from general knowledge.")
                    elif method == "direct_llm":
                        st.caption("🤖 Direct LLM answer")
                    else:
                        st.caption(f"ℹ️ Method: {method}")
                    
                    # Add assistant message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "method": method
                    })
                    
                    # Display sources if available
                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources:
                                st.markdown(f"**Source {source['index']}:**")
                                st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                                if source['metadata']:
                                    st.caption(f"Metadata: {source['metadata']}")
                                st.divider()
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="RAG Chat Application",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG Chat Application")
    st.markdown("*Powered by Ollama and LangChain*")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        upload_documents()
    
    with col2:
        chat_interface()
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit, LangChain, and Ollama")


if __name__ == "__main__":
    main()

# Made with Bob
