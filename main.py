"""CLI entry point for the RAG application."""
import sys
import argparse
from pathlib import Path

from config import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain


def process_documents(file_paths: list[str]) -> None:
    """Process documents and add them to the vector store."""
    print(f"Processing {len(file_paths)} document(s)...")
    
    # Initialize components
    processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    
    # Process documents
    chunks = processor.process_multiple_files(file_paths)
    
    # Create or load vector store
    existing_store = vector_store_manager.load_vector_store()
    if existing_store:
        print("Adding to existing vector store...")
        vector_store_manager.add_documents(chunks)
    else:
        print("Creating new vector store...")
        vector_store_manager.create_vector_store(chunks)
    
    print(f"✅ Successfully processed {len(chunks)} chunks!")


def query_documents(question: str) -> None:
    """Query the RAG system."""
    # Initialize components
    vector_store_manager = VectorStoreManager()
    
    # Load vector store
    if not vector_store_manager.load_vector_store():
        print("❌ No vector store found. Please process documents first.")
        return
    
    # Create RAG chain
    rag_chain = RAGChain(vector_store_manager)
    
    # Query
    print(f"\n❓ Question: {question}\n")
    result = rag_chain.query(question)
    
    print(f"💡 Answer: {result['answer']}\n")
    
    if result['sources']:
        print("📚 Sources:")
        for source in result['sources']:
            print(f"\n  Source {source['index']}:")
            content = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
            print(f"  {content}")


def interactive_mode() -> None:
    """Run in interactive query mode."""
    # Initialize components
    vector_store_manager = VectorStoreManager()
    
    # Load vector store
    if not vector_store_manager.load_vector_store():
        print("❌ No vector store found. Please process documents first.")
        return
    
    # Create RAG chain
    rag_chain = RAGChain(vector_store_manager)
    
    print("\n🤖 RAG Chat - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n💭 Thinking...\n")
            result = rag_chain.query(question)
            
            print(f"💡 Answer: {result['answer']}\n")
            
            show_sources = input("Show sources? (y/n): ").strip().lower()
            if show_sources == 'y' and result['sources']:
                print("\n📚 Sources:")
                for source in result['sources']:
                    print(f"\n  Source {source['index']}:")
                    content = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    print(f"  {content}")
            
            print("\n" + "-" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Chat Application - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process documents
  python main.py process document1.pdf document2.txt
  
  # Query documents
  python main.py query "What is the main topic?"
  
  # Interactive mode
  python main.py interactive
  
  # Launch web UI
  streamlit run app.py
        """
    )
    
    parser.add_argument(
        '--ollama-url',
        default=config.ollama_base_url,
        help='Ollama base URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--model',
        default=config.ollama_model,
        help='Ollama model name (default: llama2)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('files', nargs='+', help='Document files to process')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query documents')
    query_parser.add_argument('question', help='Question to ask')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Interactive query mode')
    
    args = parser.parse_args()
    
    # Update config if specified
    if args.ollama_url != config.ollama_base_url:
        config.update_ollama_url(args.ollama_url)
    if args.model != config.ollama_model:
        config.update_ollama_model(args.model)
    
    # Execute command
    if args.command == 'process':
        process_documents(args.files)
    elif args.command == 'query':
        query_documents(args.question)
    elif args.command == 'interactive':
        interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# Made with Bob
