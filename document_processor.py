"""Document processing and chunking utilities."""
from typing import List
from pathlib import Path
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from file path."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(file_path)
                logger.info(f"Loading PDF: {file_path}")
            elif path.suffix.lower() in ['.txt', '.md']:
                loader = TextLoader(file_path)
                logger.info(f"Loading text file: {file_path}")
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and common noise."""
        import re
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Remove common PDF artifacts or weird characters if necessary
        # (This is a basic implementation, can be expanded)
        return text.strip()

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with cleaning."""
        try:
            # Clean document content before splitting
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)
            
            chunks = self.text_splitter.split_documents(documents)
            
            # Additional filtering for very small chunks that might be noise
            filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
            
            logger.info(f"Created {len(filtered_chunks)} cleaned chunks from {len(documents)} document(s)")
            return filtered_chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise
    
    def process_file(self, file_path: str) -> List[Document]:
        """Load and chunk a document file."""
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        return chunks
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Process multiple document files."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

# Made with Bob
