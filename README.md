# 🤖 Ollama RAG Chat Application

A powerful Retrieval-Augmented Generation (RAG) chat application that allows you to chat with your documents using Ollama and LangChain.

- 📄 **Document Processing**: Upload and process PDF, TXT, and Markdown files with automatic cleaning
- 🔍 **Semantic Search**: Uses ChromaDB for efficient vector storage and retrieval
- � **Stop-First RAG**: Prevents hallucinations by evaluating context relevance before generation
- 🏠 **Hybrid Fallback**: Gracefully falls back to general knowledge if no relevant document context is found
- �💬 **Interactive Chat**: Natural conversation interface with mode-specific indicators (RAG vs. Fallback)
- ⚙️ **Configurable Ollama**: Easily configure Ollama URL, model selection, and similarity thresholds
- 📚 **Source Citations**: View the source documents used to generate each answer
- 🎨 **Modern UI**: Clean and intuitive Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ollama-rag-chat
```

2. **Install dependencies**

Using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. **Start Ollama**

Make sure Ollama is running on your system:
```bash
ollama serve
```

4. **Pull an Ollama model** (if you haven't already)
```bash
ollama pull llama2
# or
ollama pull mistral
```

### Running the Application

Using UV:
```bash
uv run streamlit run app.py
```

Or directly:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

### 1. Configure Ollama Settings

In the sidebar, you can configure:
- **Ollama Base URL**: Default is `http://localhost:11434`
- **Ollama Model**: Choose your preferred model (e.g., `llama2`, `mistral`, `llama3`)

### 2. Upload Documents

- Click on "Upload PDF or text files" in the left panel
- Select one or more documents (PDF, TXT, or MD files)
- Click "Process Documents" to add them to the knowledge base

### 3. Chat with Your Documents

- Once documents are processed, use the chat interface on the right
- Ask questions about your documents
- View source citations by expanding the "View Sources" section

### 4. Advanced Settings

Adjust document processing parameters:
- **Chunk Size**: Size of text chunks (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Top K Results**: Number of relevant documents to retrieve (default: 4)
- **Similarity Threshold**: Minimum relevance score for a document to be used (default: 0.5)

## 🔧 Configuration

### Environment Variables

You can configure the application using environment variables:

```bash
# Ollama Configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama2"

# ChromaDB Configuration
export CHROMA_PERSIST_DIR="./chroma_db"
export COLLECTION_NAME="documents"

# Document Processing
export CHUNK_SIZE="1000"
export CHUNK_OVERLAP="200"
export TOP_K_RESULTS="4"

# Retrieval Protection
export SIMILARITY_THRESHOLD="0.5"
```

### Configuration File

The application uses [`config.py`](config.py) for centralized configuration management. You can modify default values there.

## 📁 Project Structure

```
ollama-rag-chat/
├── app.py                  # Streamlit UI application
├── config.py               # Configuration management
├── document_processor.py   # Document loading and chunking
├── vector_store.py         # ChromaDB vector store management
├── rag_chain.py           # RAG chain implementation
├── main.py                # CLI entry point (optional)
├── pyproject.toml         # Project dependencies
├── README.md              # This file
└── chroma_db/             # Vector database (created at runtime)
```

## 🛠️ Architecture

The application follows a standard RAG workflow:

1. **Document Processing**: Documents are loaded, cleaned (whitespace/noise removal), and split into semantic chunks.
2. **Embedding**: Text chunks are converted to embeddings using Ollama.
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval.
4. **Query & Retrieval**: User questions are embedded; the top K similar chunks are retrieved and filtered by a **Similarity Threshold**.
5. **Stop-First Evaluation**: The system evaluates if the retrieved context is relevant and meaningful.
6. **Hybrid Generation**:
    - **RAG Mode**: If relevant context exists, a strict prompt forces the LLM to answer using only the provided data.
    - **Fallback Mode**: If no relevant context is found, the system alerts the user and falls back to a general LLM query to remain helpful.

```
┌─────────────┐
│  Documents  │───► [Text Cleaning & Semantic Chunking]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │◄─── [Vector Storage]
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────┐
│  Retrieval  │◄────┤  Query   │
└──────┬──────┘     └──────────┘
       │
       ▼
┌─────────────┐
│ Stop-First  │───► [Score Threshold Check]
│ Evaluation  │
└──────┬──────┘
       │
       ├─ [Relevant Context Found] ──────┐
       │                                 │
       ▼                                 ▼
┌─────────────┐                   ┌─────────────┐
│  RAG Mode   │                   │  Fallback   │
│ (Strict)    │                   │   Mode      │
└──────┬──────┘                   └──────┬──────┘
       │                                 │
       └────────────────┬────────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   Answer    │
                 │ (with Badge)│
                 └─────────────┘
```

## 🔌 Ollama Models

The application works with any Ollama model. Popular choices:

- **llama2**: General purpose, good balance
- **mistral**: Fast and efficient
- **llama3**: Latest and most capable
- **codellama**: Optimized for code
- **phi**: Lightweight option

Pull models using:
```bash
ollama pull <model-name>
```

## 🐛 Troubleshooting

### Ollama Connection Issues

If you see connection errors:
1. Ensure Ollama is running: `ollama serve`
2. Check the Ollama URL in the sidebar
3. Verify the model is pulled: `ollama list`

### Memory Issues

For large documents:
1. Reduce chunk size in settings
2. Reduce top K results
3. Process documents in smaller batches

### Import Errors

If you encounter import errors:
```bash
uv sync --reinstall
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the GitHub repository.

---

Built with ❤️ using [Streamlit](https://streamlit.io/), [LangChain](https://langchain.com/), and [Ollama](https://ollama.ai/)