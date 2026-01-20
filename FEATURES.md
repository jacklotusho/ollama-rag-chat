# Features

## Core Capabilities

### 1. Hybrid RAG + Direct LLM
- **RAG Mode**: When documents are loaded, answers are based on document context
- **Direct LLM Mode**: Works without documents for general questions
- **Automatic Fallback**: If no relevant context found in documents, falls back to direct LLM
- **Method Indicator**: Shows whether answer came from documents (🔍) or direct LLM (🤖)

### 2. Configurable Ollama Integration
- **Dynamic URL Configuration**: Change Ollama URL without restarting
- **Model Selection**: Switch between any Ollama model (llama2, mistral, llama3, etc.)
- **Real-time Updates**: Configuration changes apply immediately

### 3. Document Processing
- **Multiple Formats**: PDF, TXT, and Markdown files
- **Intelligent Chunking**: Configurable chunk size and overlap
- **Batch Processing**: Upload and process multiple documents at once
- **Incremental Updates**: Add new documents to existing knowledge base

### 4. Vector Storage
- **ChromaDB Integration**: Efficient vector storage and retrieval
- **Persistent Storage**: Documents persist between sessions
- **Semantic Search**: Find relevant context using embeddings
- **Configurable Retrieval**: Adjust number of results (top K)

### 5. Interactive Chat Interface
- **Conversation History**: Maintains chat context
- **Source Citations**: View exact document passages used for answers
- **Metadata Display**: See document source information
- **Clear Visual Feedback**: Know when RAG vs direct LLM is used

### 6. Flexible Interfaces
- **Web UI**: Full-featured Streamlit interface
- **CLI**: Command-line tools for automation
  - Process documents
  - Query documents
  - Interactive mode

## Technical Features

### Architecture
- **Modern LangChain**: Uses LCEL (LangChain Expression Language)
- **Modular Design**: Separate components for easy maintenance
- **Error Handling**: Graceful fallbacks and error messages
- **Logging**: Comprehensive logging for debugging

### Configuration
- **Environment Variables**: Configure via `.env` file
- **Runtime Configuration**: Change settings in UI
- **Sensible Defaults**: Works out of the box

### Performance
- **Efficient Retrieval**: Fast vector similarity search
- **Streaming Support**: Real-time response generation
- **Resource Management**: Automatic cleanup of temporary files

## Use Cases

### 1. Document Q&A
Upload your documents and ask questions about them. Perfect for:
- Research papers
- Technical documentation
- Legal documents
- Meeting notes
- Knowledge bases

### 2. General Chat
Use without documents for:
- General questions
- Brainstorming
- Code assistance
- Writing help

### 3. Hybrid Mode
Best of both worlds:
- Ask document-specific questions (uses RAG)
- Ask general questions (uses direct LLM)
- Automatic switching based on context relevance

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    User Question                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Documents Loaded?    │
         └───────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
       Yes               No
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ Try RAG      │   │ Direct LLM   │
│ Retrieval    │   │              │
└──────┬───────┘   └──────┬───────┘
       │                   │
       ▼                   │
┌──────────────┐          │
│ Relevant     │          │
│ Context?     │          │
└──────┬───────┘          │
       │                   │
  ┌────┴────┐             │
  │         │             │
 Yes       No             │
  │         │             │
  │         └─────────────┤
  │                       │
  ▼                       ▼
┌──────────────┐   ┌──────────────┐
│ RAG Answer   │   │ LLM Answer   │
│ (with        │   │ (no sources) │
│  sources)    │   │              │
└──────────────┘   └──────────────┘
```

## Configuration Options

### Ollama Settings
- Base URL (default: http://localhost:11434)
- Model name (default: llama2)

### Document Processing
- Chunk size (default: 1000 characters)
- Chunk overlap (default: 200 characters)
- Top K results (default: 4 documents)

### Storage
- ChromaDB directory (default: ./chroma_db)
- Collection name (default: documents)

## Future Enhancements

Potential improvements:
- Multi-language support
- Advanced relevance scoring
- Document summarization
- Conversation memory
- Export chat history
- Custom prompt templates
- Multiple vector stores
- Document versioning