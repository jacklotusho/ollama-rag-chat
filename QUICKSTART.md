# Quick Start Guide

## Prerequisites

1. **Install Ollama**
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

2. **Start Ollama**
   ```bash
   ollama serve
   ```

3. **Pull a model**
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```

## Installation

1. **Clone and navigate to the project**
   ```bash
   cd ollama-rag-chat
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

## Running the Application

### Web UI (Recommended)

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`

### CLI Mode

**Process documents:**
```bash
uv run python main.py process document1.pdf document2.txt
```

**Query documents:**
```bash
uv run python main.py query "What is the main topic?"
```

**Interactive mode:**
```bash
uv run python main.py interactive
```

## Using the Web UI

1. **Configure Ollama** (in sidebar)
   - Set Ollama URL (default: `http://localhost:11434`)
   - Choose your model (e.g., `llama2`, `mistral`)
   - Click "Update Ollama Settings"

2. **Upload Documents** (left panel)
   - Click "Upload PDF or text files"
   - Select your documents
   - Click "Process Documents"

3. **Chat** (right panel)
   - Ask questions about your documents
   - View source citations by expanding "View Sources"

## Configuration

### Environment Variables

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
```

### Supported File Types

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)

## Troubleshooting

### "Connection refused" error
- Ensure Ollama is running: `ollama serve`
- Check the Ollama URL in settings

### "Model not found" error
- Pull the model: `ollama pull llama2`
- Verify with: `ollama list`

### Import errors
- Reinstall dependencies: `uv sync --reinstall`

## Tips

- **Better answers**: Use more specific questions
- **Performance**: Adjust chunk size and top K in settings
- **Memory**: Process large documents in batches
- **Models**: Try different models for different tasks
  - `llama2`: General purpose
  - `mistral`: Fast and efficient
  - `codellama`: For code-related questions

## Next Steps

- Explore different Ollama models
- Adjust document processing settings
- Try the CLI for automation
- Check the full README for advanced features