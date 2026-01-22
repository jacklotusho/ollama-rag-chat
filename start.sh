export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=granite4:latest

source .venv/bin/activate

uv run streamlit run app.py
