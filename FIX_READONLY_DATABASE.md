# Fix for ChromaDB Readonly Database Error

## Problem
The application was encountering the following error:
```
ERROR:vector_store:Error creating vector store: Database error: error returned from database: (code: 1032) attempt to write a readonly database
```

## Root Cause
ChromaDB was attempting to create its SQLite database in a directory that either:
1. Had incorrect permissions
2. Was not properly initialized
3. Had corrupted state from previous failed attempts

## Solution Implemented

### Changes to `vector_store.py`

1. **Added proper imports**:
   - `shutil` for directory cleanup
   - `os` for permission management
   - `chromadb` and `chromadb.config.Settings` for explicit client configuration

2. **New method `_ensure_persist_directory()`**:
   - Checks if the persist directory exists
   - Removes corrupted directories if detected
   - Creates directory with proper permissions (0o755)
   - Called during initialization and before creating vector stores

3. **Updated `create_vector_store()` method**:
   - Creates ChromaDB client with explicit settings:
     - `persist_directory`: Specifies where to store data
     - `anonymized_telemetry`: Disabled for privacy
     - `allow_reset`: Enabled for flexibility
     - `is_persistent`: Enabled to ensure data persists
   - Uses `PersistentClient` instead of relying on default client
   - Passes explicit client to Chroma constructor
   - Added cleanup on failure to prevent corrupted state

4. **Updated `load_vector_store()` method**:
   - Checks if directory is empty before attempting to load
   - Uses same explicit client settings as create method
   - Better error handling and logging

5. **Fixed type hints**:
   - Changed `k: int = None` to `k: Optional[int] = None`
   - Changed `threshold: float = None` to `threshold: Optional[float] = None`

## How It Works

The fix ensures that:
1. The persist directory always has correct permissions
2. ChromaDB uses explicit, persistent client settings
3. Failed operations clean up after themselves
4. The database is created with write permissions from the start

## Testing

After the dependencies are installed, test with:

```bash
# Clean start
rm -rf chroma_db

# Test document processing
python main.py process sample_document.txt

# Test querying
python main.py query "What is this about?"
```

Or use the Streamlit UI:
```bash
streamlit run app.py
```

## Prevention

The fix prevents the readonly database error by:
- Ensuring proper directory permissions upfront
- Using explicit ChromaDB client configuration
- Cleaning up corrupted state automatically
- Providing better error messages for debugging