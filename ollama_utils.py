"""Utilities for interacting with Ollama server."""
import logging
from typing import List, Dict, Optional
import requests

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_models(base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Fetch available models from Ollama server.
    
    Args:
        base_url: Ollama server URL (uses config default if not provided)
    
    Returns:
        List of model dictionaries with 'name' and 'size' keys
    """
    url = base_url or config.ollama_base_url
    
    try:
        # Ollama API endpoint for listing models
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        models = []
        
        if "models" in data:
            for model in data["models"]:
                models.append({
                    "name": model.get("name", "unknown"),
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", ""),
                    "digest": model.get("digest", "")
                })
        
        logger.info(f"Found {len(models)} models on Ollama server")
        return models
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models from Ollama: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching models: {str(e)}")
        return []


def format_model_size(size_bytes: int) -> str:
    """Format model size in human-readable format."""
    if size_bytes == 0:
        return "Unknown size"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def test_ollama_connection(base_url: Optional[str] = None) -> bool:
    """
    Test connection to Ollama server.
    
    Args:
        base_url: Ollama server URL (uses config default if not provided)
    
    Returns:
        True if connection successful, False otherwise
    """
    url = base_url or config.ollama_base_url
    
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        logger.info(f"Successfully connected to Ollama at {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {url}: {str(e)}")
        return False


def get_model_info(model_name: str, base_url: Optional[str] = None) -> Optional[Dict]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        base_url: Ollama server URL (uses config default if not provided)
    
    Returns:
        Model information dictionary or None if not found
    """
    models = get_available_models(base_url)
    
    for model in models:
        if model["name"] == model_name:
            return model
    
    return None

# Made with Bob
