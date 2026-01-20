"""
Configuration settings for the embedding pipeline.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for the pipeline."""
    
    # Weaviate settings
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
    
    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Weaviate schema
    COLLECTION_NAME = "Document"
