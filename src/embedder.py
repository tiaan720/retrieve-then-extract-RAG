"""
Embedding module using Ollama through LangChain.
"""
from typing import List
from langchain_ollama import OllamaEmbeddings
import requests


class EmbeddingGenerator:
    """Generates embeddings using Ollama via LangChain."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """
        Initialize the embedding generator.
        
        Args:
            base_url: Base URL for Ollama API
            model: Name of the embedding model to use
            
        Raises:
            ConnectionError: If Ollama service is not reachable
            ValueError: If the specified model is not available
        """
        self.base_url = base_url
        self.model = model
        
        # Validate Ollama connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            available_models = response.json().get('models', [])
            model_names = [m.get('name', '').split(':')[0] for m in available_models]
            
            if model not in model_names:
                print(f"⚠️  Warning: Model '{model}' not found in Ollama.")
                print(f"   Available models: {', '.join(model_names) if model_names else 'none'}")
                print(f"   To install: ollama pull {model}")
                print(f"   Proceeding anyway - model will be pulled on first use.")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {base_url}. "
                f"Please ensure Ollama is running. Error: {e}"
            )
        
        # Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=base_url,
                model=model
            )
            print(f"✓ Initialized Ollama embeddings with model: {model}")
        except Exception as e:
            raise ValueError(f"Failed to initialize OllamaEmbeddings: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for document chunks and add to chunk data.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            
        Returns:
            List of chunk dictionaries with added 'embedding' key
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            embedded_chunks.append(chunk_with_embedding)
        
        print(f"Successfully generated {len(embedded_chunks)} embeddings")
        return embedded_chunks
