from typing import List, Union, Optional
from abc import ABC, abstractmethod
from langchain_ollama import OllamaEmbeddings
import requests
import os
from tqdm import tqdm
from src.logger import logger

# Optional imports for local ColBERT
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch and/or transformers not available - HuggingFaceEmbedder will not work")


class BaseEmbedder(ABC):
    """Base class for embedding generators with shared functionality."""
    
    @abstractmethod
    def embed_text(self, text: str) -> Union[List[float], List[List[float]]]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> Union[List[List[float]], List[List[List[float]]]]:
        """Generate embeddings for multiple texts."""
        pass
    
    def embed_chunks(
        self, 
        chunks: List[dict], 
        embedding_key: str = 'embedding',
        batch_size: Optional[int] = None
    ) -> List[dict]:
        """
        Generate embeddings for document chunks (shared implementation).
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            embedding_key: Key name to store embeddings in chunks
            batch_size: If provided, process in batches of this size
            
        Returns:
            List of chunk dictionaries with added embedding key
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings (with optional batching)
        if batch_size:
            embeddings = []
            pbar = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch")
            for i in pbar:
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embed_texts(batch)
                embeddings.extend(batch_embeddings)
                pbar.set_postfix({"chunks": f"{min(i + batch_size, len(texts))}/{len(texts)}"})
        else:
            embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding[embedding_key] = embedding
            embedded_chunks.append(chunk_with_embedding)
        
        logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")
        
        # Log shape info for multi-vector embeddings
        if isinstance(embeddings[0], list) and isinstance(embeddings[0][0], list):
            logger.info(f"Sample embedding shape: {len(embeddings[0])} tokens × {len(embeddings[0][0])} dimensions")
        
        return embedded_chunks


class JinaAIEmbedder(BaseEmbedder):
    """Generates multi-vector embeddings using Jina AI API.
    
    Supports various Jina AI embedding models including:
    - jina-colbert-v2 (multi-vector ColBERT embeddings)
    - Other Jina AI embedding models
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "jina-colbert-v2"):
        """
        Initialize Jina AI embedder.
        
        Args:
            api_key: Jina AI API key (if None, reads from JINAAI_APIKEY env var)
            model: Model name (e.g., 'jina-colbert-v2' for ColBERT embeddings)
            
        Raises:
            ValueError: If API key is not provided or found
        """
        self.api_key = api_key or os.getenv("JINAAI_APIKEY")
        if not self.api_key:
            raise ValueError(
                "Jina AI API key not found. Provide via api_key parameter "
                "or set JINAAI_APIKEY environment variable."
            )
        
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"
        logger.info(f"Initialized Jina AI embedder with model: {model}")
    
    def _make_api_request(self, input_data: Union[str, List[str]], timeout: int = 30) -> dict:
        """
        Make API request to Jina AI (internal helper method).
        
        Args:
            input_data: Single text or list of texts to embed
            timeout: Request timeout in seconds
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "input": input_data,
            "embedding_type": "float"
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get ColBERT embedding: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise
    
    def _extract_embedding(self, data_item: dict) -> List[List[float]]:
        """
        Extract embedding from API response data item (internal helper method).
        
        Args:
            data_item: Single item from API response data array
            
        Returns:
            Multi-vector embedding
            
        Raises:
            ValueError: If embedding not found in expected format
        """
        embedding = data_item.get('embeddings') or data_item.get('embedding')
        if embedding is None:
            logger.error(f"Unexpected API response format: {list(data_item.keys())}")
            raise ValueError(f"Could not find embedding in response: {list(data_item.keys())}")
        return embedding
    
    def embed_text(self, text: str) -> List[List[float]]:
        """
        Generate ColBERT multi-vector embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Multi-vector embedding as list of lists (shape: [num_tokens, embedding_dim])
        """
        result = self._make_api_request(text)
        
        if 'data' not in result or not result['data']:
            logger.error(f"Invalid API response: {result}")
            raise ValueError("Invalid API response format")
        
        return self._extract_embedding(result['data'][0])
    
    def embed_texts(self, texts: List[str]) -> List[List[List[float]]]:
        """
        Generate multi-vector embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of multi-vector embeddings
        """
        result = self._make_api_request(texts, timeout=60)
        
        if 'data' not in result or not result['data']:
            logger.error(f"Invalid API response: {result}")
            raise ValueError("Invalid API response format")
        
        return [self._extract_embedding(item) for item in result['data']]
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate multi-vector embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            
        Returns:
            List of chunk dictionaries with added 'multi_vector_embedding' key
        """
        # Use shared base implementation with batching and custom key name
        return super().embed_chunks(
            chunks, 
            embedding_key='multi_vector_embedding',
            batch_size=8  # Jina AI has rate limits
        )


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Generates multi-vector embeddings using local HuggingFace models.
    
    This embedder loads a transformer model locally and generates token-level embeddings
    for late interaction. No API keys required - everything runs locally.
    
    Works with any HuggingFace model that outputs token-level embeddings.
    
    Popular ColBERT models:
    - colbert-ir/colbertv2.0 (16M+ downloads, classic ColBERT v2)
    - mixedbread-ai/mxbai-edge-colbert-v0-17m (360K+ downloads, optimized for edge)
    - mixedbread-ai/mxbai-edge-colbert-v0-32m (smaller, faster)
    
    Other models:
    - Any transformer model from HuggingFace that outputs hidden states
    """
    
    def __init__(
        self, 
        model_name: str = "colbert-ir/colbertv2.0",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model_name: HuggingFace model identifier (any transformer model)
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            
        Raises:
            ImportError: If torch or transformers not installed
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "HuggingFaceEmbedder requires torch and transformers. "
                "Install with: pip install torch transformers"
            )
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading HuggingFace model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _encode(self, texts: Union[str, List[str]]) -> List[List[List[float]]]:
        """
        Internal method to encode texts into multi-vector embeddings.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of multi-vector embeddings (batch_size × num_tokens × embedding_dim)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get last hidden state (token-level embeddings)
                embeddings = outputs.last_hidden_state
            
            # Convert to list format and remove padding tokens
            result = []
            for i, embedding in enumerate(embeddings):
                # Get attention mask to identify real tokens (not padding)
                attention_mask = inputs['attention_mask'][i]
                # Only keep embeddings for real tokens
                real_tokens = embedding[attention_mask.bool()]
                # Convert to nested list format
                token_embeddings = real_tokens.cpu().tolist()
                result.append(token_embeddings)
            
            return result
        except Exception as e:
            logger.error(
                f"Failed to encode {len(texts)} texts: {type(e).__name__}: {str(e)[:200]}"
            )
            raise
    
    def embed_text(self, text: str) -> List[List[float]]:
        """
        Generate ColBERT multi-vector embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Multi-vector embedding (num_tokens × embedding_dim)
        """
        embeddings = self._encode(text)
        return embeddings[0]
    
    def embed_texts(self, texts: List[str], batch_size: int = 8) -> List[List[List[float]]]:
        """
        Generate ColBERT multi-vector embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of multi-vector embeddings
        """
        all_embeddings = []
        
        # Process in batches to avoid memory issues
        pbar = tqdm(range(0, len(texts), batch_size), desc="Encoding texts", unit="batch")
        for i in pbar:
            batch = texts[i:i + batch_size]
            batch_embeddings = self._encode(batch)
            all_embeddings.extend(batch_embeddings)
            pbar.set_postfix({"texts": f"{min(i + batch_size, len(texts))}/{len(texts)}"})
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[dict], batch_size: int = 8) -> List[dict]:
        """
        Generate multi-vector embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            batch_size: Batch size for processing
            
        Returns:
            List of chunk dictionaries with added 'multi_vector_embedding' key
        """
        logger.info(f"Generating HuggingFace embeddings for {len(chunks)} chunks")
        
        try:
            # Extract text content
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embed_texts(texts, batch_size=batch_size)
            
            # Add embeddings to chunks
            embedded_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['multi_vector_embedding'] = embedding
                embedded_chunks.append(chunk_with_embedding)
            
            logger.info(f"Successfully generated {len(embedded_chunks)} HuggingFace embeddings")
            
            # Log shape info
            avg_tokens = sum(len(emb) for emb in embeddings) / len(embeddings)
            logger.info(f"Average tokens per chunk: {avg_tokens:.1f}")
            if embeddings and embeddings[0]:
                logger.info(f"Embedding dimensions: {len(embeddings[0][0])}")
            
            return embedded_chunks
        except Exception as e:
            logger.error(
                f"Failed to embed {len(chunks)} chunks: {type(e).__name__}: {str(e)[:200]}"
            )
            raise


class EmbeddingGenerator(BaseEmbedder):
    """Generates embeddings using Ollama via LangChain."""
    # ollama also has a jina model nabed: jina/jina-embeddings-v2-base-en (its not the colbert model)
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "snowflake-arctic-embed:33m"):
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
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            available_models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in available_models]
            
            model_base = model.split(':')[0]
            model_found = model in model_names or any(
                m == model or m.startswith(f"{model_base}:") 
                for m in model_names
            )
            
            if not model_found:
                logger.warning(f"Model '{model}' not found in Ollama")
                logger.warning(f"Available models: {', '.join(model_names) if model_names else 'none'}")
                logger.warning(f"To install: ollama pull {model}")
                logger.warning("Proceeding anyway - model will be pulled on first use")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {base_url}. "
                f"Please ensure Ollama is running. Error: {e}"
            )
        
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=base_url,
                model=model
            )
            logger.info(f"Initialized Ollama embeddings with model: {model}")
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
        # Use shared base implementation without batching
        return super().embed_chunks(chunks, embedding_key='embedding')


def create_embedder(
    embedder_type: str = "single",
    base_url: str = "http://localhost:11434",
    model: str = "snowflake-arctic-embed:33m",
    api_key: Optional[str] = None,
    colbert_model: str = "colbert-ir/colbertv2.0"
) -> Union[EmbeddingGenerator, JinaAIEmbedder, HuggingFaceEmbedder]:
    """
    Factory function to create appropriate embedder.
    
    Args:
        embedder_type: Type of embedder ("single", "jina", or "huggingface")
        base_url: Base URL for Ollama (used for single embedder)
        model: Model name
        api_key: API key for Jina AI - only needed for "jina" type
        colbert_model: HuggingFace model for local embeddings (default: mxbai-edge-colbert-v0-17m)
        
    Returns:
        EmbeddingGenerator, JinaAIEmbedder, or HuggingFaceEmbedder instance
    """
    if embedder_type == "jina":
        return JinaAIEmbedder(api_key=api_key, model="jina-colbert-v2")
    elif embedder_type == "huggingface":
        return HuggingFaceEmbedder(model_name=colbert_model)
    elif embedder_type == "single":
        return EmbeddingGenerator(base_url=base_url, model=model)
    else:
        raise ValueError(f"Unknown embedder_type: {embedder_type}. Use 'single', 'jina', or 'huggingface'.")

