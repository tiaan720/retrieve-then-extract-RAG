from typing import List, Union, Optional, Type, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import requests
import os

from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings

from src.utils.logger import logger
from src.config import Config

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def detect_device() -> str:
    """
    Auto-detect the best available device for PyTorch.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding generators.
    
    Provides shared functionality for embedding document chunks.
    Subclasses must implement embed_text() and embed_texts() methods.
    
    Attributes:
        model_name: Name/identifier of the embedding model
        embedding_key: Key used to store embeddings in chunk dictionaries
    """
    
    model_name: str
    embedding_key: str = "embedding"
    
    @abstractmethod
    def _validate_configuration(self) -> None:
        """Validate configuration and credentials. Raise on failure."""
        pass
    
    @abstractmethod
    def _initialize_backend(self) -> None:
        """Initialize the embedding backend (API client, model, etc.)."""
        pass
    
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
        batch_size: Optional[int] = None
    ) -> List[dict]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            batch_size: If provided, process in batches of this size
            
        Returns:
            List of chunk dictionaries with added embedding key
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.__class__.__name__}")
        
        texts = [chunk['content'] for chunk in chunks]
        
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
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding[self.embedding_key] = embedding
            embedded_chunks.append(chunk_with_embedding)
        
        logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")
        self._log_embedding_stats(embeddings)
        
        return embedded_chunks
    
    def _log_embedding_stats(self, embeddings: List) -> None:
        """Log statistics about generated embeddings. Override in subclasses for custom logging."""
        pass


class SingleVectorEmbedder(BaseEmbedder):
    """
    Base class for embedders that produce single dense vectors.
    
    Single-vector embeddings represent an entire text as one fixed-dimension vector.
    Used for traditional dense retrieval with cosine similarity.
    
    Returns: List[float] for single text, List[List[float]] for multiple texts
    """
    
    embedding_key: str = "embedding"
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate single-vector embedding for a text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate single-vector embeddings for multiple texts."""
        pass


class MultiVectorEmbedder(BaseEmbedder):
    """
    Base class for embedders that produce multi-vector (token-level) embeddings.
    
    Multi-vector embeddings produce one vector per token, enabling late interaction
    scoring (e.g., ColBERT's MaxSim). Better for fine-grained semantic matching.
    
    Returns: List[List[float]] for single text, List[List[List[float]]] for multiple texts
    """
    
    embedding_key: str = "multi_vector_embedding"
    
    @abstractmethod
    def embed_text(self, text: str) -> List[List[float]]:
        """Generate multi-vector embedding for a text (num_tokens × embedding_dim)."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[List[float]]]:
        """Generate multi-vector embeddings for multiple texts."""
        pass
    
    def _log_embedding_stats(self, embeddings: List[List[List[float]]]) -> None:
        """Log multi-vector specific statistics."""
        if embeddings:
            avg_tokens = sum(len(emb) for emb in embeddings) / len(embeddings)
            logger.info(f"Average tokens per chunk: {avg_tokens:.1f}")
            if embeddings[0]:
                logger.info(f"Embedding dimensions: {len(embeddings[0][0])}")


class OllamaEmbedder(SingleVectorEmbedder):
    """
    Single-vector embeddings using Ollama via LangChain.
    
    Connects to a local Ollama instance and uses models like snowflake-arctic-embed.
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None, 
        model: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize Ollama embedder.
        
        Args:
            base_url: Ollama API URL (default: from config or http://localhost:11434)
            model: Model name (default: from config or snowflake-arctic-embed:33m)
            config: Configuration object (optional, creates new if not provided)
        """
        self._config = config or Config()
        self.base_url = base_url or self._config.OLLAMA_BASE_URL
        self.model_name = model or self._config.OLLAMA_MODEL
        
        self._validate_configuration()
        self._initialize_backend()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model_name}")
    
    def _validate_configuration(self) -> None:
        """Validate Ollama is reachable and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            available_models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in available_models]
            
            model_base = self.model_name.split(':')[0]
            model_found = self.model_name in model_names or any(
                m == self.model_name or m.startswith(f"{model_base}:") 
                for m in model_names
            )
            
            if not model_found:
                logger.warning(f"Model '{self.model_name}' not found in Ollama")
                logger.warning(f"Available models: {', '.join(model_names) if model_names else 'none'}")
                logger.warning(f"To install: ollama pull {self.model_name}")
                logger.warning("Proceeding anyway - model will be pulled on first use")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running. Error: {e}"
            )
    
    def _initialize_backend(self) -> None:
        """Initialize LangChain Ollama embeddings."""
        try:
            self._embeddings = OllamaEmbeddings(
                base_url=self.base_url,
                model=self.model_name
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OllamaEmbeddings: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self._embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self._embeddings.embed_documents(texts)


EmbeddingGenerator = OllamaEmbedder


class JinaAIEmbedder(MultiVectorEmbedder):
    """
    Multi-vector ColBERT embeddings using Jina AI API.
    
    Uses jina-colbert-v2 model for token-level embeddings with late interaction.
    Requires a Jina AI API key.
    """
    
    API_URL = "https://api.jina.ai/v1/embeddings"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize Jina AI embedder.
        
        Args:
            api_key: Jina AI API key (default: from config/env JINAAI_APIKEY)
            model: Model name (default: jina-colbert-v2)
            config: Configuration object (optional)
        """
        self._config = config or Config()
        self._api_key = api_key or self._config.JINAAI_APIKEY or os.getenv("JINAAI_APIKEY")
        self.model_name = model or self._config.JINA_MODEL
        
        self._validate_configuration()
        self._initialize_backend()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model_name}")
    
    def _validate_configuration(self) -> None:
        """Validate API key is present."""
        if not self._api_key:
            raise ValueError(
                "Jina AI API key not found. Provide via api_key parameter, "
                "config.JINAAI_APIKEY, or JINAAI_APIKEY environment variable."
            )
    
    def _initialize_backend(self) -> None:
        """No initialization needed for API-based embedder."""
        pass
    
    def _make_api_request(self, input_data: Union[str, List[str]], timeout: int = 30) -> dict:
        """Make API request to Jina AI."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "input": input_data,
            "embedding_type": "float"
        }
        
        try:
            response = requests.post(self.API_URL, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina AI API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise
    
    def _extract_embedding(self, data_item: dict) -> List[List[float]]:
        """Extract embedding from API response item."""
        embedding = data_item.get('embeddings') or data_item.get('embedding')
        if embedding is None:
            raise ValueError(f"Could not find embedding in response: {list(data_item.keys())}")
        return embedding
    
    def embed_text(self, text: str) -> List[List[float]]:
        """Generate multi-vector embedding for a single text."""
        result = self._make_api_request(text)
        
        if 'data' not in result or not result['data']:
            raise ValueError("Invalid API response format")
        
        return self._extract_embedding(result['data'][0])
    
    def embed_texts(self, texts: List[str]) -> List[List[List[float]]]:
        """Generate multi-vector embeddings for multiple texts."""
        result = self._make_api_request(texts, timeout=60)
        
        if 'data' not in result or not result['data']:
            raise ValueError("Invalid API response format")
        
        return [self._extract_embedding(item) for item in result['data']]
    
    def embed_chunks(self, chunks: List[dict], batch_size: int = 8) -> List[dict]:
        """Generate embeddings with rate-limit-friendly batching."""
        return super().embed_chunks(chunks, batch_size=batch_size)


class HuggingFaceEmbedder(MultiVectorEmbedder):
    """
    Multi-vector ColBERT embeddings using local HuggingFace models.
    
    Runs locally without API keys. Supports various ColBERT models:
    - colbert-ir/colbertv2.0 (classic ColBERT v2)
    - mixedbread-ai/mxbai-edge-colbert-v0-17m (optimized for edge)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model_name: HuggingFace model identifier (default: from config)
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length (default: from config or 512)
            config: Configuration object (optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "HuggingFaceEmbedder requires torch and transformers. "
                "Install with: pip install torch transformers"
            )
        
        self._config = config or Config()
        self.model_name = model_name or self._config.COLBERT_MODEL
        self.max_length = max_length or self._config.HUGGINGFACE_MAX_LENGTH
        self.device = torch.device(device or detect_device())
        
        self._validate_configuration()
        self._initialize_backend()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model_name} on {self.device}")
    
    def _validate_configuration(self) -> None:
        """Validate torch is available."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers are required")
    
    def _initialize_backend(self) -> None:
        """Load tokenizer and model."""
        logger.info(f"Loading HuggingFace model: {self.model_name}")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _encode(self, texts: Union[str, List[str]]) -> List[List[List[float]]]:
        """Encode texts into multi-vector embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state
            
            result = []
            for i, embedding in enumerate(embeddings):
                attention_mask = inputs['attention_mask'][i]
                real_tokens = embedding[attention_mask.bool()]
                token_embeddings = real_tokens.cpu().tolist()
                result.append(token_embeddings)
            
            return result
        except Exception as e:
            logger.error(f"Failed to encode {len(texts)} texts: {type(e).__name__}: {str(e)[:200]}")
            raise
    
    def embed_text(self, text: str) -> List[List[float]]:
        """Generate multi-vector embedding for a single text."""
        return self._encode(text)[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[List[float]]]:
        """Generate multi-vector embeddings for multiple texts."""
        return self._encode(texts)
    
    def embed_chunks(self, chunks: List[dict], batch_size: int = 8) -> List[dict]:
        """Generate embeddings with memory-friendly batching."""
        return super().embed_chunks(chunks, batch_size=batch_size)


class EmbedderType(Enum):
    """Supported embedder types."""
    OLLAMA = "ollama"
    JINA = "jina"
    HUGGINGFACE = "huggingface"


# Registry mapping embedder types to their classes
EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "ollama": OllamaEmbedder,
    "jina": JinaAIEmbedder,
    "huggingface": HuggingFaceEmbedder,
}


def create_embedder(
    embedder_type: str = "ollama",
    config: Optional[Config] = None,
    **kwargs
) -> BaseEmbedder:
    """
    Factory function to create an embedder instance.
    
    Uses a registry pattern for easy extensibility. All embedders accept
    an optional `config` parameter for centralized configuration.
    
    Args:
        embedder_type: Type of embedder ("ollama", "jina", "huggingface")
        config: Configuration object (optional, embedders create their own if not provided)
        **kwargs: Additional arguments passed to the embedder constructor
            - For ollama: base_url, model
            - For jina: api_key, model
            - For huggingface: model_name, device, max_length
    
    Returns:
        Configured embedder instance
    
    Raises:
        ValueError: If embedder_type is not recognized
    
    Examples:
        >>> embedder = create_embedder("ollama")  # Uses config defaults
        >>> embedder = create_embedder("jina", api_key="your-key")
        >>> embedder = create_embedder("huggingface", model_name="colbert-ir/colbertv2.0")
    """
    embedder_type = embedder_type.lower()
    
    if embedder_type not in EMBEDDER_REGISTRY:
        available = list(EMBEDDER_REGISTRY.keys())
        raise ValueError(f"Unknown embedder_type: '{embedder_type}'. Available: {available}")
    
    embedder_class = EMBEDDER_REGISTRY[embedder_type]
    
    if config is not None:
        kwargs['config'] = config
    
    return embedder_class(**kwargs)


def register_embedder(name: str, embedder_class: Type[BaseEmbedder]) -> None:
    """
    Register a custom embedder class.
    
    Args:
        name: Name to register the embedder under
        embedder_class: Embedder class (must inherit from BaseEmbedder)
    """
    if not issubclass(embedder_class, BaseEmbedder):
        raise TypeError(f"{embedder_class} must inherit from BaseEmbedder")
    EMBEDDER_REGISTRY[name.lower()] = embedder_class
    logger.info(f"Registered custom embedder: {name}")


