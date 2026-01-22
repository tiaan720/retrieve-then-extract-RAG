from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class for the pipeline."""
    
    # Weaviate settings
    WEAVIATE_URL: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    COLLECTION_NAME: str = Field(default="Document", alias="COLLECTION_NAME")
    ENABLE_BINARY_QUANTIZATION: bool = Field(default=False, alias="ENABLE_BINARY_QUANTIZATION")
    
    # Ollama settings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="snowflake-arctic-embed:33m", alias="OLLAMA_MODEL")
    EMBEDDING_DIMENSIONS: int = Field(default=384, alias="EMBEDDING_DIMENSIONS")
    
    # Jina AI settings (for ColBERT multi-vector embeddings)
    JINAAI_APIKEY: str = Field(default="", alias="JINAAI_APIKEY")
    JINA_MODEL: str = Field(default="jina-colbert-v2", alias="JINA_MODEL")
    
    # HuggingFace settings (for local ColBERT embeddings)
    COLBERT_MODEL: str = Field(default="colbert-ir/colbertv2.0", alias="COLBERT_MODEL")
    HUGGINGFACE_MAX_LENGTH: int = Field(default=512, alias="HUGGINGFACE_MAX_LENGTH")
    
    # Chunking settings
    CHUNK_SIZE: int = Field(default=500, alias="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=50, alias="CHUNK_OVERLAP")
    
    # Retrieval strategy defaults
    HYBRID_ALPHA: float = Field(default=0.7, alias="HYBRID_ALPHA")
    RERANK_MULTIPLIER: int = Field(default=4, alias="RERANK_MULTIPLIER")
    RESCORE_MULTIPLIER: int = Field(default=4, alias="RESCORE_MULTIPLIER")
    DEFAULT_SEARCH_LIMIT: int = Field(default=5, alias="DEFAULT_SEARCH_LIMIT")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        use_enum_values=True,
        extra="ignore"
    )

