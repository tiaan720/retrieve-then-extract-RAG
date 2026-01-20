"""
Configuration settings for the embedding pipeline.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class for the pipeline."""
    
    # Weaviate settings
    WEAVIATE_URL: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    COLLECTION_NAME: str = Field(default="Document", alias="COLLECTION_NAME")
    
    # Ollama settings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="snowflake-arctic-embed:33m", alias="OLLAMA_MODEL")
    EMBEDDING_DIMENSIONS: int = Field(default=384, alias="EMBEDDING_DIMENSIONS")
    
    # Chunking settings
    CHUNK_SIZE: int = Field(default=500, alias="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=50, alias="CHUNK_OVERLAP")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        use_enum_values=True,
        extra="ignore"
    )

