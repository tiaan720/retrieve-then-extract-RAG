from typing import List, Dict, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Tokenization
from tqdm import tqdm
from src.utils.logger import logger


class CollectionConfig:
    """Configuration for a Weaviate collection."""
    
    def __init__(
        self,
        name: str,
        enable_binary_quantization: bool = False,
        enable_reranker: bool = False,
        ef_construction: int = 128,
        max_connections: int = 64,
        description: str = ""
    ):
        """
        Initialize collection configuration.
        
        Args:
            name: Collection name
            enable_binary_quantization: Enable binary quantization (32x compression)
            enable_reranker: Enable transformer reranker
            ef_construction: HNSW ef_construction parameter
            max_connections: HNSW max_connections parameter
            description: Human-readable description
        """
        self.name = name
        self.enable_binary_quantization = enable_binary_quantization
        self.enable_reranker = enable_reranker
        self.ef_construction = ef_construction
        self.max_connections = max_connections
        self.description = description


class CollectionManager:
    """Manages multiple Weaviate collections for strategy comparison."""
    
    # Predefined collection configurations
    CONFIGS = {
        "StandardHNSW": CollectionConfig(
            name="Document_StandardHNSW",
            enable_binary_quantization=False,
            enable_reranker=False,
            ef_construction=128,
            max_connections=64,
            description="Standard fp32 HNSW (baseline)"
        ),
        "BinaryQuantized": CollectionConfig(
            name="Document_BinaryQuantized",
            enable_binary_quantization=True,
            enable_reranker=False,
            ef_construction=128,
            max_connections=64,
            description="Binary quantization with HNSW (32x compression)"
        ),
        "HybridSearch": CollectionConfig(
            name="Document_HybridSearch",
            enable_binary_quantization=False,
            enable_reranker=False,
            ef_construction=128,
            max_connections=64,
            description="Standard HNSW for hybrid search (BM25 + vector)"
        ),
        "CrossEncoderRerank": CollectionConfig(
            name="Document_CrossEncoderRerank",
            enable_binary_quantization=False,
            enable_reranker=True,
            ef_construction=128,
            max_connections=64,
            description="HNSW with cross-encoder reranking"
        ),
        "BinaryInt8Staged": CollectionConfig(
            name="Document_BinaryInt8Staged",
            enable_binary_quantization=True,
            enable_reranker=False,
            ef_construction=64,  # Faster build, article's approach
            max_connections=32,  # Lower memory
            description="Article's staged retrieval (binary → int8 → fp32)"
        ),
        "ColBERTMultiVector": CollectionConfig(
            name="Document_ColBERTMultiVector",
            enable_binary_quantization=False,
            enable_reranker=False,
            ef_construction=128,
            max_connections=64,
            description="ColBERT multi-vector embeddings with late interaction"
        ),
    }
    
    def __init__(self, client: weaviate.WeaviateClient, vector_dimensions: int = 384):
        """
        Initialize collection manager.
        
        Args:
            client: Connected Weaviate client
            vector_dimensions: Embedding dimension (must match model)
        """
        self.client = client
        self.vector_dimensions = vector_dimensions
        self.collections = {}
    
    def create_collection(self, config: CollectionConfig) -> None:
        """
        Create a single collection with given configuration.
        
        Args:
            config: Collection configuration
        """
        if self.client.collections.exists(config.name):
            logger.info(f"Collection '{config.name}' already exists")
            return
        
        logger.info(f"Creating collection: {config.name}")
        logger.info(f"  Description: {config.description}")
        logger.info(f"  Binary quantization: {config.enable_binary_quantization}")
        logger.info(f"  Reranker: {config.enable_reranker}")
        
        if config.enable_binary_quantization:
            hnsw_config = Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=config.ef_construction,
                max_connections=config.max_connections,
                vector_cache_max_objects=10000,
                quantizer=Configure.VectorIndex.Quantizer.bq()  # Binary quantization
            )
        else:
            hnsw_config = Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=config.ef_construction,
                max_connections=config.max_connections,
                vector_cache_max_objects=10000,
            )
        
        reranker_config = None
        if config.enable_reranker:
            reranker_config = Configure.Reranker.transformers()
        
        # Determine if this is a multi-vector collection
        is_multi_vector = "ColBERT" in config.name
        
        if is_multi_vector:
            vector_config = Configure.MultiVectors.self_provided(
                name="colbert",  # Named vector
                vector_index_config=hnsw_config,
            )
        else:
            vector_config = Configure.Vectors.self_provided(
                vector_index_config=hnsw_config,
            )
        
        collection = self.client.collections.create(
            name=config.name,
            vector_config=vector_config,
            reranker_config=reranker_config,
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.WORD,
                    index_searchable=True,
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.WORD,
                    index_searchable=True,
                ),
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    index_searchable=False,
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    index_filterable=True,
                ),
                Property(
                    name="total_chunks",
                    data_type=DataType.INT,
                    index_filterable=True,
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    index_filterable=True,
                ),
                Property(
                    name="language",
                    data_type=DataType.TEXT,
                    index_filterable=True,
                ),
            ]
        )
        
        logger.info(f"Created collection: {config.name}")
        self.collections[config.name] = collection
    
    def create_all_collections(self) -> None:
        """Create all predefined collections for strategy comparison."""
        logger.info(f"Creating {len(self.CONFIGS)} collections for strategy comparison")
        
        for strategy_name, config in self.CONFIGS.items():
            self.create_collection(config)
        
        logger.info("All collections created")
    
    def get_collection(self, strategy_name: str):
        """
        Get collection for a strategy.
        
        Args:
            strategy_name: Strategy name (e.g., "StandardHNSW")
            
        Returns:
            Weaviate collection object
        """
        if strategy_name not in self.CONFIGS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        config = self.CONFIGS[strategy_name]
        return self.client.collections.get(config.name)
    
    def store_chunks_in_collection(
        self,
        collection_name: str,
        chunks: List[Dict]
    ) -> None:
        """
        Store chunks in a specific collection.
        
        Args:
            collection_name: Name of the collection
            chunks: List of chunk dictionaries with embeddings
        """
        collection = self.client.collections.get(collection_name)
        
        logger.info(f"Storing {len(chunks)} chunks in {collection_name}")
        
        # Check if this is a multi-vector collection
        is_multi_vector = "ColBERT" in collection_name
        
        # Use context manager with appropriate settings
        if is_multi_vector:
            with collection.batch.fixed_size(batch_size=20) as batch:
                pbar = tqdm(enumerate(chunks), total=len(chunks), desc=f"Storing in {collection_name}", unit="chunk")
                for idx, chunk in pbar:
                    properties = {
                        "content": chunk.get("content", ""),
                        "title": chunk.get("title", ""),
                        "url": chunk.get("url", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "total_chunks": chunk.get("total_chunks", 0),
                        "source": chunk.get("source", "wikipedia"),
                        "language": chunk.get("language", "en"),
                    }
                    
                    # Multi-vector embeddings - wrap in dictionary with named vector
                    multi_vec = chunk.get("multi_vector_embedding", [])
                    vector = {"colbert": multi_vec} 
                    
                    try:
                        batch.add_object(
                            properties=properties,
                            vector=vector
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to add chunk {idx} (title: {properties.get('title', 'N/A')}): "
                            f"{type(e).__name__}: {str(e)[:200]}"
                        )
                        raise
        else:
            # Dynamic batch for single-vector (faster)
            with collection.batch.dynamic() as batch:
                pbar = tqdm(enumerate(chunks), total=len(chunks), desc=f"Storing in {collection_name}", unit="chunk")
                for idx, chunk in pbar:
                    properties = {
                        "content": chunk.get("content", ""),
                        "title": chunk.get("title", ""),
                        "url": chunk.get("url", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "total_chunks": chunk.get("total_chunks", 0),
                        "source": chunk.get("source", "wikipedia"),
                        "language": chunk.get("language", "en"),
                    }
                    
                    vector = chunk.get("embedding", [])
                    
                    try:
                        batch.add_object(
                            properties=properties,
                            vector=vector
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to add chunk {idx} (title: {properties.get('title', 'N/A')}): "
                            f"{type(e).__name__}: {str(e)[:200]}"
                        )
                        raise
        
        logger.info(f"Stored {len(chunks)} chunks in {collection_name}")
    
    def store_chunks_in_all_collections(self, chunks: List[Dict]) -> None:
        """
        Store same chunks in all collections for fair comparison.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
        """
        logger.info(f"Storing {len(chunks)} chunks in all collections")
        
        for strategy_name, config in self.CONFIGS.items():
            self.store_chunks_in_collection(config.name, chunks)
        
        logger.info("All collections populated.")
    
    def delete_collection(self, strategy_name: str) -> None:
        """
        Delete a collection.
        
        Args:
            strategy_name: Strategy name
        """
        if strategy_name not in self.CONFIGS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        config = self.CONFIGS[strategy_name]
        
        if self.client.collections.exists(config.name):
            self.client.collections.delete(config.name)
            logger.info(f"Deleted collection: {config.name}")
    
    def delete_all_collections(self) -> None:
        """Delete all predefined collections."""
        logger.info("Deleting all collections")
        
        for strategy_name in self.CONFIGS.keys():
            try:
                self.delete_collection(strategy_name)
            except Exception as e:
                logger.warning(f"Failed to delete {strategy_name}: {e}")
        
        logger.info("All collections deleted")
    
    def list_collections(self) -> List[str]:
        """List all created collections."""
        existing = []
        for strategy_name, config in self.CONFIGS.items():
            if self.client.collections.exists(config.name):
                existing.append(config.name)
        return existing
