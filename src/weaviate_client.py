"""
Weaviate client module for storing and retrieving documents.
"""
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Rerank
from typing import List, Dict, Optional
from urllib.parse import urlparse
import time
from src.logger import logger


class WeaviateClient:
    """Client for interacting with Weaviate vector database."""
    
    def __init__(
        self, 
        url: str = "http://localhost:8080", 
        collection_name: str = "Document", 
        vector_dimensions: int = 384,
        enable_binary_quantization: bool = False
    ):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            collection_name: Name of the collection to use
            vector_dimensions: Dimension of the embedding vectors (must match embedding model)
            enable_binary_quantization: Enable binary quantization for 32x memory reduction and faster search
        """
        self.url = url
        self.collection_name = collection_name
        self.vector_dimensions = vector_dimensions
        self.enable_binary_quantization = enable_binary_quantization
        self.client = None
        
    def connect(self, max_retries: int = 5, retry_delay: int = 2):
        """
        Connect to Weaviate instance with retry logic.
        
        Args:
            max_retries: Maximum number of connection retries
            retry_delay: Delay between retries in seconds
        """
        for attempt in range(max_retries):
            try:
                # Parse URL to extract host and port
                parsed = urlparse(self.url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                
                self.client = weaviate.connect_to_local(host=host, port=port)
                logger.info(f"Connected to Weaviate at {self.url}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    logger.info(f"Retrying in {retry_delay} seconds")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to connect to Weaviate after {max_retries} attempts: {e}")
    
    def create_schema(self):
        """
        Create the collection schema in Weaviate.
        
        Schema includes:
        - Vector index (HNSW for fast similarity search)
        - Optional binary quantization (32x memory reduction, faster search)
        - Text properties with tokenization for hybrid search
        - Metadata for document tracking
        - No auto-vectorizer (we provide embeddings via Ollama)
        """
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            # Check if collection already exists
            if self.client.collections.exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            # Configure vector index based on binary quantization setting
            if self.enable_binary_quantization:
                # Binary quantization: 32x memory reduction, faster search
                # Compresses float32 vectors to 1-bit per dimension
                # Minimal accuracy loss for most use cases
                vector_index_config = Configure.VectorIndex.hnsw(
                    distance_metric="cosine",
                    ef_construction=128,
                    max_connections=64,
                    vector_cache_max_objects=10000,
                    quantizer=Configure.VectorIndex.Quantizer.bq()  # Binary quantization
                )
                logger.info("Enabling binary quantization (32x memory reduction)")
            else:
                # Standard HNSW configuration (no quantization)
                vector_index_config = Configure.VectorIndex.hnsw(
                    distance_metric="cosine",
                    ef_construction=128,
                    max_connections=64,
                    vector_cache_max_objects=10000,
                )
            
            # Create collection with enhanced configuration
            self.client.collections.create(
                name=self.collection_name,
                # No vectorizer - we provide embeddings externally via Ollama
                vectorizer_config=Configure.Vectorizer.none(),
                # Configure vector index for similarity search (HNSW algorithm)
                # Vector dimensions MUST match the embedding model dimensions
                vector_index_config=vector_index_config,
                # Explicitly set vector dimensions to match embedding model
                # For snowflake-arctic-embed:33m = 384 dimensions
                # For nomic-embed-text = 768 dimensions
                vector_index_type="hnsw",
                properties=[
                    # Main content - tokenized for hybrid search (BM25 + vector)
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        tokenization="word",  # Enable BM25 keyword search
                        index_searchable=True,
                    ),
                    # Document metadata
                    Property(
                        name="title",
                        data_type=DataType.TEXT,
                        tokenization="word",
                        index_searchable=True,
                    ),
                    Property(
                        name="url",
                        data_type=DataType.TEXT,
                        index_searchable=False,  # URL doesn't need full-text search
                    ),
                    # Chunk tracking - chunk_index is this chunk's position (0-based)
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        index_filterable=True,  # Enable filtering by chunk position
                    ),
                    # total_chunks - total number of chunks the source document was split into
                    # Useful for reconstruction and context awareness
                    Property(
                        name="total_chunks",
                        data_type=DataType.INT,
                        index_filterable=True,
                    ),
                    # Additional metadata for better retrieval
                    Property(
                        name="source",
                        data_type=DataType.TEXT,
                        description="Source type (e.g., 'wikipedia')",
                        index_filterable=True,
                    ),
                    Property(
                        name="language",
                        data_type=DataType.TEXT,
                        description="Document language code",
                        index_filterable=True,
                    ),
                ]
            )
            bq_status = "with binary quantization" if self.enable_binary_quantization else "without binary quantization"
            logger.info(f"Created collection '{self.collection_name}' with hybrid search support {bq_status}")
            
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise
    
    def delete_collection(self):
        """Delete the collection if it exists."""
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def store_chunks(self, chunks: List[Dict]):
        """
        Store document chunks in Weaviate.
        
        Args:
            chunks: List of chunk dictionaries with 'content', 'embedding', and metadata
        """
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            logger.info(f"Storing {len(chunks)} chunks in Weaviate")
            
            # Batch insert for efficiency
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
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
                    
                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def query(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """
        Query Weaviate for similar documents using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of similar documents with metadata
        """
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "title": obj.properties.get("title", ""),
                    "url": obj.properties.get("url", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "total_chunks": obj.properties.get("total_chunks", 0),
                    "source": obj.properties.get("source", ""),
                    "language": obj.properties.get("language", ""),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying: {e}")
            raise
    
    def hybrid_query(self, query_text: str, query_vector: List[float], limit: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and BM25 keyword search.
        
        Args:
            query_text: Text query for BM25 keyword search
            query_vector: Query embedding vector for semantic search
            limit: Maximum number of results
            alpha: Balance between vector (1.0) and keyword (0.0) search
                  0.5 = equal weight, 1.0 = pure vector, 0.0 = pure keyword
            
        Returns:
            List of similar documents with metadata
        """
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=limit,
                alpha=alpha,  # Fusion algorithm weight
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "title": obj.properties.get("title", ""),
                    "url": obj.properties.get("url", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "total_chunks": obj.properties.get("total_chunks", 0),
                    "source": obj.properties.get("source", ""),
                    "language": obj.properties.get("language", ""),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid query: {e}")
            raise
    
    def rerank_query(
        self, 
        query_text: str, 
        query_vector: List[float], 
        limit: int = 5, 
        rerank_limit: int = 100,
        rerank_property: str = "content"
    ) -> List[Dict]:
        """
        Perform search with reranking for improved relevance.
        
        This is a two-stage retrieval:
        1. Initial retrieval: Get top N candidates using hybrid search
        2. Reranking: Apply cross-encoder model to rerank candidates for better relevance
        
        Args:
            query_text: Text query for initial search
            query_vector: Query embedding vector for semantic search
            limit: Final number of results to return after reranking
            rerank_limit: Number of candidates to retrieve before reranking (higher = better but slower)
            rerank_property: Property to use for reranking (default: "content")
            
        Returns:
            List of reranked documents with metadata and scores
            
        Note:
            Weaviate uses Cohere reranker by default. For local reranking, consider:
            - Cross-encoder models (e.g., ms-marco-MiniLM-L-12-v2)
            - Can be configured via Weaviate's reranker-transformers module
        """
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Perform hybrid search with reranking
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=limit,
                # Rerank using cross-encoder for improved relevance
                # First retrieves rerank_limit candidates, then reranks and returns top 'limit'
                rerank=Rerank(
                    prop=rerank_property,
                    query=query_text
                ),
                # Get more candidates for reranking to improve final results
                # The reranker will select the best 'limit' from these candidates
                return_metadata=["score"]
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "content": obj.properties.get("content", ""),
                    "title": obj.properties.get("title", ""),
                    "url": obj.properties.get("url", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "total_chunks": obj.properties.get("total_chunks", 0),
                    "source": obj.properties.get("source", ""),
                    "language": obj.properties.get("language", ""),
                }
                # Add rerank score if available
                if hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                    result["rerank_score"] = obj.metadata.score
                results.append(result)
            
            logger.info(f"Rerank query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in rerank query: {e}")
            # Fall back to hybrid search without reranking
            logger.warning("Falling back to hybrid search without reranking")
            return self.hybrid_query(query_text, query_vector, limit=limit)
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")
