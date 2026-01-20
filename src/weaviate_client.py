"""
Weaviate client module for storing and retrieving documents.
"""
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from typing import List, Dict
from urllib.parse import urlparse
import time
from src.logger import logger


class WeaviateClient:
    """Client for interacting with Weaviate vector database."""
    
    def __init__(self, url: str = "http://localhost:8080", collection_name: str = "Document"):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            collection_name: Name of the collection to use
        """
        self.url = url
        self.collection_name = collection_name
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
            
            # Create collection with enhanced configuration
            self.client.collections.create(
                name=self.collection_name,
                # No vectorizer - we provide embeddings externally via Ollama
                vectorizer_config=Configure.Vectorizer.none(),
                # Configure vector index for similarity search (HNSW algorithm)
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric="cosine",  # Cosine similarity for embeddings
                    ef_construction=128,  # Higher = better recall, slower indexing
                    max_connections=64,  # Higher = better recall, more memory
                ),
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
            logger.info(f"Created collection '{self.collection_name}' with hybrid search support")
            
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
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")
