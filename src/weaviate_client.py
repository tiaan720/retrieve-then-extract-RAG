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
        """Create the collection schema in Weaviate."""
        if not self.client:
            raise Exception("Client not connected. Call connect() first.")
        
        try:
            # Check if collection already exists
            if self.client.collections.exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create collection with properties
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="total_chunks", data_type=DataType.INT),
                ]
            )
            logger.info(f"Created collection '{self.collection_name}'")
            
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
        Query Weaviate for similar documents.
        
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
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying: {e}")
            raise
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")
