"""
Main pipeline orchestration script.
"""
import sys
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator
from src.weaviate_client import WeaviateClient
from src.logger import logger


def main():
    """Run the complete embedding pipeline."""
    
    logger.info("=" * 60)
    logger.info("Starting Weaviate Embedding Pipeline")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components")
    config = Config()
    fetcher = DocumentFetcher()
    extractor = TextExtractor()
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    embedder = EmbeddingGenerator(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL
    )
    weaviate_client = WeaviateClient(
        url=config.WEAVIATE_URL,
        collection_name=config.COLLECTION_NAME
    )
    
    try:
        # Fetch documents
        logger.info("Fetching documents from Wikipedia")
        topics = [
            "Artificial intelligence",
            "Machine learning",
            "Natural language processing",
            "Deep learning",
            "Neural network"
        ]
        docs = fetcher.fetch_wikipedia_articles(topics, max_docs=5)
        logger.info(f"Fetched {len(docs)} documents")
        
        if not docs:
            logger.warning("No documents fetched. Exiting")
            return
        
        # Extract and clean text
        logger.info("Extracting and cleaning text")
        cleaned_docs = extractor.extract_from_documents(docs)
        logger.info(f"Cleaned {len(cleaned_docs)} documents")
        
        # Chunk documents
        logger.info("Chunking documents")
        chunks = chunker.chunk_documents(cleaned_docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings")
        embedded_chunks = embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Connect to Weaviate
        logger.info("Connecting to Weaviate")
        weaviate_client.connect()
        
        # Create schema
        logger.info("Creating/verifying schema")
        weaviate_client.create_schema()
        
        # Store chunks
        logger.info("Storing chunks in Weaviate")
        weaviate_client.store_chunks(embedded_chunks)
        
        # Test query
        logger.info("Testing query")
        if embedded_chunks:
            test_embedding = embedded_chunks[0]['embedding']
            results = weaviate_client.query(test_embedding, limit=3)
            logger.info(f"Test query returned {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"Result {i}: {result['title']}")
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.info("Closing connections")
        weaviate_client.close()


if __name__ == "__main__":
    main()

