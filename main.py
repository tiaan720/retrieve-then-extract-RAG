import sys
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import create_embedder
from src.weaviate_client import WeaviateClient
from src.logger import logger


def main():
    """Run the complete embedding pipeline."""
    

    logger.info("Initializing components")
    config = Config()
    fetcher = DocumentFetcher()
    extractor = TextExtractor()
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Use factory to create embedder - config defaults are used automatically
    embedder = create_embedder("ollama", config=config)
    
    weaviate_client = WeaviateClient(
        url=config.WEAVIATE_URL,
        collection_name=config.COLLECTION_NAME,
        vector_dimensions=config.EMBEDDING_DIMENSIONS,
        enable_binary_quantization=config.ENABLE_BINARY_QUANTIZATION
    )
    
    try:
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
        
        logger.info("Extracting and cleaning text")
        cleaned_docs = extractor.extract_from_documents(docs)
        logger.info(f"Cleaned {len(cleaned_docs)} documents")
        
        logger.info("Chunking documents")
        chunks = chunker.chunk_documents(cleaned_docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        logger.info("Generating embeddings")
        embedded_chunks = embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        logger.info("Connecting to Weaviate")
        weaviate_client.connect()
        
        logger.info("Creating/verifying schema")
        weaviate_client.create_schema()
        
        logger.info("Storing chunks in Weaviate")
        weaviate_client.store_chunks(embedded_chunks)
        
        logger.info("Testing query")
        if embedded_chunks:
            test_embedding = embedded_chunks[0]['embedding']
            
            logger.info("Standard vector query:")
            results = weaviate_client.query(test_embedding, limit=3)
            logger.info(f"Test query returned {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"  Result {i}: {result['title']}")
            
            logger.info("Hybrid query (BM25 + vector):")
            test_query_text = "What is machine learning?"
            hybrid_results = weaviate_client.hybrid_query(
                query_text=test_query_text,
                query_vector=test_embedding,
                limit=3,
                alpha=0.7  # 70% vector, 30% keyword
            )
            logger.info(f"Hybrid query returned {len(hybrid_results)} results")
            for i, result in enumerate(hybrid_results, 1):
                logger.info(f"  Result {i}: {result['title']}")
            
            logger.info("Reranked query (with cross-encoder):")
            try:
                reranked_results = weaviate_client.rerank_query(
                    query_text=test_query_text,
                    query_vector=test_embedding,
                    limit=3,
                    rerank_limit=10  # Retrieve 10, rerank to top 3
                )
                logger.info(f"Rerank query returned {len(reranked_results)} results")
                for i, result in enumerate(reranked_results, 1):
                    score_info = f" (score: {result.get('rerank_score', 'N/A')})" if 'rerank_score' in result else ""
                    logger.info(f"  Result {i}: {result['title']}{score_info}")
            except Exception as e:
                logger.warning(f"Reranking not available: {e}")
        
        
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

