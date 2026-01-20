import sys
from typing import List
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator
from src.collection_manager import CollectionManager
from src.retrieval_strategies import (
    StandardHNSW,
    BinaryQuantized,
    HybridSearch,
    CrossEncoderRerank,
    BinaryInt8Staged
)
from src.evaluator import RetrievalEvaluator
from src.logger import logger
import weaviate


def create_test_queries() -> List[str]:
    """Create test queries for benchmarking."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning neural networks",
        "What are natural language processing applications?",
        "Difference between supervised and unsupervised learning",
        "What is a convolutional neural network?",
        "How do transformers work in NLP?",
        "What is reinforcement learning?",
        "Explain backpropagation in neural networks",
        "What are recurrent neural networks?",
        "How does transfer learning work?",
        "What is computer vision?",
        "Explain gradient descent optimization",
        "What are generative adversarial networks?",
        "How does BERT model work?",
        "What is semantic search?",
        "Explain word embeddings and word2vec",
        "What are attention mechanisms?",
        "How does GPT work?",
        "What is few-shot learning?",
    ]


def main():
    """Run strategy comparison."""
    
    logger.info("=" * 80)
    logger.info("RETRIEVAL STRATEGY COMPARISON")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("\n1. Initializing components...")
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
    
    # Connect to Weaviate
    logger.info("\n2. Connecting to Weaviate...")
    # Use WeaviateClient's connection logic
    from src.weaviate_client import WeaviateClient
    temp_client = WeaviateClient(
        url=config.WEAVIATE_URL,
        collection_name="temp",
        vector_dimensions=config.EMBEDDING_DIMENSIONS
    )
    temp_client.connect()
    client = temp_client.client
    logger.info(f"✓ Connected to Weaviate at {config.WEAVIATE_URL}")
    
    try:
        # Initialize collection manager
        collection_manager = CollectionManager(
            client=client,
            vector_dimensions=config.EMBEDDING_DIMENSIONS
        )
        
        # Create all collections
        logger.info("\n3. Creating collections for each strategy...")
        collection_manager.create_all_collections()
        
        # Fetch and process documents
        logger.info("\n4. Fetching and processing documents...")
        topics = [
            "Artificial intelligence",
            "Machine learning",
            "Natural language processing",
            "Deep learning",
            "Neural network",
            "Computer vision",
            "Reinforcement learning",
            "Transformer (machine learning)",
        ]
        docs = fetcher.fetch_wikipedia_articles(topics, max_docs=10)
        logger.info(f"✓ Fetched {len(docs)} documents")
        
        cleaned_docs = extractor.extract_from_documents(docs)
        logger.info(f"✓ Cleaned {len(cleaned_docs)} documents")
        
        chunks = chunker.chunk_documents(cleaned_docs)
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("\n5. Generating embeddings...")
        embedded_chunks = embedder.embed_chunks(chunks)
        logger.info(f"✓ Generated {len(embedded_chunks)} embeddings")
        
        # Store chunks in all collections
        logger.info("\n6. Storing chunks in all collections...")
        collection_manager.store_chunks_in_all_collections(embedded_chunks)
        
        # Initialize strategies
        logger.info("\n7. Initializing retrieval strategies...")
        strategies = [
            StandardHNSW(collection_manager.get_collection("StandardHNSW")),
            BinaryQuantized(collection_manager.get_collection("BinaryQuantized")),
            HybridSearch(collection_manager.get_collection("HybridSearch"), alpha=0.7),
            CrossEncoderRerank(collection_manager.get_collection("CrossEncoderRerank"), rerank_multiplier=4),
            BinaryInt8Staged(collection_manager.get_collection("BinaryInt8Staged"), rescore_multiplier=4),
        ]
        logger.info(f"✓ Initialized {len(strategies)} strategies")
        
        # Create test queries
        logger.info("\n8. Creating test queries...")
        test_queries = create_test_queries()
        logger.info(f"✓ Created {len(test_queries)} test queries")
        
        # Benchmark all strategies
        logger.info("\n9. Benchmarking strategies...")
        logger.info("   (This may take a few minutes...)")
        evaluator = RetrievalEvaluator(embedder)
        
        evaluator.benchmark_all_strategies(
            strategies=strategies,
            queries=test_queries,
            limit=5,
            warmup_queries=2
        )
        
        # Print comparison table
        logger.info("\n10. Results:")
        evaluator.print_comparison_table()
        
        # Save results
        evaluator.save_results("benchmark_results.json")
        
        # Test a sample query
        logger.info("\n11. Sample query test:")
        sample_query = "What is machine learning?"
        sample_embedding = embedder.embed_text(sample_query)
        
        logger.info(f"Query: '{sample_query}'")
        logger.info("\nTop result from each strategy:")
        for strategy in strategies:
            results, metrics = strategy.search(sample_embedding, sample_query, limit=1)
            if results:
                logger.info(f"  {strategy.name}:")
                logger.info(f"    Title: {results[0]['title']}")
                logger.info(f"    Time: {metrics.total_time_ms:.2f}ms")
        
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON COMPLETE")
        logger.info("=" * 80)
        logger.info("\nKey Findings:")
        logger.info("  - Check benchmark_results.json for detailed metrics")
        logger.info("  - BinaryInt8Staged implements the article's approach")
        logger.info("  - All strategies tested on identical data")
        logger.info("  - Same embedding model used for fair comparison")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.info("\nClosing connections...")
        client.close()
        logger.info("✓ Done")


if __name__ == "__main__":
    main()
