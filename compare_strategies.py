import sys
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import create_embedder
from src.collection_manager import CollectionManager
from src.retrieval_strategies import (
    StandardHNSW,
    BinaryQuantized,
    HybridSearch,
    CrossEncoderRerank,
    BinaryInt8Staged,
    ColBERTMultiVector
)
from src.evaluator import RetrievalEvaluator
from src.ground_truth import get_ground_truth_queries, get_ground_truth_map
from src.utils.logger import logger



ENABLE_MULTI_VECTOR = False

# Topics to fetch from Wikipedia
TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Deep learning",
    "Neural network",
    "Computer vision",
    "Reinforcement learning",
    "Transformer (deep learning architecture)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Generative adversarial network",
    "Long short-term memory",
    "Backpropagation",
    "Gradient descent",
    "Support vector machine",
    "Decision tree",
    "Random forest",
    "K-nearest neighbors algorithm",
    "Naive Bayes classifier",
    "Logistic regression",
    "Linear regression",
    "Word embedding",
    "Word2vec",
    "BERT (language model)",
    "GPT-4",
    "Attention (machine learning)",
    "Sequence-to-sequence",
    "Named entity recognition",
    "Sentiment analysis",
    "Text mining",
    "Algorithm",
    "Data structure",
    "Database",
    "Distributed computing",
    "Cloud computing",
    "Computer programming",
    "Software engineering",
    "Operating system",
    "Statistics",
    "Probability theory",
    "Linear algebra",
    "Calculus",
    "Optimization (mathematics)",
    "Data science",
    "Big data",
    "Data mining",
    "Feature engineering",
    "Dimensionality reduction",
    "Principal component analysis",
    "Clustering",
    "Classification",
]

def build_single_vector_strategies(collection_manager, config: Config):
    """Build list of single-vector retrieval strategies."""
    return [
        StandardHNSW(collection_manager.get_collection("StandardHNSW"), config=config),
        BinaryQuantized(collection_manager.get_collection("BinaryQuantized"), config=config),
        HybridSearch(collection_manager.get_collection("HybridSearch"), config=config),
        CrossEncoderRerank(collection_manager.get_collection("CrossEncoderRerank"), config=config),
        BinaryInt8Staged(collection_manager.get_collection("BinaryInt8Staged"), config=config),
    ]


def build_multi_vector_strategies(collection_manager, config: Config):
    """Build list of multi-vector (ColBERT) retrieval strategies."""
    return [
        ColBERTMultiVector(collection_manager.get_collection("ColBERTMultiVector"), config=config),
    ]


def main():
    """Run strategy comparison."""
    

    config = Config()
    fetcher = DocumentFetcher()
    extractor = TextExtractor()
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    embedder = create_embedder("ollama", config=config)
    

    colbert_embedder = None
    if ENABLE_MULTI_VECTOR:
        # Option A: Local HuggingFace model (no API key needed)
        colbert_embedder = create_embedder("huggingface", config=config)
        
        # Option B: Jina AI API (requires API key in config/env)
        # colbert_embedder = create_embedder("jina", config=config)

    from src.weaviate_client import WeaviateClient
    temp_client = WeaviateClient(
        url=config.WEAVIATE_URL,
        collection_name="temp",
        vector_dimensions=config.EMBEDDING_DIMENSIONS
    )
    temp_client.connect()
    client = temp_client.client
    logger.info(f"Connected to Weaviate at {config.WEAVIATE_URL}")
    
    try:
        collection_manager = CollectionManager(
            client=client,
            vector_dimensions=config.EMBEDDING_DIMENSIONS
        )
        
        exclude_strategies = [] if ENABLE_MULTI_VECTOR else ["ColBERTMultiVector"]
        
        logger.info("Deleting existing collections for clean benchmark run...")
        collection_manager.delete_all_collections(exclude_strategies=exclude_strategies)
        collection_manager.create_all_collections(exclude_strategies=exclude_strategies)
        
        docs = fetcher.fetch_wikipedia_articles(TOPICS, max_docs=2)
        logger.info(f"Fetched {len(docs)} documents")
        
        cleaned_docs = extractor.extract_from_documents(docs)
        logger.info(f"Cleaned {len(cleaned_docs)} documents")
        
        chunks = chunker.chunk_documents(cleaned_docs)
        logger.info(f"Created {len(chunks)} chunks")

        embedded_chunks = embedder.embed_chunks(chunks)
        logger.info(f"Generated {len(embedded_chunks)} single-vector embeddings")
        
        if ENABLE_MULTI_VECTOR and colbert_embedder:
            colbert_chunks = colbert_embedder.embed_chunks(chunks, batch_size=4)
            logger.info(f"Generated {len(colbert_chunks)} multi-vector embeddings")
            
            # Merge both embedding types into chunks
            for i, chunk in enumerate(embedded_chunks):
                chunk['multi_vector_embedding'] = colbert_chunks[i]['multi_vector_embedding']
            logger.info("Merged single and multi-vector embeddings")
        

        collection_manager.store_chunks_in_all_collections(
            embedded_chunks, 
            exclude_strategies=exclude_strategies
        )
        
        strategies = build_single_vector_strategies(collection_manager, config)
        
        if ENABLE_MULTI_VECTOR:
            strategies.extend(build_multi_vector_strategies(collection_manager, config))
        
        logger.info(f"Initialized {len(strategies)} strategies")
        
        test_queries = get_ground_truth_queries()
        ground_truth = get_ground_truth_map()
        logger.info(f"Loaded {len(test_queries)} queries with ground truth annotations")
        
        logger.info("\nBenchmarking strategies.")
        evaluator = RetrievalEvaluator(
            embedder,
            ground_truth=ground_truth,
            multi_vector_embedder=colbert_embedder  
        )
        
        evaluator.benchmark_all_strategies(
            strategies=strategies,
            queries=test_queries,
            limit=5,
            warmup_queries=2
        )
        
        evaluator.print_comparison_table()
        evaluator.save_results("benchmark_results.json")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.info("\nClosing connections.")
        client.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
