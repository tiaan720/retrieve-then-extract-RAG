import sys
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator, HuggingFaceEmbedder
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
from src.logger import logger


def main():
    """Run strategy comparison."""
    
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


        logger.info("Deleting existing collections for clean benchmark run...")
        collection_manager.delete_all_collections()
        
        collection_manager.create_all_collections()
 
        topics = [
            # Core AI/ML topics (original)
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
        docs = fetcher.fetch_wikipedia_articles(topics, max_docs=2)
        logger.info(f"Fetched {len(docs)} documents")
        
        cleaned_docs = extractor.extract_from_documents(docs)
        logger.info(f"Cleaned {len(cleaned_docs)} documents")
        
        chunks = chunker.chunk_documents(cleaned_docs)
        logger.info(f"Created {len(chunks)} chunks")
   
        embedded_chunks = embedder.embed_chunks(chunks)
        logger.info(f"Generated {len(embedded_chunks)} Ollama embeddings")
        
        # colbert_embedder = HuggingFaceEmbedder(model_name="colbert-ir/colbertv2.0")
        # colbert_chunks = colbert_embedder.embed_chunks(chunks, batch_size=4)
        # logger.info(f"Generated {len(colbert_chunks)} ColBERT embeddings")
        
        # # Jina AI Embedder (API-based - requires API key, using jina-colbert-v2 model)
        # from src.embedder import JinaAIEmbedder
        # colbert_embedder = JinaAIEmbedder(api_key=config.JINAAI_APIKEY, model='jina-colbert-v2')
        # colbert_chunks = colbert_embedder.embed_chunks(chunks)
        
        # # Merge both embedding types into a single chunk list
        # for i, chunk in enumerate(embedded_chunks):
        #     chunk['multi_vector_embedding'] = colbert_chunks[i]['multi_vector_embedding']
        # logger.info(f"Combined embeddings: each chunk has both single and multi-vector embeddings")
        
        collection_manager.store_chunks_in_all_collections(embedded_chunks)
        
        strategies = [
            StandardHNSW(collection_manager.get_collection("StandardHNSW")),
            BinaryQuantized(collection_manager.get_collection("BinaryQuantized")),
            HybridSearch(collection_manager.get_collection("HybridSearch"), alpha=0.7),
            CrossEncoderRerank(collection_manager.get_collection("CrossEncoderRerank"), rerank_multiplier=4),
            BinaryInt8Staged(collection_manager.get_collection("BinaryInt8Staged"), rescore_multiplier=4),
            # ColBERTMultiVector(collection_manager.get_collection("ColBERTMultiVector")),
        ]
        logger.info(f"Initialized {len(strategies)} strategies")
        
        test_queries = get_ground_truth_queries()
        ground_truth = get_ground_truth_map()
        logger.info(f"Loaded {len(test_queries)} queries with ground truth annotations")
     
        logger.info("\nBenchmarking strategies.")
        evaluator = RetrievalEvaluator(
            embedder, 
            ground_truth=ground_truth,
            # colbert_embedder=colbert_embedder
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
        logger.info("\nClosing connections...")
        client.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
