"""
Example script showing how to use the pipeline with custom documents.
"""
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator
from src.weaviate_client import WeaviateClient


def run_with_custom_urls():
    """Run pipeline with custom URLs."""
    
    # Define your custom URLs here
    custom_urls = [
        "https://python.langchain.com/docs/get_started/introduction",
        "https://python.langchain.com/docs/get_started/quickstart",
    ]
    
    print("Running pipeline with custom URLs...")
    
    # Initialize components
    config = Config()
    fetcher = DocumentFetcher()
    extractor = TextExtractor()
    chunker = DocumentChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    embedder = EmbeddingGenerator(base_url=config.OLLAMA_BASE_URL, model=config.OLLAMA_MODEL)
    weaviate_client = WeaviateClient(url=config.WEAVIATE_URL, collection_name=config.COLLECTION_NAME)
    
    try:
        # Fetch custom documents
        docs = fetcher.fetch_custom_docs(custom_urls)
        print(f"Fetched {len(docs)} documents")
        
        # Process documents
        cleaned_docs = extractor.extract_from_documents(docs)
        chunks = chunker.chunk_documents(cleaned_docs)
        embedded_chunks = embedder.embed_chunks(chunks)
        
        # Store in Weaviate
        weaviate_client.connect()
        weaviate_client.create_schema()
        weaviate_client.store_chunks(embedded_chunks)
        
        print("Successfully stored documents in Weaviate!")
        
    finally:
        weaviate_client.close()


def query_example():
    """Example of querying the stored documents."""
    
    print("\nQuerying stored documents...")
    
    config = Config()
    embedder = EmbeddingGenerator(base_url=config.OLLAMA_BASE_URL, model=config.OLLAMA_MODEL)
    weaviate_client = WeaviateClient(url=config.WEAVIATE_URL, collection_name=config.COLLECTION_NAME)
    
    try:
        weaviate_client.connect()
        
        # Create query embedding
        query_text = "What is LangChain?"
        query_embedding = embedder.embed_text(query_text)
        
        # Query Weaviate
        results = weaviate_client.query(query_embedding, limit=5)
        
        print(f"\nQuery: '{query_text}'")
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Title: {result['title']}")
            print(f"  URL: {result['url']}")
            print(f"  Content: {result['content'][:200]}...")
            print()
        
    finally:
        weaviate_client.close()


if __name__ == "__main__":
    # Uncomment the function you want to run
    run_with_custom_urls()
    # query_example()
