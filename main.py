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


def main():
    """Run the complete embedding pipeline."""
    
    print("=" * 60)
    print("Starting Weaviate Embedding Pipeline")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
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
        print("\n2. Fetching documents...")
        # Using sample documentation - can be replaced with custom URLs
        docs = fetcher.fetch_langchain_docs(max_docs=5)
        print(f"Fetched {len(docs)} documents")
        
        if not docs:
            print("No documents fetched. Exiting.")
            return
        
        # Extract and clean text
        print("\n3. Extracting and cleaning text...")
        cleaned_docs = extractor.extract_from_documents(docs)
        print(f"Cleaned {len(cleaned_docs)} documents")
        
        # Chunk documents
        print("\n4. Chunking documents...")
        chunks = chunker.chunk_documents(cleaned_docs)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("\n5. Generating embeddings...")
        embedded_chunks = embedder.embed_chunks(chunks)
        print(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Connect to Weaviate
        print("\n6. Connecting to Weaviate...")
        weaviate_client.connect()
        
        # Create schema
        print("\n7. Creating/verifying schema...")
        weaviate_client.create_schema()
        
        # Store chunks
        print("\n8. Storing chunks in Weaviate...")
        weaviate_client.store_chunks(embedded_chunks)
        
        # Test query
        print("\n9. Testing query...")
        if embedded_chunks:
            test_embedding = embedded_chunks[0]['embedding']
            results = weaviate_client.query(test_embedding, limit=3)
            print(f"\nTest query returned {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Title: {result['title']}")
                print(f"    URL: {result['url']}")
                print(f"    Content preview: {result['content'][:100]}...")
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up
        print("\n10. Closing connections...")
        weaviate_client.close()


if __name__ == "__main__":
    main()
