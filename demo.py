"""
Quick start script with mock data to demonstrate the pipeline without external services.
This script demonstrates the pipeline flow without requiring Ollama or Weaviate to be running.
"""
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker


def demo_pipeline():
    """Demonstrate the pipeline with sample data."""
    
    print("=" * 70)
    print("WEAVIATE EMBEDDING PIPELINE - DEMONSTRATION")
    print("=" * 70)
    
    # Sample documents
    sample_docs = [
        {
            "title": "Introduction to RAG",
            "url": "https://example.com/rag-intro",
            "content": """
            Retrieval-Augmented Generation (RAG) is a technique that combines 
            information retrieval with text generation. It works by first retrieving 
            relevant documents from a knowledge base, then using those documents to 
            generate informed responses. This approach helps large language models 
            access up-to-date information and cite sources. RAG systems typically 
            use vector databases like Weaviate to store and retrieve documents 
            efficiently. The retrieval step uses semantic similarity to find the 
            most relevant chunks of text.
            """
        },
        {
            "title": "Vector Databases Explained",
            "url": "https://example.com/vector-db",
            "content": """
            Vector databases store data as high-dimensional vectors, which are 
            numerical representations of content. These embeddings capture the 
            semantic meaning of text, images, or other data types. Vector databases 
            enable fast similarity search using algorithms like HNSW or IVF. 
            Popular vector databases include Weaviate, Pinecone, and Qdrant. 
            They are essential for building modern AI applications like semantic 
            search, recommendation systems, and RAG pipelines.
            """
        },
        {
            "title": "Embeddings and LLMs",
            "url": "https://example.com/embeddings",
            "content": """
            Embeddings are dense vector representations that capture semantic 
            relationships between pieces of text. They are created by neural 
            networks trained on large amounts of data. Modern embedding models 
            like Ollama's nomic-embed-text can generate high-quality embeddings 
            locally. These embeddings are used in various NLP tasks including 
            similarity search, clustering, and classification. The quality of 
            embeddings directly impacts the performance of RAG systems.
            """
        }
    ]
    
    print("\n📚 Step 1: Sample Documents Loaded")
    print(f"   Loaded {len(sample_docs)} documents")
    for doc in sample_docs:
        print(f"   - {doc['title']}")
    
    # Text Extraction
    print("\n🔍 Step 2: Text Extraction & Cleaning")
    extractor = TextExtractor()
    cleaned_docs = extractor.extract_from_documents(sample_docs)
    print(f"   Cleaned {len(cleaned_docs)} documents")
    
    # Document Chunking
    print("\n✂️  Step 3: Document Chunking")
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_documents(cleaned_docs)
    print(f"   Created {len(chunks)} chunks")
    
    print("\n📊 Chunk Details:")
    for i, chunk in enumerate(chunks[:5], 1):  # Show first 5 chunks
        print(f"\n   Chunk {i}:")
        print(f"   - Title: {chunk['title']}")
        print(f"   - Chunk {chunk['chunk_index'] + 1} of {chunk['total_chunks']}")
        print(f"   - Length: {len(chunk['content'])} characters")
        print(f"   - Preview: {chunk['content'][:100]}...")
    
    if len(chunks) > 5:
        print(f"\n   ... and {len(chunks) - 5} more chunks")
    
    # Simulated Embedding
    print("\n🧮 Step 4: Embedding Generation")
    print("   [DEMO MODE: Skipping actual embedding generation]")
    print("   In production, this would:")
    print("   - Connect to Ollama API")
    print("   - Generate embeddings for each chunk using nomic-embed-text")
    print("   - Each embedding would be a vector of ~768 dimensions")
    
    # Simulated Weaviate Storage
    print("\n💾 Step 5: Weaviate Storage")
    print("   [DEMO MODE: Skipping actual database storage]")
    print("   In production, this would:")
    print("   - Connect to Weaviate at http://localhost:8080")
    print("   - Create schema with Document collection")
    print(f"   - Store {len(chunks)} chunks with their embeddings")
    print("   - Enable semantic search over the documents")
    
    # Simulated Query
    print("\n🔎 Step 6: Query Example")
    print("   Query: 'What is RAG?'")
    print("   [DEMO MODE: Showing sample results]")
    print("\n   In production, this would:")
    print("   - Generate embedding for the query")
    print("   - Search Weaviate for similar vectors")
    print("   - Return ranked results\n")
    
    # Show sample chunk that would match
    rag_chunks = [c for c in chunks if 'RAG' in c['content'] or 'Retrieval' in c['content']]
    if rag_chunks:
        print("   Sample matching chunk:")
        print(f"   - Title: {rag_chunks[0]['title']}")
        print(f"   - Content: {rag_chunks[0]['content'][:200]}...")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("   1. Install Ollama: https://ollama.ai/")
    print("   2. Pull embedding model: ollama pull nomic-embed-text")
    print("   3. Start Weaviate: docker-compose up -d")
    print("   4. Run full pipeline: python main.py")
    print("\n   See README.md for detailed instructions.")
    print("=" * 70)


if __name__ == "__main__":
    demo_pipeline()
