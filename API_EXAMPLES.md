# API Examples

This document provides detailed examples of how to use each component of the pipeline.

## 1. Document Fetching

### Fetch from LangChain Documentation

```python
from src.document_fetcher import DocumentFetcher

fetcher = DocumentFetcher()
docs = fetcher.fetch_langchain_docs(max_docs=5)

for doc in docs:
    print(f"Title: {doc['title']}")
    print(f"URL: {doc['url']}")
    print(f"Content length: {len(doc['content'])} chars\n")
```

### Fetch Custom URLs

```python
from src.document_fetcher import DocumentFetcher

fetcher = DocumentFetcher()
custom_urls = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://docs.python.org/3/library/index.html",
]

docs = fetcher.fetch_custom_docs(custom_urls)
print(f"Fetched {len(docs)} documents")
```

## 2. Text Extraction

### Clean Single Document

```python
from src.text_extractor import TextExtractor

extractor = TextExtractor()

# Clean raw text
raw_text = "This  has   extra    spaces\n\n\n\nand newlines"
clean_text = extractor.extract_and_clean(raw_text)
print(clean_text)  # "This has extra spaces\n\nand newlines"
```

### Clean Multiple Documents

```python
from src.text_extractor import TextExtractor

extractor = TextExtractor()
docs = [...]  # List of document dictionaries

cleaned_docs = extractor.extract_from_documents(docs)
```

## 3. Document Chunking

### Basic Chunking

```python
from src.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

# Chunk text directly
text = "Your long document text here..."
chunks = chunker.chunk_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk)} chars")
```

### Chunk Documents with Metadata

```python
from src.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

doc = {
    'title': 'My Document',
    'url': 'https://example.com/doc',
    'content': 'Long document content...'
}

chunks = chunker.chunk_document(doc)

for chunk in chunks:
    print(f"Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}")
    print(f"Title: {chunk['title']}")
    print(f"Content: {chunk['content'][:100]}...\n")
```

### Chunk Multiple Documents

```python
from src.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
docs = [...]  # List of documents

all_chunks = chunker.chunk_documents(docs)
print(f"Total chunks: {len(all_chunks)}")
```

## 4. Embedding Generation

### Initialize with Ollama

```python
from src.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)
```

### Embed Single Text

```python
from src.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()

text = "This is a sample text to embed"
embedding = embedder.embed_text(text)

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Embed Multiple Texts

```python
from src.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()

texts = [
    "First document",
    "Second document",
    "Third document"
]

embeddings = embedder.embed_texts(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Embed Document Chunks

```python
from src.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()
chunks = [...]  # List of chunk dictionaries

embedded_chunks = embedder.embed_chunks(chunks)

# Each chunk now has an 'embedding' key
for chunk in embedded_chunks:
    print(f"Chunk: {chunk['title']}")
    print(f"Embedding dimensions: {len(chunk['embedding'])}")
```

## 5. Weaviate Operations

### Connect to Weaviate

```python
from src.weaviate_client import WeaviateClient

client = WeaviateClient(
    url="http://localhost:8080",
    collection_name="Document"
)

client.connect()
```

### Create Schema

```python
from src.weaviate_client import WeaviateClient

client = WeaviateClient()
client.connect()
client.create_schema()
```

### Store Chunks

```python
from src.weaviate_client import WeaviateClient

client = WeaviateClient()
client.connect()
client.create_schema()

chunks = [...]  # List of embedded chunks
client.store_chunks(chunks)
```

### Query Similar Documents

```python
from src.weaviate_client import WeaviateClient
from src.embedder import EmbeddingGenerator

# Initialize
embedder = EmbeddingGenerator()
client = WeaviateClient()
client.connect()

# Create query embedding
query_text = "What is machine learning?"
query_embedding = embedder.embed_text(query_text)

# Search
results = client.query(query_embedding, limit=5)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content: {result['content'][:200]}...")
```

### Delete Collection

```python
from src.weaviate_client import WeaviateClient

client = WeaviateClient()
client.connect()
client.delete_collection()
```

### Close Connection

```python
client.close()
```

## 6. Complete Pipeline Example

```python
from src.config import Config
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator
from src.weaviate_client import WeaviateClient

# Initialize
config = Config()
fetcher = DocumentFetcher()
extractor = TextExtractor()
chunker = DocumentChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
embedder = EmbeddingGenerator(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
weaviate = WeaviateClient(config.WEAVIATE_URL, config.COLLECTION_NAME)

# Fetch documents
docs = fetcher.fetch_langchain_docs(max_docs=3)

# Process documents
cleaned_docs = extractor.extract_from_documents(docs)
chunks = chunker.chunk_documents(cleaned_docs)
embedded_chunks = embedder.embed_chunks(chunks)

# Store in Weaviate
weaviate.connect()
weaviate.create_schema()
weaviate.store_chunks(embedded_chunks)

# Query
query_text = "How does RAG work?"
query_embedding = embedder.embed_text(query_text)
results = weaviate.query(query_embedding, limit=3)

for result in results:
    print(f"\nTitle: {result['title']}")
    print(f"Content: {result['content'][:200]}...")

# Cleanup
weaviate.close()
```

## 7. Configuration Examples

### Using Environment Variables

Create a `.env` file:

```env
WEAVIATE_URL=http://localhost:8080
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Then use the config:

```python
from src.config import Config

config = Config()
print(f"Weaviate URL: {config.WEAVIATE_URL}")
print(f"Chunk size: {config.CHUNK_SIZE}")
```

### Custom Configuration

```python
# Override defaults
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator

# Custom chunk sizes
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

# Different embedding model
embedder = EmbeddingGenerator(
    base_url="http://localhost:11434",
    model="mxbai-embed-large"  # Use a different model
)
```

## 8. Error Handling

### Retry Logic Example

```python
from src.weaviate_client import WeaviateClient
import time

client = WeaviateClient()

# Connection with custom retry
try:
    client.connect(max_retries=10, retry_delay=3)
    print("Connected successfully")
except Exception as e:
    print(f"Failed to connect: {e}")
```

### Safe Document Fetching

```python
from src.document_fetcher import DocumentFetcher

fetcher = DocumentFetcher()
urls = ["http://url1.com", "http://url2.com", "http://invalid.com"]

docs = fetcher.fetch_custom_docs(urls)
# Automatically skips URLs that fail
print(f"Successfully fetched {len(docs)} out of {len(urls)} URLs")
```

## 9. Batch Processing

### Process Documents in Batches

```python
from src.document_fetcher import DocumentFetcher
from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker
from src.embedder import EmbeddingGenerator
from src.weaviate_client import WeaviateClient

def process_batch(urls, batch_size=10):
    """Process URLs in batches."""
    
    fetcher = DocumentFetcher()
    extractor = TextExtractor()
    chunker = DocumentChunker()
    embedder = EmbeddingGenerator()
    weaviate = WeaviateClient()
    
    weaviate.connect()
    weaviate.create_schema()
    
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}")
        
        # Fetch and process
        docs = fetcher.fetch_custom_docs(batch_urls)
        cleaned_docs = extractor.extract_from_documents(docs)
        chunks = chunker.chunk_documents(cleaned_docs)
        embedded_chunks = embedder.embed_chunks(chunks)
        
        # Store
        weaviate.store_chunks(embedded_chunks)
        print(f"Stored {len(embedded_chunks)} chunks")
    
    weaviate.close()

# Usage
urls = [...]  # List of many URLs
process_batch(urls, batch_size=5)
```

## 10. Advanced Querying

### Multi-Query Example

```python
from src.embedder import EmbeddingGenerator
from src.weaviate_client import WeaviateClient

embedder = EmbeddingGenerator()
weaviate = WeaviateClient()
weaviate.connect()

queries = [
    "What is machine learning?",
    "How does neural network work?",
    "Explain gradient descent"
]

for query in queries:
    print(f"\nQuery: {query}")
    query_embedding = embedder.embed_text(query)
    results = weaviate.query(query_embedding, limit=3)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title']}")

weaviate.close()
```
