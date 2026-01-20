# retrieve-then-extract-RAG

A RAG (Retrieval-Augmented Generation) pipeline that can do ranged queries using Weaviate vector database and Ollama embeddings.

## Features

- **Document Fetching**: Automatically fetches documentation from open-source libraries
- **Text Extraction**: Cleans and preprocesses text content
- **Chunking**: Intelligently splits documents into overlapping chunks
- **Embeddings**: Generates embeddings using Ollama via LangChain
- **Vector Storage**: Stores and retrieves documents using Weaviate

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Ollama installed locally ([Install Ollama](https://ollama.ai/))

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and Pull the Embedding Model

```bash
# Start Ollama (if not already running)
ollama serve

# In another terminal, pull the embedding model
ollama pull nomic-embed-text
```

### 3. Start Weaviate with Docker

```bash
docker-compose up -d
```

This will start a Weaviate instance on `http://localhost:8080`.

### 4. Configure Environment (Optional)

Copy the example environment file and adjust settings if needed:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- Weaviate URL
- Ollama base URL and model
- Chunk size and overlap

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Fetch sample documentation (LangChain docs by default)
2. Extract and clean the text
3. Chunk the documents
4. Generate embeddings using Ollama
5. Store everything in Weaviate
6. Run a test query

### Use Custom Documents

Edit `example_usage.py` to add your own URLs:

```python
custom_urls = [
    "https://your-docs-url.com/page1",
    "https://your-docs-url.com/page2",
]
```

Then run:

```bash
python example_usage.py
```

### Query Example

```python
from src.config import Config
from src.embedder import EmbeddingGenerator
from src.weaviate_client import WeaviateClient

# Initialize
config = Config()
embedder = EmbeddingGenerator(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
weaviate_client = WeaviateClient(config.WEAVIATE_URL)

# Connect and query
weaviate_client.connect()
query_embedding = embedder.embed_text("Your query here")
results = weaviate_client.query(query_embedding, limit=5)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Content: {result['content'][:200]}...")
    print()

weaviate_client.close()
```

## Project Structure

```
.
├── docker-compose.yml          # Weaviate Docker configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment configuration
├── main.py                    # Main pipeline script
├── example_usage.py           # Example usage with custom URLs
└── src/
    ├── config.py              # Configuration settings
    ├── document_fetcher.py    # Document fetching logic
    ├── text_extractor.py      # Text extraction and cleaning
    ├── chunker.py             # Document chunking
    ├── embedder.py            # Embedding generation with Ollama
    └── weaviate_client.py     # Weaviate database client
```

## Configuration

Default settings in `.env`:

- `WEAVIATE_URL`: Weaviate instance URL (default: `http://localhost:8080`)
- `OLLAMA_BASE_URL`: Ollama API URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Embedding model name (default: `nomic-embed-text`)
- `CHUNK_SIZE`: Size of text chunks in characters (default: `500`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `50`)

## Troubleshooting

### Weaviate Connection Issues

Make sure Weaviate is running:
```bash
docker-compose ps
```

Check Weaviate logs:
```bash
docker-compose logs weaviate
```

### Ollama Connection Issues

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

Make sure the embedding model is pulled:
```bash
ollama list
```

### Module Import Errors

Make sure you're running from the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Stopping Services

```bash
# Stop Weaviate
docker-compose down

# Optional: Remove volumes to clear all data
docker-compose down -v
```
