# retrieve-then-extract-RAG

A RAG (Retrieval-Augmented Generation) pipeline that can do ranged queries using Weaviate vector database and Ollama embeddings.

## Features

- **Document Fetching**: Fetches Wikipedia articles for RAG testing
- **Text Extraction**: Cleans and preprocesses text content using LangChain approach
- **Chunking**: Intelligently splits documents into overlapping chunks
- **Embeddings**: Generates embeddings using Ollama via LangChain
- **Vector Storage**: Stores and retrieves documents using Weaviate

## Pipeline Architecture

The pipeline follows a sequential flow:

```
┌─────────────────────┐
│  Document Fetcher   │  Fetch articles from Wikipedia
│  (document_fetcher) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Text Extractor     │  Clean and normalize text
│  (text_extractor)   │  (LangChain approach)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Document Chunker   │  Split into overlapping chunks
│  (chunker)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Embedding Gen.     │  Generate vectors via Ollama
│  (embedder)         │  using nomic-embed-text
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Weaviate Client    │  Store chunks + embeddings
│  (weaviate_client)  │  in vector database
└─────────────────────┘
           │
           ▼
    [Query Interface]
```

**Key Components:**

1. **DocumentFetcher**: Retrieves articles from Wikipedia using the wikipedia-py library
2. **TextExtractor**: Cleans and normalizes raw text content using LangChain-style processing
3. **DocumentChunker**: Splits documents with configurable chunk size and overlap
4. **EmbeddingGenerator**: Uses LangChain's OllamaEmbeddings for flexibility
5. **WeaviateClient**: Manages vector database operations with retry logic

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Ollama installed locally ([Install Ollama](https://ollama.ai/))

## Quick Start

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
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

### Run Component Tests

Verify the core components are working:

```bash
python test_components.py
```

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Fetch Wikipedia articles on AI topics
2. Extract and clean the text
3. Chunk the documents
4. Generate embeddings using Ollama
5. Store everything in Weaviate
6. Run a test query

## Project Structure

```
.
├── docker-compose.yml          # Weaviate Docker configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment configuration
├── main.py                    # Main pipeline script
├── test_components.py         # Component tests
└── src/
    ├── config.py              # Configuration settings
    ├── document_fetcher.py    # Document fetching from Wikipedia
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

