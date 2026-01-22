# retrieve-then-extract-RAG

A benchmarking framework to compare different retrieval strategies in Weaviate, helping determine the fastest and most accurate approach for RAG pipelines.

## Retrieval Strategies

The framework benchmarks 6 retrieval strategies:

| Strategy | Description |
|----------|-------------|
| **StandardHNSW** | Baseline fp32 HNSW vector search |
| **BinaryQuantized** | 32x memory reduction with binary quantization |
| **HybridSearch** | BM25 + vector similarity combined |
| **CrossEncoderRerank** | Two-stage retrieval with cross-encoder reranking |
| **BinaryInt8Staged** | Binary filter → Int8 rescore staged retrieval |
| **ColBERTMultiVector** | ColBERT late interaction scoring (requires Jina API key) |

## Prerequisites

- Python 3.11+
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
ollama serve

ollama pull snowflake-arctic-embed:33m
```

### 3. Start Weaviate with Docker

```bash
docker-compose up -d
```

This will start a Weaviate instance on `http://localhost:8080`.

## Usage

### Compare Retrieval Strategies (Main Purpose)

```bash
uv run python compare_strategies.py
```

This will:
1. Fetch  Wikipedia articles on AI/ML topics
2. Create separate Weaviate collections for each strategy
3. Benchmark all strategies against ground truth queries
4. Output a comparison table with latency, precision, and recall metrics
5. Save detailed results to `benchmark_results.json`

### Run Basic Pipeline Demo

```bash
uv run python main.py
```

Runs a simple demo that fetches 5 articles, stores them in Weaviate, and executes test queries.

### Run Tests

```bash
uv run pytest
```

## Configuration

Create a `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate instance URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `snowflake-arctic-embed:33m` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `384` | Vector dimensions |
| `CHUNK_SIZE` | `500` | Chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `HYBRID_ALPHA` | `0.7` | Hybrid search balance (1.0=vector, 0.0=keyword) |
| `JINA_API_KEY` | `""` | Required for ColBERT strategy |

## Project Structure

```
├── compare_strategies.py   # Main benchmarking script
├── main.py                 # Simple pipeline demo
├── benchmark_results.json  # Benchmark output
├── docker-compose.yml      # Weaviate Docker setup
└── src/
    ├── retrieval_strategies.py  # Strategy implementations
    ├── evaluator.py             # Benchmark evaluation
    ├── ground_truth.py          # Test queries with expected results
    ├── collection_manager.py    # Weaviate collection setup
    ├── embedder.py              # Ollama/Jina/HuggingFace embeddings
    ├── chunker.py               # Document chunking
    ├── document_fetcher.py      # Wikipedia fetcher
    └── text_extractor.py        # Text cleaning
```
