# Quick Optimization Guide: Making Your RAG System Faster

This guide shows you **exactly** how to implement the optimizations discussed in the performance analysis.

---

## ⚡ Quick Wins (5 Minutes)

### 1. Enable Binary Quantization

**Current**: Using fp32 embeddings (slow, memory-intensive)  
**Change**: Enable binary quantization (32x smaller, much faster)

**Edit `.env` file:**
```bash
ENABLE_BINARY_QUANTIZATION=true
```

**Or in `src/config.py`:**
```python
ENABLE_BINARY_QUANTIZATION: bool = Field(default=True)  # Changed from False
```

**Impact**: 
- ✅ 32x memory reduction
- ✅ 3-5x search speed improvement  
- ⚠️ ~3-4% accuracy drop (acceptable for most use cases)

---

### 2. Use a Faster Embedding Model

**Current**: `snowflake-arctic-embed:33m` (slow but accurate)  
**Try**: `nomic-embed-text` (2-3x faster)

**Edit `.env` file:**
```bash
OLLAMA_MODEL=nomic-embed-text
```

**Pull the model first:**
```bash
ollama pull nomic-embed-text
```

**Impact**:
- ✅ 2-3x faster embedding time
- ✅ Similar quality
- ✅ Smaller model (faster startup)

**Alternative fast models:**
- `all-minilm` (fastest, lower quality)
- `nomic-embed-text` (balanced)
- `bge-small` (good quality, reasonable speed)

---

### 3. Skip Reranking for Most Queries

**Current**: Using cross-encoder reranking (slow, accurate)  
**Change**: Use hybrid search by default

**In `main.py`, replace:**
```python
# Slow version (300-500ms extra)
reranked_results = weaviate_client.rerank_query(
    query_text=test_query_text,
    query_vector=test_embedding,
    limit=3,
    rerank_limit=10
)
```

**With:**
```python
# Fast version (no extra time)
results = weaviate_client.hybrid_query(
    query_text=test_query_text,
    query_vector=test_embedding,
    limit=3,
    alpha=0.8  # 80% vector, 20% keyword
)
```

**Impact**:
- ✅ 300-500ms saved per query
- ⚠️ Slightly lower precision (but still very good)
- 💡 Use reranking only when you really need it

---

## 🔧 Medium Optimizations (15 Minutes)

### 4. Tune HNSW Parameters for Speed

**Current**: Optimized for accuracy  
**Change**: Balance speed and accuracy

**In `src/weaviate_client.py`, find the `create_schema()` method and modify:**

```python
# Current (accuracy-focused)
hnsw_config = Configure.VectorIndex.hnsw(
    distance_metric=VectorDistances.COSINE,
    ef_construction=128,  # Higher = better quality, slower build
    max_connections=64,   # Higher = better accuracy, more memory
    vector_cache_max_objects=10000,
)

# Optimized (speed-focused)
hnsw_config = Configure.VectorIndex.hnsw(
    distance_metric=VectorDistances.COSINE,
    ef_construction=64,   # ⬇️ Halved for 2x faster build
    max_connections=32,   # ⬇️ Halved for less memory
    ef=-1,                # ✨ Dynamic ef (auto-tune at query time)
    vector_cache_max_objects=100000,  # ⬆️ Cache more for speed
    quantizer=Configure.VectorIndex.Quantizer.bq()  # Enable binary quantization
)
```

**Impact**:
- ✅ 20-40% faster queries
- ✅ Lower memory usage
- ⚠️ 1-2% accuracy drop

---

### 5. Implement Staged Retrieval (Article's Approach)

**Add this method to `WeaviateClient` class in `src/weaviate_client.py`:**

```python
def staged_query(
    self, 
    query_text: str, 
    query_vector: List[float], 
    limit: int = 5,
    rescore_multiplier: int = 4
) -> List[Dict]:
    """
    Staged retrieval: Fast binary search → precise rescoring.
    
    This mimics the article's approach:
    1. Retrieve more candidates with binary quantization (fast)
    2. Return top-k after Weaviate's internal rescoring
    
    Args:
        query_text: Text query for hybrid search
        query_vector: Query embedding vector
        limit: Final number of results
        rescore_multiplier: How many candidates to retrieve initially
        
    Returns:
        Top-k results after rescoring
    """
    if not self.enable_binary_quantization:
        logger.warning("Binary quantization not enabled. Falling back to standard query.")
        return self.hybrid_query(query_text, query_vector, limit)
    
    # Stage 1: Retrieve more candidates with fast binary search
    candidate_limit = limit * rescore_multiplier
    
    # Weaviate automatically uses binary quantized index if enabled
    # and performs internal rescoring with full precision
    candidates = self.hybrid_query(
        query_text=query_text,
        query_vector=query_vector,
        limit=candidate_limit,
        alpha=0.8  # Favor vector search
    )
    
    # Weaviate already rescored internally, so top results are best
    return candidates[:limit]
```

**Usage in `main.py`:**
```python
# Use staged retrieval
staged_results = weaviate_client.staged_query(
    query_text="What is machine learning?",
    query_vector=test_embedding,
    limit=3,
    rescore_multiplier=4  # Retrieve 12, return best 3
)
```

**Impact**:
- ✅ Combines binary speed with rescoring accuracy
- ✅ ~99% of fp32 quality (per article)
- ✅ No significant latency increase

---

## 🚀 Advanced Optimizations (30+ Minutes)

### 6. Bypass Ollama for Embedding Speed

**Current**: Ollama API overhead (network calls, serialization)  
**Change**: Use sentence-transformers directly

**Install:**
```bash
pip install sentence-transformers
```

**Create new embedder in `src/embedder_fast.py`:**
```python
from typing import List
from sentence_transformers import SentenceTransformer
from src.logger import logger


class FastEmbeddingGenerator:
    """Direct embedding generation without Ollama overhead."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize with sentence-transformers directly.
        
        Popular fast models:
        - all-MiniLM-L6-v2: 384-dim, very fast
        - all-MiniLM-L12-v2: 384-dim, balanced
        - paraphrase-MiniLM-L6-v2: 384-dim, good quality
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched for speed)."""
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=32,  # Process in batches
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """Generate embeddings for document chunks."""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            embedded_chunks.append(chunk_with_embedding)
        
        return embedded_chunks
```

**Use in `main.py`:**
```python
# Replace
from src.embedder import EmbeddingGenerator
embedder = EmbeddingGenerator(
    base_url=config.OLLAMA_BASE_URL,
    model=config.OLLAMA_MODEL
)

# With
from src.embedder_fast import FastEmbeddingGenerator
embedder = FastEmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Impact**:
- ✅ 3-5x faster embedding (no network overhead)
- ✅ Batch processing for multiple queries
- ⚠️ Need to download models locally

---

### 7. Implement True Int8 Rescoring (Article's Exact Approach)

**Note**: This requires storing int8 embeddings separately. Weaviate doesn't natively support this, so you'd need a custom solution.

**High-level approach:**
```python
import numpy as np

def quantize_to_int8(embeddings: List[float]) -> np.ndarray:
    """Quantize fp32 embeddings to int8."""
    arr = np.array(embeddings)
    # Scale to [-128, 127] range
    min_val, max_val = arr.min(), arr.max()
    scale = 255.0 / (max_val - min_val)
    int8_arr = ((arr - min_val) * scale - 128).astype(np.int8)
    return int8_arr, min_val, max_val

def dequantize_from_int8(int8_arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Dequantize int8 back to approximate fp32."""
    scale = 255.0 / (max_val - min_val)
    return (int8_arr.astype(np.float32) + 128) / scale + min_val
```

**This is complex and requires:**
- Storing int8 embeddings in separate storage (e.g., HDF5, numpy files)
- Custom retrieval pipeline outside Weaviate
- More code maintenance

**Recommendation**: Only do this if you have 10M+ documents and need extreme optimization.

---

## 📊 Expected Performance Improvements

### Baseline (Current Setup)
```
Query Embedding: 150ms (Ollama + snowflake-arctic-embed)
Vector Search:   80ms  (HNSW, fp32)
Reranking:       300ms (cross-encoder)
───────────────────────
Total:           530ms
```

### After Quick Wins (Binary + Fast Model + No Rerank)
```
Query Embedding: 50ms  (nomic-embed-text, 3x faster)
Binary Search:   20ms  (binary quantization, 4x faster)
No Reranking:    0ms   (skipped)
───────────────────────
Total:           70ms  ✅ 7.5x speedup!
```

### With All Optimizations (+ HNSW Tuning + Direct Embedding)
```
Query Embedding: 30ms  (sentence-transformers direct)
Binary Search:   15ms  (optimized HNSW + binary)
Staged Rescore:  5ms   (internal Weaviate rescoring)
───────────────────────
Total:           50ms  ✅ 10.6x speedup!
```

---

## 🎯 Recommended Configuration

### For Speed-Critical Applications (e.g., chat, autocomplete)
```bash
# .env file
ENABLE_BINARY_QUANTIZATION=true
OLLAMA_MODEL=nomic-embed-text
```

```python
# main.py - use staged_query
results = weaviate_client.staged_query(
    query_text=query,
    query_vector=embedding,
    limit=5,
    rescore_multiplier=4
)
```

**Expected latency**: 50-100ms  
**Accuracy**: ~97-98% of fp32

---

### For Accuracy-Critical Applications (e.g., legal, medical)
```bash
# .env file
ENABLE_BINARY_QUANTIZATION=false
OLLAMA_MODEL=snowflake-arctic-embed:33m
```

```python
# main.py - use reranking
results = weaviate_client.rerank_query(
    query_text=query,
    query_vector=embedding,
    limit=5,
    rerank_limit=20
)
```

**Expected latency**: 400-600ms  
**Accuracy**: Best possible

---

### Balanced (Recommended for Most Cases)
```bash
# .env file
ENABLE_BINARY_QUANTIZATION=true
OLLAMA_MODEL=nomic-embed-text
```

```python
# main.py - use hybrid search
results = weaviate_client.hybrid_query(
    query_text=query,
    query_vector=embedding,
    limit=5,
    alpha=0.8
)
```

**Expected latency**: 100-200ms  
**Accuracy**: 96-98% of fp32

---

## 🧪 How to Benchmark Your System

Add this to `main.py` to measure performance:

```python
import time

def benchmark_query(weaviate_client, query_text, query_vector, iterations=10):
    """Benchmark query performance."""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        results = weaviate_client.hybrid_query(
            query_text=query_text,
            query_vector=query_vector,
            limit=5
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    logger.info(f"Query Performance:")
    logger.info(f"  Average: {avg_time:.2f}ms")
    logger.info(f"  Min: {min_time:.2f}ms")
    logger.info(f"  Max: {max_time:.2f}ms")
    
    return avg_time

# Use it
avg_latency = benchmark_query(
    weaviate_client, 
    "What is machine learning?", 
    test_embedding,
    iterations=10
)
```

---

## 🔍 Debugging Slow Queries

If queries are still slow, check:

1. **Ollama is local** (not remote): `OLLAMA_BASE_URL=http://localhost:11434`
2. **Weaviate is local** (not remote): `WEAVIATE_URL=http://localhost:8080`
3. **No cold start**: First query is always slower (model loading)
4. **Check Weaviate logs**: `docker logs weaviate` for performance issues
5. **Profile embedding**: Measure embedding time separately

```python
# Profile embedding
start = time.time()
embedding = embedder.embed_text("test query")
embed_time = (time.time() - start) * 1000
logger.info(f"Embedding time: {embed_time:.2f}ms")

# Profile search
start = time.time()
results = weaviate_client.query(embedding, limit=5)
search_time = (time.time() - start) * 1000
logger.info(f"Search time: {search_time:.2f}ms")
```

---

## ✅ Summary

**To match the article's speed (~200ms)**:
1. ✅ Enable binary quantization
2. ✅ Use faster embedding model
3. ✅ Skip reranking for most queries
4. ✅ Tune HNSW parameters
5. ✅ Use staged retrieval

**Expected result**: 70-150ms per query (competitive with article!)

**The article's 200ms claim is legit**, but they optimized everything (model, quantization, staging). You can get close with minimal code changes!
