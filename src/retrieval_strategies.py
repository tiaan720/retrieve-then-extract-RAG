"""
Retrieval strategies for comparing different search approaches.

Implements multiple retrieval strategies to compare performance and accuracy:
1. StandardHNSW - Default fp32 HNSW index
2. BinaryQuantized - Binary quantization (32x compression)
3. HybridSearch - BM25 + Vector fusion
4. CrossEncoderRerank - Two-stage retrieval with reranking
5. BinaryInt8Staged - Article's approach (binary filter → int8 rescore)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
from weaviate.classes.query import Rerank
from src.logger import logger


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""
    query_time_ms: float
    embedding_time_ms: float
    search_time_ms: float
    postprocess_time_ms: float
    total_time_ms: float
    results_count: int
    strategy_name: str


class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""
    
    def __init__(self, name: str, collection):
        """
        Initialize retrieval strategy.
        
        Args:
            name: Strategy name for identification
            collection: Weaviate collection to query
        """
        self.name = name
        self.collection = collection
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """
        Execute search with this strategy.
        
        Args:
            query_vector: Query embedding vector
            query_text: Query text (for hybrid/reranking)
            limit: Number of results to return
            
        Returns:
            Tuple of (results, metrics)
        """
        pass
    
    def _extract_results(self, response) -> List[Dict]:
        """Helper to extract results from Weaviate response."""
        results = []
        for obj in response.objects:
            results.append({
                "content": obj.properties.get("content", ""),
                "title": obj.properties.get("title", ""),
                "url": obj.properties.get("url", ""),
                "chunk_index": obj.properties.get("chunk_index", 0),
                "total_chunks": obj.properties.get("total_chunks", 0),
                "source": obj.properties.get("source", ""),
                "language": obj.properties.get("language", ""),
            })
        return results


class StandardHNSW(RetrievalStrategy):
    """Standard fp32 HNSW search (baseline)."""
    
    def __init__(self, collection):
        super().__init__("StandardHNSW", collection)
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Standard vector similarity search."""
        search_start = time.time()
        
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=limit
        )
        
        search_time = (time.time() - search_start) * 1000
        
        results = self._extract_results(response)
        
        metrics = RetrievalMetrics(
            query_time_ms=0,  # Set by caller
            embedding_time_ms=0,  # Set by caller
            search_time_ms=search_time,
            postprocess_time_ms=0,
            total_time_ms=search_time,
            results_count=len(results),
            strategy_name=self.name
        )
        
        return results, metrics


class BinaryQuantized(RetrievalStrategy):
    """Binary quantization with HNSW (32x compression)."""
    
    def __init__(self, collection):
        super().__init__("BinaryQuantized", collection)
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Binary quantized search (Weaviate handles quantization internally)."""
        search_start = time.time()
        
        # Weaviate automatically uses binary quantization if configured
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=limit
        )
        
        search_time = (time.time() - search_start) * 1000
        
        results = self._extract_results(response)
        
        metrics = RetrievalMetrics(
            query_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=search_time,
            postprocess_time_ms=0,
            total_time_ms=search_time,
            results_count=len(results),
            strategy_name=self.name
        )
        
        return results, metrics


class HybridSearch(RetrievalStrategy):
    """Hybrid search combining BM25 and vector similarity."""
    
    def __init__(self, collection, alpha: float = 0.7):
        """
        Initialize hybrid search.
        
        Args:
            collection: Weaviate collection
            alpha: Balance between vector (1.0) and keyword (0.0) search
        """
        super().__init__("HybridSearch", collection)
        self.alpha = alpha
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Hybrid search with BM25 + vector."""
        search_start = time.time()
        
        response = self.collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            limit=limit,
            alpha=self.alpha
        )
        
        search_time = (time.time() - search_start) * 1000
        
        results = self._extract_results(response)
        
        metrics = RetrievalMetrics(
            query_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=search_time,
            postprocess_time_ms=0,
            total_time_ms=search_time,
            results_count=len(results),
            strategy_name=self.name
        )
        
        return results, metrics


class CrossEncoderRerank(RetrievalStrategy):
    """Two-stage retrieval with cross-encoder reranking."""
    
    def __init__(self, collection, rerank_multiplier: int = 4):
        """
        Initialize reranking strategy.
        
        Args:
            collection: Weaviate collection
            rerank_multiplier: Retrieve N×limit candidates for reranking
        """
        super().__init__("CrossEncoderRerank", collection)
        self.rerank_multiplier = rerank_multiplier
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Two-stage retrieval with cross-encoder reranking."""
        search_start = time.time()
        
        # Stage 1: Retrieve candidates
        candidate_limit = limit * self.rerank_multiplier
        
        # Use hybrid search with reranking
        response = self.collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            limit=limit,
            alpha=0.7,
            # Rerank using cross-encoder
            rerank=Rerank(
                prop="content",
                query=query_text
            )
        )
        
        search_time = (time.time() - search_start) * 1000
        
        results = self._extract_results(response)
        
        metrics = RetrievalMetrics(
            query_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=search_time,
            postprocess_time_ms=0,
            total_time_ms=search_time,
            results_count=len(results),
            strategy_name=self.name
        )
        
        return results, metrics


class BinaryInt8Staged(RetrievalStrategy):
    """
    Article's approach: Binary filter → Int8 rescore.
    
    Mimics the article's staged retrieval strategy:
    1. Fast binary search for broad recall
    2. Retrieve more candidates (rescore_multiplier × limit)
    3. Weaviate internally rescores with higher precision
    4. Return top-k results
    
    This approximates the article's binary→int8→fp32 cascade
    using Weaviate's internal quantization and rescoring.
    """
    
    def __init__(self, collection, rescore_multiplier: int = 4):
        """
        Initialize staged retrieval.
        
        Args:
            collection: Weaviate collection (must have binary quantization)
            rescore_multiplier: How many candidates to retrieve for rescoring
        """
        super().__init__("BinaryInt8Staged", collection)
        self.rescore_multiplier = rescore_multiplier
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """
        Staged retrieval mimicking article's approach.
        
        Stage 1: Binary search (fast, broad recall)
        Stage 2: Weaviate internal rescoring (precision refinement)
        Stage 3: Return top-k
        """
        # Stage 1: Binary search for candidates
        search_start = time.time()
        candidate_limit = limit * self.rescore_multiplier
        
        # Weaviate uses binary quantization for initial search
        # then automatically rescores top candidates with fp32
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=candidate_limit,  # Get more candidates
            return_metadata=["distance"]
        )
        
        stage1_time = (time.time() - search_start) * 1000
        
        # Stage 2: Extract and sort by distance (already done by Weaviate)
        postprocess_start = time.time()
        results = self._extract_results(response)
        
        # Take top-k after Weaviate's internal rescoring
        results = results[:limit]
        
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        total_time = stage1_time + postprocess_time
        
        metrics = RetrievalMetrics(
            query_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=stage1_time,
            postprocess_time_ms=postprocess_time,
            total_time_ms=total_time,
            results_count=len(results),
            strategy_name=self.name
        )
        
        logger.debug(
            f"{self.name}: Retrieved {candidate_limit} candidates, "
            f"returned top {limit} (search: {stage1_time:.2f}ms, "
            f"postprocess: {postprocess_time:.2f}ms)"
        )
        
        return results, metrics
