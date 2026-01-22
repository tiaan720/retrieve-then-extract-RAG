from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Optional, Type, Generator
import time

from weaviate.classes.query import Rerank

from src.config import Config
from src.utils.logger import logger


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
    """
    Abstract base class for all retrieval strategies.
    
    Provides shared functionality for search operations including timing helpers
    and metrics creation. Subclasses must implement the _execute_search() method.
    
    Attributes:
        name: Strategy identifier (class-level, override in subclasses)
        collection: Weaviate collection to query
        embedder_type: Type of embedder needed ('single' or 'multi')
    """
    
    # Override in subclasses
    name: str = "BaseStrategy"
    embedder_type: str = "single"
    
    def __init__(self, collection, config: Optional[Config] = None):
        """
        Initialize retrieval strategy.
        
        Args:
            collection: Weaviate collection to query
            config: Configuration object (optional, creates new if not provided)
        """
        self._config = config or Config()
        self.collection = collection
        
        self._validate_configuration()
        self._initialize_backend()
    
    def _validate_configuration(self) -> None:
        """Validate strategy configuration. Override in subclasses if needed."""
        pass
    
    def _initialize_backend(self) -> None:
        """Initialize strategy-specific resources. Override in subclasses if needed."""
        pass
    
    @contextmanager
    def _timed_operation(self) -> Generator[Dict[str, float], None, None]:
        """
        Context manager for timing search operations.
        
        Yields a dict that will be populated with elapsed_ms after the block.
        
        Usage:
            with self._timed_operation() as timing:
                # do work
            elapsed = timing['elapsed_ms']
        """
        timing = {'elapsed_ms': 0.0}
        start = time.time()
        try:
            yield timing
        finally:
            timing['elapsed_ms'] = (time.time() - start) * 1000
    
    def _create_metrics(
        self,
        search_time_ms: float,
        postprocess_time_ms: float = 0.0,
        results_count: int = 0
    ) -> RetrievalMetrics:
        """
        Create standardized metrics object.
        
        Args:
            search_time_ms: Time spent in search operation
            postprocess_time_ms: Time spent in post-processing (optional)
            results_count: Number of results returned
            
        Returns:
            RetrievalMetrics with query/embedding times set to 0 (filled by caller)
        """
        return RetrievalMetrics(
            query_time_ms=0,  # Set by caller (evaluator)
            embedding_time_ms=0,  # Set by caller (evaluator)
            search_time_ms=search_time_ms,
            postprocess_time_ms=postprocess_time_ms,
            total_time_ms=search_time_ms + postprocess_time_ms,
            results_count=results_count,
            strategy_name=self.name
        )
    
    def _extract_results(self, response) -> List[Dict]:
        """Extract results from Weaviate response into standardized format."""
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
    
    @abstractmethod
    def search(
        self,
        query_vector,
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """
        Execute search with this strategy.
        
        Args:
            query_vector: Query embedding (List[float] for single, List[List[float]] for multi)
            query_text: Query text (for hybrid/reranking)
            limit: Number of results to return
            
        Returns:
            Tuple of (results, metrics)
        """
        pass


class SingleVectorStrategy(RetrievalStrategy):
    """
    Base class for strategies using single-vector embeddings.
    
    Single-vector strategies expect query_vector as List[float] - one dense
    vector representing the entire query.
    """
    
    embedder_type: str = "single"
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Search with single-vector embedding."""
        pass


class MultiVectorStrategy(RetrievalStrategy):
    """
    Base class for strategies using multi-vector (ColBERT) embeddings.
    
    Multi-vector strategies expect query_vector as List[List[float]] - one
    vector per token for late interaction scoring.
    """
    
    embedder_type: str = "multi"
    
    @abstractmethod
    def search(
        self,
        query_vector: List[List[float]],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Search with multi-vector (ColBERT) embedding."""
        pass


class StandardHNSW(SingleVectorStrategy):
    """Standard fp32 HNSW search (baseline)."""
    
    name: str = "StandardHNSW"
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Standard vector similarity search."""
        with self._timed_operation() as timing:
            response = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=limit
            )
        
        results = self._extract_results(response)
        metrics = self._create_metrics(
            search_time_ms=timing['elapsed_ms'],
            results_count=len(results)
        )
        
        return results, metrics


class BinaryQuantized(SingleVectorStrategy):
    """Binary quantization with HNSW (32x compression)."""
    
    name: str = "BinaryQuantized"
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Binary quantized search (Weaviate handles quantization internally)."""
        with self._timed_operation() as timing:
            response = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=limit
            )
        
        results = self._extract_results(response)
        metrics = self._create_metrics(
            search_time_ms=timing['elapsed_ms'],
            results_count=len(results)
        )
        
        return results, metrics


class HybridSearch(SingleVectorStrategy):
    """Hybrid search combining BM25 and vector similarity."""
    
    name: str = "HybridSearch"
    
    def __init__(
        self,
        collection,
        alpha: Optional[float] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize hybrid search.
        
        Args:
            collection: Weaviate collection
            alpha: Balance between vector (1.0) and keyword (0.0) search.
                   If None, uses config.HYBRID_ALPHA (default: 0.7)
            config: Configuration object
        """
        super().__init__(collection, config)
        self.alpha = alpha if alpha is not None else self._config.HYBRID_ALPHA
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Hybrid search with BM25 + vector."""
        with self._timed_operation() as timing:
            response = self.collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=limit,
                alpha=self.alpha
            )
        
        results = self._extract_results(response)
        metrics = self._create_metrics(
            search_time_ms=timing['elapsed_ms'],
            results_count=len(results)
        )
        
        return results, metrics


class CrossEncoderRerank(SingleVectorStrategy):
    """Two-stage retrieval with cross-encoder reranking."""
    
    name: str = "CrossEncoderRerank"
    
    def __init__(
        self,
        collection,
        rerank_multiplier: Optional[int] = None,
        alpha: Optional[float] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize reranking strategy.
        
        Args:
            collection: Weaviate collection
            rerank_multiplier: Retrieve N×limit candidates for reranking.
                               If None, uses config.RERANK_MULTIPLIER (default: 4)
            alpha: Balance between vector (1.0) and keyword (0.0) search.
                   If None, uses config.HYBRID_ALPHA (default: 0.7)
            config: Configuration object
        """
        super().__init__(collection, config)
        self.rerank_multiplier = rerank_multiplier if rerank_multiplier is not None else self._config.RERANK_MULTIPLIER
        self.alpha = alpha if alpha is not None else self._config.HYBRID_ALPHA
    
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """Two-stage retrieval with cross-encoder reranking."""
        with self._timed_operation() as timing:
            response = self.collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=limit,
                alpha=self.alpha,
                rerank=Rerank(
                    prop="content",
                    query=query_text
                )
            )
        
        results = self._extract_results(response)
        metrics = self._create_metrics(
            search_time_ms=timing['elapsed_ms'],
            results_count=len(results)
        )
        
        return results, metrics


class BinaryInt8Staged(SingleVectorStrategy):
    """
    Binary filter → Int8 rescore staged retrieval.
    
    Mimics the article's staged retrieval strategy:
    1. Fast binary search for broad recall
    2. Retrieve more candidates (rescore_multiplier × limit)
    3. Weaviate internally rescores with higher precision
    4. Return top-k results
    """
    
    name: str = "BinaryInt8Staged"
    
    def __init__(
        self,
        collection,
        rescore_multiplier: Optional[int] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize staged retrieval.
        
        Args:
            collection: Weaviate collection (must have binary quantization)
            rescore_multiplier: How many candidates to retrieve for rescoring.
                               If None, uses config.RESCORE_MULTIPLIER (default: 4)
            config: Configuration object
        """
        super().__init__(collection, config)
        self.rescore_multiplier = rescore_multiplier if rescore_multiplier is not None else self._config.RESCORE_MULTIPLIER
    
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
        candidate_limit = limit * self.rescore_multiplier
        
        # Stage 1: Binary search for candidates
        with self._timed_operation() as search_timing:
            response = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=candidate_limit,
                return_metadata=["distance"]
            )
        
        # Stage 2: Extract and take top-k (already sorted by Weaviate)
        with self._timed_operation() as postprocess_timing:
            results = self._extract_results(response)
            results = results[:limit]
        
        metrics = self._create_metrics(
            search_time_ms=search_timing['elapsed_ms'],
            postprocess_time_ms=postprocess_timing['elapsed_ms'],
            results_count=len(results)
        )
        
        logger.debug(
            f"{self.name}: Retrieved {candidate_limit} candidates, "
            f"returned top {limit} (search: {search_timing['elapsed_ms']:.2f}ms, "
            f"postprocess: {postprocess_timing['elapsed_ms']:.2f}ms)"
        )
        
        return results, metrics


class ColBERTMultiVector(MultiVectorStrategy):
    """
    ColBERT multi-vector embedding search with late interaction.
    
    Uses ColBERT embeddings where each text is represented by multiple vectors
    (one per token). This enables more nuanced similarity matching through
    late interaction - comparing individual parts of texts rather than whole
    document representations.
    """
    
    name: str = "ColBERTMultiVector"
    
    def search(
        self,
        query_vector: List[List[float]],
        query_text: str,
        limit: int = 5
    ) -> tuple[List[Dict], RetrievalMetrics]:
        """
        Search using ColBERT multi-vector embeddings.
        
        Args:
            query_vector: Multi-vector query embedding (list of vectors)
            query_text: Query text (for logging)
            limit: Number of results to return
            
        Returns:
            Tuple of (results, metrics)
        """
        with self._timed_operation() as timing:
            # Weaviate handles multi-vector comparison internally
            # using MaxSim (maximum similarity) late interaction
            response = self.collection.query.near_vector(
                target_vector="colbert",
                near_vector=query_vector,
                limit=limit
            )
        
        results = self._extract_results(response)
        metrics = self._create_metrics(
            search_time_ms=timing['elapsed_ms'],
            results_count=len(results)
        )
        
        return results, metrics



# Registry mapping strategy types to their classes
STRATEGY_REGISTRY: Dict[str, Type[RetrievalStrategy]] = {
    "standard": StandardHNSW,
    "hnsw": StandardHNSW,
    "binary": BinaryQuantized,
    "bq": BinaryQuantized,
    "hybrid": HybridSearch,
    "rerank": CrossEncoderRerank,
    "cross-encoder": CrossEncoderRerank,
    "staged": BinaryInt8Staged,
    "binary-int8": BinaryInt8Staged,
    "colbert": ColBERTMultiVector,
    "multi-vector": ColBERTMultiVector,
}


def create_retrieval_strategy(
    strategy_type: str,
    collection,
    config: Optional[Config] = None,
    **kwargs
) -> RetrievalStrategy:
    """
    Factory function to create a retrieval strategy instance.
    
    Uses a registry pattern for easy extensibility. All strategies accept
    an optional `config` parameter for centralized configuration.
    
    Args:
        strategy_type: Type of strategy (see STRATEGY_REGISTRY keys)
        collection: Weaviate collection to query
        config: Configuration object (optional)
        **kwargs: Additional arguments passed to the strategy constructor
            - For hybrid: alpha
            - For rerank: rerank_multiplier, alpha
            - For staged: rescore_multiplier
    
    Returns:
        Configured strategy instance
    
    Raises:
        ValueError: If strategy_type is not recognized
    
    Examples:
        >>> strategy = create_retrieval_strategy("hybrid", collection, alpha=0.8)
        >>> strategy = create_retrieval_strategy("rerank", collection, rerank_multiplier=5)
        >>> strategy = create_retrieval_strategy("colbert", collection)
    """
    strategy_type = strategy_type.lower()
    
    if strategy_type not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy_type: '{strategy_type}'. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_type]
    
    if config is not None:
        kwargs['config'] = config
    
    return strategy_class(collection, **kwargs)


def register_strategy(name: str, strategy_class: Type[RetrievalStrategy]) -> None:
    """
    Register a custom strategy class.
    
    Args:
        name: Name to register the strategy under
        strategy_class: Strategy class (must inherit from RetrievalStrategy)
    """
    if not issubclass(strategy_class, RetrievalStrategy):
        raise TypeError(f"{strategy_class} must inherit from RetrievalStrategy")
    STRATEGY_REGISTRY[name.lower()] = strategy_class
    logger.info(f"Registered custom strategy: {name}")
