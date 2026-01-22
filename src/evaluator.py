import time
import statistics
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import json

from src.logger import logger
from src.retrieval_strategies import RetrievalStrategy, RetrievalMetrics
from src.embedder import BaseEmbedder, SingleVectorEmbedder, MultiVectorEmbedder


@dataclass
class StrategyBenchmark:
    """Benchmark results for a single strategy."""
    strategy_name: str
    
    # Speed
    avg_latency_ms: float
    
    # Accuracy metrics
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    
    # Count
    num_queries: int = 0


class RetrievalEvaluator:
    """
    Evaluates and compares retrieval strategies.
    
    Automatically selects the appropriate embedder based on each strategy's
    requirements (single-vector vs multi-vector).
    
    Example usage:
        # Single-vector strategies only
        evaluator = RetrievalEvaluator(embedder, ground_truth=ground_truth)
        
        # Both single and multi-vector strategies
        evaluator = RetrievalEvaluator(
            embedder,
            ground_truth=ground_truth,
            multi_vector_embedder=colbert_embedder
        )
    """
    
    def __init__(
        self, 
        embedder: BaseEmbedder,
        ground_truth: Optional[Dict[str, List[str]]] = None,
        multi_vector_embedder: Optional[BaseEmbedder] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            embedder: Primary embedder for single-vector strategies
            ground_truth: Dict mapping query -> list of expected document titles
            multi_vector_embedder: Optional embedder for multi-vector strategies
                                   (only needed if benchmarking ColBERT strategies)
        """
        self._embedders: Dict[str, BaseEmbedder] = {
            "single": embedder
        }
        
        if multi_vector_embedder is not None:
            self._embedders["multi"] = multi_vector_embedder
        
        self.results: Dict[str, StrategyBenchmark] = {}
        self.ground_truth = ground_truth or {}
    
    def _get_embedder_for_strategy(self, strategy: RetrievalStrategy) -> BaseEmbedder:
        """
        Get the appropriate embedder for a strategy.
        
        Args:
            strategy: The retrieval strategy
            
        Returns:
            Appropriate embedder based on strategy.embedder_type
            
        Raises:
            ValueError: If required embedder type is not available
        """
        embedder_type = getattr(strategy, 'embedder_type', 'single')
        
        if embedder_type not in self._embedders:
            available = list(self._embedders.keys())
            raise ValueError(
                f"Strategy '{strategy.name}' requires '{embedder_type}' embedder, "
                f"but only {available} embedder(s) provided. "
                f"Pass multi_vector_embedder to RetrievalEvaluator to use this strategy."
            )
        
        return self._embedders[embedder_type]
    
    def benchmark_strategy(
        self,
        strategy: RetrievalStrategy,
        queries: List[str],
        limit: int = 5,
        warmup_queries: int = 2
    ) -> StrategyBenchmark:
        """
        Benchmark a single retrieval strategy.
        
        Args:
            strategy: Strategy to benchmark
            queries: List of query strings
            limit: Number of results per query
            warmup_queries: Number of warmup queries (excluded from metrics)
            
        Returns:
            StrategyBenchmark with performance and accuracy metrics
        """
        logger.info(f"Benchmarking: {strategy.name}")
        
        # Auto-select embedder based on strategy requirements
        embedder = self._get_embedder_for_strategy(strategy)
        embedder_type = getattr(strategy, 'embedder_type', 'single')
        logger.debug(f"  Using {embedder.__class__.__name__} ({embedder_type})")
        
        # Warmup
        if warmup_queries > 0:
            for i in range(min(warmup_queries, len(queries))):
                query_vec = embedder.embed_text(queries[i])
                strategy.search(query_vec, queries[i], limit=limit)
        
        # Collect metrics
        latencies = []
        all_results = {}
        
        for query in queries:
            start_time = time.time()
            query_vec = embedder.embed_text(query)
            results, _ = strategy.search(query_vec, query, limit=10)
            total_time = (time.time() - start_time) * 1000
            
            latencies.append(total_time)
            all_results[query] = [r["title"] for r in results]
        
        # Calculate accuracy using ground truth
        recall_at_5, recall_at_10 = self._calculate_recall(queries, all_results)
        
        benchmark = StrategyBenchmark(
            strategy_name=strategy.name,
            avg_latency_ms=statistics.mean(latencies),
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            num_queries=len(queries)
        )
        
        self.results[strategy.name] = benchmark
        
        recall_str = f", recall@5={recall_at_5:.1%}" if recall_at_5 is not None else ""
        logger.info(f"  {strategy.name}: {benchmark.avg_latency_ms:.1f}ms{recall_str}")
        
        return benchmark
    
    def _calculate_recall(
        self, 
        queries: List[str], 
        all_results: Dict[str, List[str]]
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate recall@5 and recall@10 against ground truth."""
        if not self.ground_truth:
            return None, None
        
        recall_5_scores = []
        recall_10_scores = []
        
        for query in queries:
            if query in self.ground_truth and query in all_results:
                expected_titles = set(self.ground_truth[query])
                retrieved_5 = set(all_results[query][:5])
                retrieved_10 = set(all_results[query][:10])
                
                # Binary recall: did we find ANY expected document?
                found_in_5 = 1.0 if expected_titles & retrieved_5 else 0.0
                found_in_10 = 1.0 if expected_titles & retrieved_10 else 0.0
                
                recall_5_scores.append(found_in_5)
                recall_10_scores.append(found_in_10)
        
        if not recall_5_scores:
            return None, None
        
        return statistics.mean(recall_5_scores), statistics.mean(recall_10_scores)
    
    def benchmark_all_strategies(
        self,
        strategies: List[RetrievalStrategy],
        queries: List[str],
        limit: int = 5,
        warmup_queries: int = 2
    ) -> Dict[str, StrategyBenchmark]:
        """
        Benchmark multiple strategies.
        
        Automatically skips strategies that require embedders not provided.
        
        Args:
            strategies: List of strategies to benchmark
            queries: Test queries
            limit: Results per query
            warmup_queries: Warmup queries per strategy
            
        Returns:
            Dictionary mapping strategy name to benchmark results
        """
        logger.info(f"Benchmarking {len(strategies)} strategies on {len(queries)} queries")
        
        if self.ground_truth:
            logger.info(f"Using ground truth with {len(self.ground_truth)} annotated queries")
        else:
            logger.warning("No ground truth provided - accuracy metrics will be N/A")
        
        # Log available embedders
        logger.info(f"Available embedders: {list(self._embedders.keys())}")
        
        for strategy in strategies:
            try:
                self.benchmark_strategy(strategy, queries, limit, warmup_queries)
            except ValueError as e:
                logger.warning(f"Skipping {strategy.name}: {e}")
        
        return self.results
    
    def print_comparison_table(self) -> None:
        """Log comparison table of all benchmarked strategies."""
        if not self.results:
            logger.warning("No results to display")
            return
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("RETRIEVAL STRATEGY COMPARISON")
        logger.info("=" * 70)
        logger.info(f"{'Strategy':<25} {'Latency (ms)':<15} {'Recall@5':<12} {'Recall@10':<12}")
        logger.info("-" * 70)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].avg_latency_ms)
        
        for name, benchmark in sorted_results:
            recall5_str = f"{benchmark.recall_at_5:.1%}" if benchmark.recall_at_5 is not None else "N/A"
            recall10_str = f"{benchmark.recall_at_10:.1%}" if benchmark.recall_at_10 is not None else "N/A"
            logger.info(f"{name:<25} {benchmark.avg_latency_ms:<15.2f} {recall5_str:<12} {recall10_str:<12}")
        
        logger.info("=" * 70)
        
        # Summary
        fastest = sorted_results[0]
        logger.info(f"Fastest: {fastest[0]} ({fastest[1].avg_latency_ms:.1f}ms)")
        
        with_accuracy = [(n, b) for n, b in sorted_results if b.recall_at_5 is not None]
        if with_accuracy:
            best_accuracy = max(with_accuracy, key=lambda x: x[1].recall_at_5)
            logger.info(f"Best accuracy: {best_accuracy[0]} ({best_accuracy[1].recall_at_5:.1%} recall@5)")
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save benchmark results to JSON file."""
        output = {
            name: asdict(benchmark)
            for name, benchmark in self.results.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
