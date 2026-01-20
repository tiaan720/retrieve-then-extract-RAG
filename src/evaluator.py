import time
import statistics
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
from src.logger import logger
from src.retrieval_strategies import RetrievalStrategy, RetrievalMetrics


@dataclass
class StrategyBenchmark:
    """Benchmark results for a single strategy."""
    strategy_name: str
    
    # Speed (simplified)
    avg_latency_ms: float
    
    # Accuracy metrics
    recall_at_5: Optional[float] = None  # % of baseline results found
    recall_at_10: Optional[float] = None
    
    # Count
    num_queries: int = 0


class RetrievalEvaluator:
    """Evaluates and compares retrieval strategies."""
    
    def __init__(self, embedder):
        """
        Initialize evaluator.
        
        Args:
            embedder: Embedding generator for queries
        """
        self.embedder = embedder
        self.results = {}
        self.baseline_results = {}  # Store baseline (StandardHNSW) results for accuracy comparison
    
    def _get_baseline_results(
        self,
        baseline_strategy: RetrievalStrategy,
        queries: List[str],
        limit: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get baseline results to use as ground truth.
        
        Uses StandardHNSW (fp32) results as the "correct" answers.
        Other strategies are measured by how well they match these results.
        """
        baseline = {}
        
        for query in queries:
            query_vec = self.embedder.embed_text(query)
            results, _ = baseline_strategy.search(query_vec, query, limit=limit)
            # Use content hash as identifier (title might not be unique)
            baseline[query] = [r["content"][:100] for r in results]  # First 100 chars as ID
        
        return baseline
    
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
        
        # Warmup
        if warmup_queries > 0:
            for i in range(min(warmup_queries, len(queries))):
                query_vec = self.embedder.embed_text(queries[i])
                strategy.search(query_vec, queries[i], limit=limit)
        
        # Collect metrics
        latencies = []
        all_results = {}
        for i, query in enumerate(queries):
            # Measure total time (embedding + search)
            start_time = time.time()
            query_vec = self.embedder.embed_text(query)
            results, _ = strategy.search(query_vec, query, limit=10)  # Get 10 for recall@10
            total_time = (time.time() - start_time) * 1000
            
            latencies.append(total_time)
            all_results[query] = [r["content"][:100] for r in results]
        
        # Calculate accuracy vs baseline
        recall_at_5 = None
        recall_at_10 = None
        
        if self.baseline_results:
            recall_5_scores = []
            recall_10_scores = []
            
            for query in queries:
                if query in self.baseline_results and query in all_results:
                    baseline_set_5 = set(self.baseline_results[query][:5])
                    baseline_set_10 = set(self.baseline_results[query][:10])
                    retrieved_5 = set(all_results[query][:5])
                    retrieved_10 = set(all_results[query][:10])
                    
                    # Recall = how many baseline results did we find?
                    r5 = len(baseline_set_5 & retrieved_5) / len(baseline_set_5) if baseline_set_5 else 1.0
                    r10 = len(baseline_set_10 & retrieved_10) / len(baseline_set_10) if baseline_set_10 else 1.0
                    
                    recall_5_scores.append(r5)
                    recall_10_scores.append(r10)
            
            if recall_5_scores:
                recall_at_5 = statistics.mean(recall_5_scores)
                recall_at_10 = statistics.mean(recall_10_scores)
        
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
    
    def benchmark_all_strategies(
        self,
        strategies: List[RetrievalStrategy],
        queries: List[str],
        limit: int = 5,
        warmup_queries: int = 2
    ) -> Dict[str, StrategyBenchmark]:
        """
        Benchmark multiple strategies.
        
        Uses StandardHNSW as baseline for accuracy measurement.
        
        Args:
            strategies: List of strategies to benchmark
            queries: Test queries
            limit: Results per query
            warmup_queries: Warmup queries per strategy
            
        Returns:
            Dictionary mapping strategy name to benchmark results
        """
        logger.info(f"Benchmarking {len(strategies)} strategies on {len(queries)} queries")
        
        # Find and benchmark baseline first (StandardHNSW)
        baseline_strategy = next((s for s in strategies if s.name == "StandardHNSW"), None)
        
        if baseline_strategy:
            self.benchmark_strategy(baseline_strategy, queries, limit, warmup_queries)
            self.baseline_results = self._get_baseline_results(baseline_strategy, queries, limit=10)
        
        # Benchmark all other strategies
        for strategy in strategies:
            if strategy.name != "StandardHNSW":
                self.benchmark_strategy(strategy, queries, limit, warmup_queries)
        
        return self.results
    
    def print_comparison_table(self):
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
        
        # Sort by average latency
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].avg_latency_ms)
        
        for name, benchmark in sorted_results:
            recall5_str = f"{benchmark.recall_at_5:.1%}" if benchmark.recall_at_5 is not None else "baseline"
            recall10_str = f"{benchmark.recall_at_10:.1%}" if benchmark.recall_at_10 is not None else "baseline"
            logger.info(f"{name:<25} {benchmark.avg_latency_ms:<15.2f} {recall5_str:<12} {recall10_str:<12}")
        
        logger.info("=" * 70)
        
        # Summary
        fastest = sorted_results[0]
        logger.info(f"Fastest: {fastest[0]} ({fastest[1].avg_latency_ms:.1f}ms)")
        
        with_accuracy = [(n, b) for n, b in sorted_results if b.recall_at_5 is not None]
        if with_accuracy:
            best_accuracy = max(with_accuracy, key=lambda x: x[1].recall_at_5)
            logger.info(f"Best accuracy: {best_accuracy[0]} ({best_accuracy[1].recall_at_5:.1%} recall@5)")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output = {
            name: asdict(benchmark)
            for name, benchmark in self.results.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
