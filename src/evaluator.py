"""
Evaluation framework for comparing retrieval strategies.

Measures speed and accuracy across different retrieval approaches:
- Speed: Latency (p50, p95, p99), throughput
- Accuracy: Recall@k, precision@k (with ground truth)
- Memory: Collection size, index overhead
"""

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
    
    # Speed metrics (milliseconds)
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Breakdown
    avg_embedding_time_ms: float
    avg_search_time_ms: float
    avg_postprocess_time_ms: float
    
    # Throughput
    queries_per_second: float
    
    # Accuracy (if ground truth provided)
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    precision_at_5: Optional[float] = None
    
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
            StrategyBenchmark with performance metrics
        """
        logger.info(f"Benchmarking strategy: {strategy.name}")
        
        # Warmup
        if warmup_queries > 0:
            logger.info(f"Warming up with {warmup_queries} queries...")
            for i in range(min(warmup_queries, len(queries))):
                query_vec = self.embedder.embed_text(queries[i])
                strategy.search(query_vec, queries[i], limit=limit)
        
        # Collect metrics
        all_metrics: List[RetrievalMetrics] = []
        embedding_times = []
        
        logger.info(f"Running {len(queries)} queries...")
        for i, query in enumerate(queries):
            # Measure embedding time
            embed_start = time.time()
            query_vec = self.embedder.embed_text(query)
            embedding_time = (time.time() - embed_start) * 1000
            embedding_times.append(embedding_time)
            
            # Execute search
            _, metrics = strategy.search(query_vec, query, limit=limit)
            
            # Add embedding time to metrics
            metrics.embedding_time_ms = embedding_time
            metrics.total_time_ms += embedding_time
            
            all_metrics.append(metrics)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{len(queries)} queries")
        
        # Calculate statistics
        latencies = [m.total_time_ms for m in all_metrics]
        search_times = [m.search_time_ms for m in all_metrics]
        postprocess_times = [m.postprocess_time_ms for m in all_metrics]
        
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        # Calculate correct percentile indices
        p95_idx = min(int((n - 1) * 0.95), n - 1) if n > 0 else 0
        p99_idx = min(int((n - 1) * 0.99), n - 1) if n > 0 else 0
        
        benchmark = StrategyBenchmark(
            strategy_name=strategy.name,
            avg_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=latencies_sorted[p95_idx] if n > 0 else 0,
            p99_latency_ms=latencies_sorted[p99_idx] if n > 0 else 0,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_embedding_time_ms=statistics.mean(embedding_times),
            avg_search_time_ms=statistics.mean(search_times),
            avg_postprocess_time_ms=statistics.mean(postprocess_times),
            queries_per_second=1000 / statistics.mean(latencies) if statistics.mean(latencies) > 0 else 0,
            num_queries=len(queries)
        )
        
        self.results[strategy.name] = benchmark
        
        logger.info(f"✓ {strategy.name} benchmark complete:")
        logger.info(f"  Avg latency: {benchmark.avg_latency_ms:.2f}ms")
        logger.info(f"  Median: {benchmark.median_latency_ms:.2f}ms")
        logger.info(f"  P95: {benchmark.p95_latency_ms:.2f}ms")
        logger.info(f"  Throughput: {benchmark.queries_per_second:.2f} QPS")
        
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
        
        Args:
            strategies: List of strategies to benchmark
            queries: Test queries
            limit: Results per query
            warmup_queries: Warmup queries per strategy
            
        Returns:
            Dictionary mapping strategy name to benchmark results
        """
        logger.info(f"Benchmarking {len(strategies)} strategies on {len(queries)} queries")
        
        for strategy in strategies:
            self.benchmark_strategy(strategy, queries, limit, warmup_queries)
        
        return self.results
    
    def evaluate_accuracy(
        self,
        strategy: RetrievalStrategy,
        queries: List[str],
        ground_truth: Dict[str, List[str]],
        limit: int = 5
    ):
        """
        Evaluate accuracy using ground truth.
        
        Args:
            strategy: Strategy to evaluate
            queries: Test queries
            ground_truth: Dict mapping query to list of relevant doc IDs/titles
            limit: Number of results to evaluate
        """
        logger.info(f"Evaluating accuracy for {strategy.name}")
        
        recall_at_5 = []
        recall_at_10 = []
        precision_at_5 = []
        
        for query in queries:
            if query not in ground_truth:
                continue
            
            # Get results
            query_vec = self.embedder.embed_text(query)
            results, _ = strategy.search(query_vec, query, limit=10)
            
            # Extract titles
            retrieved_titles = [r["title"] for r in results]
            relevant_titles = set(ground_truth[query])
            
            # Calculate metrics
            retrieved_at_5 = set(retrieved_titles[:5])
            retrieved_at_10 = set(retrieved_titles[:10])
            
            # Recall@k = |relevant ∩ retrieved@k| / |relevant|
            recall_5 = len(relevant_titles & retrieved_at_5) / len(relevant_titles) if relevant_titles else 0
            recall_10 = len(relevant_titles & retrieved_at_10) / len(relevant_titles) if relevant_titles else 0
            
            # Precision@5 = |relevant ∩ retrieved@5| / 5
            precision_5 = len(relevant_titles & retrieved_at_5) / 5
            
            recall_at_5.append(recall_5)
            recall_at_10.append(recall_10)
            precision_at_5.append(precision_5)
        
        # Update benchmark with accuracy metrics
        if strategy.name in self.results:
            self.results[strategy.name].recall_at_5 = statistics.mean(recall_at_5) if recall_at_5 else None
            self.results[strategy.name].recall_at_10 = statistics.mean(recall_at_10) if recall_at_10 else None
            self.results[strategy.name].precision_at_5 = statistics.mean(precision_at_5) if precision_at_5 else None
            
            logger.info(f"  Recall@5: {self.results[strategy.name].recall_at_5:.2%}")
            logger.info(f"  Recall@10: {self.results[strategy.name].recall_at_10:.2%}")
            logger.info(f"  Precision@5: {self.results[strategy.name].precision_at_5:.2%}")
    
    def print_comparison_table(self):
        """Print comparison table of all benchmarked strategies."""
        if not self.results:
            logger.warning("No results to display")
            return
        
        print("\n" + "=" * 120)
        print("RETRIEVAL STRATEGY COMPARISON")
        print("=" * 120)
        
        # Header
        print(f"{'Strategy':<25} {'Avg(ms)':<10} {'Median':<10} {'P95':<10} {'P99':<10} {'QPS':<10} {'Recall@5':<10}")
        print("-" * 120)
        
        # Sort by average latency
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].avg_latency_ms)
        
        for name, benchmark in sorted_results:
            recall_str = f"{benchmark.recall_at_5:.2%}" if benchmark.recall_at_5 is not None else "N/A"
            print(
                f"{name:<25} "
                f"{benchmark.avg_latency_ms:<10.2f} "
                f"{benchmark.median_latency_ms:<10.2f} "
                f"{benchmark.p95_latency_ms:<10.2f} "
                f"{benchmark.p99_latency_ms:<10.2f} "
                f"{benchmark.queries_per_second:<10.2f} "
                f"{recall_str:<10}"
            )
        
        print("=" * 120)
        
        # Detailed breakdown
        print("\nDETAILED BREAKDOWN (avg milliseconds)")
        print("-" * 120)
        print(f"{'Strategy':<25} {'Embedding':<12} {'Search':<12} {'Postprocess':<12} {'Total':<12}")
        print("-" * 120)
        
        for name, benchmark in sorted_results:
            print(
                f"{name:<25} "
                f"{benchmark.avg_embedding_time_ms:<12.2f} "
                f"{benchmark.avg_search_time_ms:<12.2f} "
                f"{benchmark.avg_postprocess_time_ms:<12.2f} "
                f"{benchmark.avg_latency_ms:<12.2f}"
            )
        
        print("=" * 120 + "\n")
        
        # Find fastest
        fastest = sorted_results[0]
        print(f"🏆 FASTEST: {fastest[0]} ({fastest[1].avg_latency_ms:.2f}ms avg)")
        
        # Speedup comparison
        baseline = next((b for n, b in sorted_results if "StandardHNSW" in n), None)
        if baseline:
            print(f"\n📊 SPEEDUP vs StandardHNSW baseline:")
            for name, benchmark in sorted_results:
                if name != baseline.strategy_name:
                    speedup = baseline.avg_latency_ms / benchmark.avg_latency_ms
                    print(f"  {name}: {speedup:.2f}x faster")
        
        print()
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output = {
            name: asdict(benchmark)
            for name, benchmark in self.results.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
