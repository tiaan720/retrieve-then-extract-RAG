"""
Pytest tests to validate retrieval strategies implementation.

This tests the core components without requiring a full benchmark run.

Run with: pytest test_retrieval_strategies.py -v
"""

import pytest
from src.retrieval_strategies import (
    RetrievalStrategy,
    RetrievalMetrics,
    StandardHNSW,
    BinaryQuantized,
    HybridSearch,
    CrossEncoderRerank,
    BinaryInt8Staged
)
from src.collection_manager import CollectionManager, CollectionConfig
from src.evaluator import RetrievalEvaluator


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""
    
    def test_retrieval_metrics_creation(self):
        """Test RetrievalMetrics can be created with correct values."""
        metrics = RetrievalMetrics(
            query_time_ms=100.0,
            embedding_time_ms=50.0,
            search_time_ms=30.0,
            postprocess_time_ms=5.0,
            total_time_ms=185.0,
            results_count=5,
            strategy_name="TestStrategy"
        )
        
        assert metrics.strategy_name == "TestStrategy"
        assert metrics.total_time_ms == 185.0
        assert metrics.query_time_ms == 100.0
        assert metrics.embedding_time_ms == 50.0
        assert metrics.search_time_ms == 30.0
        assert metrics.postprocess_time_ms == 5.0
        assert metrics.results_count == 5


class TestCollectionConfig:
    """Tests for CollectionConfig."""
    
    def test_collection_config_creation(self):
        """Test CollectionConfig can be created with correct values."""
        config = CollectionConfig(
            name="TestCollection",
            enable_binary_quantization=True,
            enable_reranker=False,
            ef_construction=64,
            max_connections=32,
            description="Test config"
        )
        
        assert config.name == "TestCollection"
        assert config.enable_binary_quantization is True
        assert config.enable_reranker is False
        assert config.ef_construction == 64
        assert config.max_connections == 32
        assert config.description == "Test config"


class TestCollectionManager:
    """Tests for CollectionManager."""
    
    def test_all_predefined_configs_exist(self):
        """Test that all expected strategy configs are defined."""
        expected_strategies = [
            "StandardHNSW",
            "BinaryQuantized",
            "HybridSearch",
            "CrossEncoderRerank",
            "BinaryInt8Staged"
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in CollectionManager.CONFIGS, f"Missing config: {strategy_name}"
            config = CollectionManager.CONFIGS[strategy_name]
            assert config.name.startswith("Document_")
            assert config.description != ""
    
    def test_binary_int8_staged_has_optimized_settings(self):
        """Test BinaryInt8Staged has correct optimized settings."""
        binary_int8_config = CollectionManager.CONFIGS["BinaryInt8Staged"]
        
        assert binary_int8_config.enable_binary_quantization is True
        assert binary_int8_config.ef_construction == 64  # Optimized (vs 128 standard)
        assert binary_int8_config.max_connections == 32  # Optimized (vs 64 standard)


class TestRetrievalStrategies:
    """Tests for retrieval strategy classes."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock collection object for testing."""
        class MockCollection:
            def __init__(self):
                self.query = None
        return MockCollection()
    
    def test_standard_hnsw_instantiation(self, mock_collection):
        """Test StandardHNSW strategy can be instantiated."""
        strategy = StandardHNSW(mock_collection)
        
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.name == "StandardHNSW"
        assert hasattr(strategy, 'search')
    
    def test_binary_quantized_instantiation(self, mock_collection):
        """Test BinaryQuantized strategy can be instantiated."""
        strategy = BinaryQuantized(mock_collection)
        
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.name == "BinaryQuantized"
        assert hasattr(strategy, 'search')
    
    def test_hybrid_search_instantiation(self, mock_collection):
        """Test HybridSearch strategy can be instantiated."""
        strategy = HybridSearch(mock_collection, alpha=0.7)
        
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.name == "HybridSearch"
        assert strategy.alpha == 0.7
        assert hasattr(strategy, 'search')
    
    def test_cross_encoder_rerank_instantiation(self, mock_collection):
        """Test CrossEncoderRerank strategy can be instantiated."""
        strategy = CrossEncoderRerank(mock_collection, rerank_multiplier=4)
        
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.name == "CrossEncoderRerank"
        assert strategy.rerank_multiplier == 4
        assert hasattr(strategy, 'search')
    
    def test_binary_int8_staged_instantiation(self, mock_collection):
        """Test BinaryInt8Staged strategy can be instantiated."""
        strategy = BinaryInt8Staged(mock_collection, rescore_multiplier=4)
        
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.name == "BinaryInt8Staged"
        assert strategy.rescore_multiplier == 4
        assert hasattr(strategy, 'search')
    
    def test_all_strategies_inherit_from_base(self, mock_collection):
        """Test all strategies inherit from RetrievalStrategy base class."""
        strategies = [
            StandardHNSW(mock_collection),
            BinaryQuantized(mock_collection),
            HybridSearch(mock_collection, alpha=0.7),
            CrossEncoderRerank(mock_collection, rerank_multiplier=4),
            BinaryInt8Staged(mock_collection, rescore_multiplier=4)
        ]
        
        for strategy in strategies:
            assert isinstance(strategy, RetrievalStrategy)
            assert hasattr(strategy, 'name')
            assert hasattr(strategy, 'search')
            assert hasattr(strategy, 'collection')


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder for testing."""
        class MockEmbedder:
            def embed_text(self, text):
                return [0.1] * 384
        return MockEmbedder()
    
    def test_evaluator_instantiation(self, mock_embedder):
        """Test RetrievalEvaluator can be instantiated."""
        evaluator = RetrievalEvaluator(mock_embedder)
        
        assert evaluator.embedder is mock_embedder
        assert evaluator.results == {}
    
    def test_evaluator_has_required_methods(self, mock_embedder):
        """Test RetrievalEvaluator has all required methods."""
        evaluator = RetrievalEvaluator(mock_embedder)
        
        assert hasattr(evaluator, 'benchmark_strategy')
        assert hasattr(evaluator, 'benchmark_all_strategies')
        assert hasattr(evaluator, 'evaluate_accuracy')
        assert hasattr(evaluator, 'print_comparison_table')
        assert hasattr(evaluator, 'save_results')
