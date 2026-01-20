"""
Quick test to validate retrieval strategies implementation.

This tests the core components without requiring a full benchmark run.
"""

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


def test_retrieval_metrics():
    """Test RetrievalMetrics dataclass."""
    print("Testing RetrievalMetrics...")
    
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
    print("✓ RetrievalMetrics works")


def test_collection_config():
    """Test CollectionConfig."""
    print("\nTesting CollectionConfig...")
    
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
    print("✓ CollectionConfig works")


def test_collection_manager_configs():
    """Test CollectionManager predefined configs."""
    print("\nTesting CollectionManager configs...")
    
    # Check all predefined configs exist
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
    
    print(f"✓ All {len(expected_strategies)} strategy configs defined")
    
    # Verify BinaryInt8Staged has correct settings
    binary_int8_config = CollectionManager.CONFIGS["BinaryInt8Staged"]
    assert binary_int8_config.enable_binary_quantization is True
    assert binary_int8_config.ef_construction == 64  # Optimized
    assert binary_int8_config.max_connections == 32  # Optimized
    print("✓ BinaryInt8Staged has optimized settings")


def test_strategy_classes():
    """Test that all strategy classes can be instantiated."""
    print("\nTesting strategy classes...")
    
    # Mock collection object
    class MockCollection:
        def __init__(self):
            self.query = None
    
    mock_collection = MockCollection()
    
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
        print(f"  ✓ {strategy.name} instantiated")
    
    print(f"✓ All {len(strategies)} strategies instantiable")


def test_evaluator_instantiation():
    """Test RetrievalEvaluator can be instantiated."""
    print("\nTesting RetrievalEvaluator...")
    
    # Mock embedder
    class MockEmbedder:
        def embed_text(self, text):
            return [0.1] * 384
    
    evaluator = RetrievalEvaluator(MockEmbedder())
    assert evaluator.results == {}
    print("✓ RetrievalEvaluator instantiated")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RETRIEVAL STRATEGIES VALIDATION TESTS")
    print("=" * 60)
    
    try:
        test_retrieval_metrics()
        test_collection_config()
        test_collection_manager_configs()
        test_strategy_classes()
        test_evaluator_instantiation()
        
        print("\n" + "=" * 60)
        print("ALL VALIDATION TESTS PASSED! ✓")
        print("=" * 60)
        print("\nFramework is ready to use.")
        print("Run: python compare_strategies.py")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
