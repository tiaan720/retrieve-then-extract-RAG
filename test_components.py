from src.text_extractor import TextExtractor
from src.chunker import DocumentChunker


def test_text_extractor():
    """Test text extraction and cleaning."""
    print("Testing TextExtractor...")
    
    extractor = TextExtractor()
    
    # Test cleaning
    dirty_text = "This  is   a    test\n\n\n\nwith extra    spaces"
    clean_text = extractor.extract_and_clean(dirty_text)
    
    assert "  " not in clean_text, "Should remove extra spaces"
    print("✓ Text cleaning works")
    
    # Test document extraction
    doc = {
        "title": "Test Doc",
        "url": "http://test.com",
        "content": "Some   content   here"
    }
    cleaned_doc = extractor.extract_from_document(doc)
    assert "Some content here" in cleaned_doc["content"]
    print("✓ Document extraction works")
    
    print("TextExtractor tests passed!\n")


def test_chunker():
    """Test document chunking."""
    print("Testing DocumentChunker...")
    
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    
    # Test text chunking
    text = "This is a test sentence. " * 10  # Long text
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 1, "Should create multiple chunks"
    print(f"✓ Created {len(chunks)} chunks from text")
    
    # Test document chunking
    doc = {
        "title": "Test Document",
        "url": "http://test.com",
        "content": "A" * 200  # 200 character content
    }
    chunked_docs = chunker.chunk_document(doc)
    
    assert len(chunked_docs) > 0, "Should create chunks"
    assert all("chunk_index" in chunk for chunk in chunked_docs), "Should have chunk metadata"
    print(f"✓ Document chunking works, created {len(chunked_docs)} chunks")
    
    # Test edge case: chunk_overlap >= chunk_size should raise error
    try:
        bad_chunker = DocumentChunker(chunk_size=50, chunk_overlap=50)
        bad_chunker.chunk_text("Some text here")
        assert False, "Should raise ValueError for invalid overlap"
    except ValueError as e:
        print("✓ Correctly raises error for invalid chunk_overlap >= chunk_size")
    
    print("DocumentChunker tests passed!\n")


def test_integration():
    """Test integration of components."""
    print("Testing component integration...")
    
    extractor = TextExtractor()
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    # Create sample documents
    docs = [
        {
            "title": "Doc 1",
            "url": "http://example.com/doc1",
            "content": "This is the first document. " * 20
        },
        {
            "title": "Doc 2",
            "url": "http://example.com/doc2",
            "content": "This is the second document. " * 20
        }
    ]
    
    # Extract and clean
    cleaned_docs = extractor.extract_from_documents(docs)
    assert len(cleaned_docs) == 2, "Should have 2 cleaned docs"
    print("✓ Cleaned documents")
    
    # Chunk
    chunks = chunker.chunk_documents(cleaned_docs)
    assert len(chunks) > 0, "Should have chunks"
    print(f"✓ Created {len(chunks)} total chunks from {len(docs)} documents")
    
    # Verify chunk structure
    for chunk in chunks:
        assert "content" in chunk
        assert "title" in chunk
        assert "url" in chunk
        assert "chunk_index" in chunk
        assert "total_chunks" in chunk
    print("✓ All chunks have required metadata")
    
    print("Integration tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Component Tests")
    print("=" * 60 + "\n")
    
    try:
        test_text_extractor()
        test_chunker()
        test_integration()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
