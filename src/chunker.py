"""
Document chunking module to split documents into smaller chunks.
"""
from typing import List, Dict


class DocumentChunker:
    """Chunks documents into smaller pieces for embedding."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks with sentence boundary detection.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If chunk_overlap >= chunk_size
        """
        if not text:
            return []
        
        # Validate configuration to prevent infinite loops
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Look for sentence endings
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_boundary = max(last_period, last_newline)
                
                if last_boundary > self.chunk_size // 2:
                    chunk = text[start:start + last_boundary + 1]
                    end = start + last_boundary + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move to next chunk with overlap
            next_start = end - self.chunk_overlap
            
            # Ensure we're making progress to avoid infinite loops
            if next_start <= start:
                next_start = start + 1
            
            start = next_start
            
            # Stop if we've reached or passed the end of text
            if end >= len(text):
                break
        
        return chunks
    
    def chunk_document(self, doc: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Chunk a single document into multiple chunks.
        
        Args:
            doc: Document dictionary with 'content', 'title', and 'url'
            
        Returns:
            List of chunk dictionaries with metadata
        """
        content = doc.get('content', '')
        text_chunks = self.chunk_text(content)
        
        chunked_docs = []
        for idx, chunk in enumerate(text_chunks):
            chunked_docs.append({
                'content': chunk,
                'title': doc.get('title', ''),
                'url': doc.get('url', ''),
                'chunk_index': idx,
                'total_chunks': len(text_chunks)
            })
        
        return chunked_docs
    
    def chunk_documents(self, docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk multiple documents.
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
