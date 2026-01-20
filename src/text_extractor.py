"""
Text extraction and preprocessing module.
"""
import re
from typing import List


class TextExtractor:
    """Extracts and cleans text from raw document content."""
    
    def __init__(self):
        pass
    
    def extract_and_clean(self, text: str) -> str:
        """
        Extract and clean text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_from_document(self, doc: dict) -> dict:
        """
        Extract and clean text from a document dictionary.
        
        Args:
            doc: Document dictionary with 'content' key
            
        Returns:
            Document dictionary with cleaned content
        """
        cleaned_doc = doc.copy()
        if 'content' in cleaned_doc:
            cleaned_doc['content'] = self.extract_and_clean(cleaned_doc['content'])
        
        return cleaned_doc
    
    def extract_from_documents(self, docs: List[dict]) -> List[dict]:
        """
        Extract and clean text from multiple documents.
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            List of document dictionaries with cleaned content
        """
        return [self.extract_from_document(doc) for doc in docs]
