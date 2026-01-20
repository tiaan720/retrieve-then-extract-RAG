from typing import List


class TextExtractor:
    """Extracts and cleans text from documents."""
    
    def __init__(self):
        """Initialize the text extractor."""
        pass
    
    def extract_and_clean(self, text: str) -> str:
        """
        Clean text content with basic normalization.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic text normalization
        lines = text.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
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



