"""
Document fetcher module to retrieve documentation from open-source libraries.
"""
import requests
from typing import List, Dict
from bs4 import BeautifulSoup


class DocumentFetcher:
    """Fetches documentation from open-source library websites."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DocumentFetcher/1.0)'
        })
    
    def fetch_langchain_docs(self, max_docs: int = 10) -> List[Dict[str, str]]:
        """
        Fetch documentation from LangChain docs as an example.
        
        Args:
            max_docs: Maximum number of documents to fetch
            
        Returns:
            List of dictionaries with 'title', 'url', and 'content' keys
        """
        docs = []
        
        # Sample URLs from LangChain documentation
        base_urls = [
            "https://python.langchain.com/docs/get_started/introduction",
            "https://python.langchain.com/docs/get_started/quickstart",
            "https://python.langchain.com/docs/concepts/",
            "https://python.langchain.com/docs/tutorials/",
        ]
        
        for url in base_urls[:max_docs]:
            try:
                print(f"Fetching: {url}")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text() if title else url
                
                # Extract main content (adjust selector based on actual site structure)
                content = ""
                
                # Try common content containers
                main_content = (
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_='content') or
                    soup.find('body')
                )
                
                if main_content:
                    # Remove script and style tags
                    for tag in main_content(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    
                    content = main_content.get_text(separator='\n', strip=True)
                
                if content:
                    docs.append({
                        'title': title_text,
                        'url': url,
                        'content': content
                    })
                    print(f"Successfully fetched: {title_text}")
                    
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue
        
        return docs
    
    def fetch_custom_docs(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Fetch documentation from custom URLs.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of dictionaries with 'title', 'url', and 'content' keys
        """
        docs = []
        
        for url in urls:
            try:
                print(f"Fetching: {url}")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text() if title else url
                
                # Extract main content
                main_content = (
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_='content') or
                    soup.find('body')
                )
                
                if main_content:
                    for tag in main_content(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    
                    content = main_content.get_text(separator='\n', strip=True)
                    
                    docs.append({
                        'title': title_text,
                        'url': url,
                        'content': content
                    })
                    print(f"Successfully fetched: {title_text}")
                    
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue
        
        return docs
