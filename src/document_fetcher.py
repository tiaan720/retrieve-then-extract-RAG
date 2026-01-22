import wikipedia
from typing import List, Dict
from src.logger import logger


class DocumentFetcher:
    """Fetches documents from Wikipedia for RAG testing."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the Wikipedia document fetcher.
        
        Args:
            language: Wikipedia language code (default: 'en')
        """
        self.language = language
        wikipedia.set_lang(language)
    
    def fetch_wikipedia_articles(self, topics: List[str], max_docs: int = 10) -> List[Dict[str, str]]:
        """
        Fetch Wikipedia articles for given topics.
        
        Args:
            topics: List of Wikipedia article topics to fetch
            max_docs: Maximum number of documents to fetch
            
        Returns:
            List of dictionaries with 'title', 'url', and 'content' keys
        """
        docs = []
        
        for topic in topics[:max_docs]:
            try:
                logger.info(f"Fetching Wikipedia article: {topic}")
                
                # Search for the topic and get the best match
                search_results = wikipedia.search(topic, results=10)
                if not search_results:
                    logger.warning(f"No results found for: {topic}")
                    continue
                
                page_title = search_results[0]
                page = wikipedia.page(page_title, auto_suggest=False)
                
                docs.append({
                    'title': page.title,
                    'url': page.url,
                    'content': page.content,
                    'source': 'wikipedia',
                    'language': self.language
                })
                logger.info(f"Successfully fetched: {page.title}")
                
            except wikipedia.exceptions.DisambiguationError as e:
                logger.warning(f"Disambiguation for '{topic}', using: {e.options[0]}")
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    docs.append({
                        'title': page.title,
                        'url': page.url,
                        'content': page.content,
                        'source': 'wikipedia',
                        'language': self.language
                    })
                    logger.info(f"Successfully fetched: {page.title}")
                except Exception as inner_e:
                    logger.error(f"Error fetching disambiguation option: {inner_e}")
                    
            except wikipedia.exceptions.PageError:
                logger.warning(f"Page not found: {topic}")
                
            except Exception as e:
                logger.error(f"Error fetching {topic}: {e}")
                continue
        
        return docs
    
    def fetch_random_articles(self, count: int = 5) -> List[Dict[str, str]]:
        """
        Fetch random Wikipedia articles.
        
        Args:
            count: Number of random articles to fetch
            
        Returns:
            List of dictionaries with 'title', 'url', and 'content' keys
        """
        docs = []
        
        try:
            random_titles = wikipedia.random(count)
            if isinstance(random_titles, str):
                random_titles = [random_titles]
                
            for title in random_titles:
                try:
                    logger.info(f"Fetching random article: {title}")
                    page = wikipedia.page(title, auto_suggest=False)
                    docs.append({
                        'title': page.title,
                        'url': page.url,
                        'content': page.content,
                        'source': 'wikipedia',
                        'language': self.language
                    })
                    logger.info(f"Successfully fetched: {page.title}")
                except Exception as e:
                    logger.error(f"Error fetching {title}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching random articles: {e}")
        
        return docs

