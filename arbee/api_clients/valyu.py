"""
Valyu AI Client for LangChain integration
Provides research and evidence gathering capabilities
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from config import settings

# Optional imports - fallback to HTTP if not available
LANGCHAIN_AVAILABLE = False
ValyuSearchTool = None
ValyuContentsTool = None

class ValyuResearchClient:
    """Client for Valyu AI research via LangChain tools"""

    def __init__(self):
        """Initialize Valyu client with API key"""
        os.environ['VALYU_API_KEY'] = settings.VALYU_API_KEY
        self._init_tools()

    def _init_tools(self):
        """Initialize LangChain Valyu tools"""
        try:
            if LANGCHAIN_AVAILABLE and ValyuSearchTool is not None:
                self.search_tool = ValyuSearchTool()
                self.contents_tool = ValyuContentsTool()
            else:
                print("Warning: Valyu LangChain tools not available. Using httpx fallback.")
                self._init_http_fallback()

        except Exception as e:
            print(f"Warning: Could not initialize Valyu tools: {e}")
            self._init_http_fallback()

    def _init_http_fallback(self):
        """Initialize basic HTTP client as fallback"""
        import httpx
        self.http_client = httpx.AsyncClient(
            base_url="https://api.valyu.ai",  # Placeholder URL
            headers={
                "Authorization": f"Bearer {settings.VALYU_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

    async def search(
        self,
        query: str,
        search_type: str = "all",
        num_results: int = 10,
        date_range: Optional[Dict[str, str]] = None,
        relevance_threshold: float = 0.5,
        max_cost: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Perform deep search using Valyu

        Args:
            query: Search query string
            search_type: "all", "web", or "proprietary"
            num_results: Max number of results
            date_range: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
            relevance_threshold: Minimum relevance score (0-1)
            max_cost: Maximum cost in dollars

        Returns:
            List of search result dicts
        """
        if self.search_tool:
            # Use LangChain tool
            try:
                results = await self.search_tool.arun(
                    query=query,
                    search_type=search_type,
                    num_results=num_results,
                    relevance_threshold=relevance_threshold,
                    max_cost=max_cost
                )
                return self._parse_search_results(results)
            except Exception as e:
                print(f"Error using Valyu search tool: {e}")
                return []
        else:
            # Fallback to direct HTTP (if API supports it)
            return await self._http_search(query, num_results, date_range)

    async def _http_search(
        self,
        query: str,
        num_results: int,
        date_range: Optional[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Fallback HTTP search implementation

        Note: This is a placeholder. Actual implementation depends on Valyu's REST API
        """
        if not hasattr(self, 'http_client'):
            return []

        try:
            payload = {
                "query": query,
                "limit": num_results,
                "date_range": date_range
            }

            response = await self.http_client.post("/search", json=payload)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            print(f"Error in HTTP fallback search: {e}")
            return []

    def _parse_search_results(self, results: Any) -> List[Dict[str, Any]]:
        """
        Parse search results into standardized format

        Args:
            results: Raw results from Valyu

        Returns:
            List of normalized result dicts
        """
        if isinstance(results, str):
            # If results are string, attempt to parse
            import json
            try:
                results = json.loads(results)
            except:
                return []

        if not isinstance(results, list):
            return []

        normalized = []
        for item in results:
            normalized.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'snippet': item.get('snippet', item.get('content', '')),
                'published_date': item.get('date', item.get('published_date', '')),
                'source': item.get('source', ''),
                'relevance_score': item.get('relevance', 0.0)
            })

        return normalized

    async def get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured content from a URL

        Args:
            url: URL to extract content from

        Returns:
            Dict with extracted content or None
        """
        if self.contents_tool:
            try:
                content = await self.contents_tool.arun(url=url)
                return self._parse_content(content)
            except Exception as e:
                print(f"Error extracting content from {url}: {e}")
                return None
        else:
            return await self._http_get_content(url)

    async def _http_get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fallback HTTP content extraction"""
        if not hasattr(self, 'http_client'):
            return None

        try:
            response = await self.http_client.post(
                "/extract",
                json={"url": url}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error in HTTP content extraction: {e}")
            return None

    def _parse_content(self, content: Any) -> Optional[Dict[str, Any]]:
        """Parse extracted content into standardized format"""
        if isinstance(content, str):
            return {
                'text': content,
                'extracted_at': datetime.utcnow().isoformat()
            }
        elif isinstance(content, dict):
            return content
        return None

    async def search_with_date_filter(
        self,
        query: str,
        days_back: int = 90,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search with automatic date filtering (recent content only)

        Args:
            query: Search query
            days_back: How many days back to search
            num_results: Max results

        Returns:
            List of recent search results
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        date_range = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }

        return await self.search(
            query=query,
            num_results=num_results,
            date_range=date_range
        )

    async def multi_query_search(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute multiple search queries in parallel

        Args:
            queries: List of search queries
            max_results_per_query: Max results per query

        Returns:
            Dict mapping query -> results
        """
        import asyncio

        tasks = [
            self.search(q, num_results=max_results_per_query)
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            query: result if not isinstance(result, Exception) else []
            for query, result in zip(queries, results)
        }
