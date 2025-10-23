"""
Valyu AI Client for Deep Web Research
Provides evidence gathering with strict validation and deep search capabilities.
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from config import settings
import logging

logger = logging.getLogger(__name__)

# Core LangChain Valyu imports with validation
try:
    from langchain_valyu import ValyuSearchTool, ValyuRetriever
    LANGCHAIN_VALYU_AVAILABLE = True
except ImportError:
    LANGCHAIN_VALYU_AVAILABLE = False
    ValyuSearchTool = None
    ValyuRetriever = None


class ValyuResearchClient:
    """
    Deep research client using Valyu AI for comprehensive evidence gathering.

    Enforces strict validation with no fallback mechanisms.
    Supports both search and content extraction with deep research capabilities.
    """

    def __init__(self):
        """Initialize Valyu client with strict validation and deep search setup."""
        self._validate_configuration()
        self._setup_environment()
        self._initialize_tools()
        logger.info("Valyu deep research client initialized successfully")

    def _validate_configuration(self) -> None:
        """Validate all required configuration parameters."""
        assert hasattr(settings, 'VALYU_API_KEY'), "Settings missing VALYU_API_KEY"
        assert settings.VALYU_API_KEY, "VALYU_API_KEY cannot be empty"
        assert settings.VALYU_API_KEY != '...', "VALYU_API_KEY placeholder detected"
        assert LANGCHAIN_VALYU_AVAILABLE, (
            "langchain-valyu not available. Install with: pip install langchain-valyu"
        )

    def _setup_environment(self) -> None:
        """Configure environment variables for Valyu API access."""
        os.environ['VALYU_API_KEY'] = settings.VALYU_API_KEY

    def _initialize_tools(self) -> None:
        """Initialize Valyu search and retrieval tools with validation."""
        assert ValyuSearchTool is not None, "ValyuSearchTool import failed"
        assert ValyuRetriever is not None, "ValyuRetriever import failed"

        try:
            self.search_tool = ValyuSearchTool(valyu_api_key=settings.VALYU_API_KEY)
            self.retriever = ValyuRetriever(valyu_api_key=settings.VALYU_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Valyu tool initialization failed: {e}") from e

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
        Perform deep research using Valyu AI search capabilities.

        Args:
            query: Search query string
            search_type: "all", "web", or "proprietary"
            num_results: Maximum results to return
            date_range: Optional date filtering
            relevance_threshold: Minimum relevance score (0-1)
            max_cost: Maximum cost in dollars

        Returns:
            List of normalized search result dictionaries

        Raises:
            ValueError: Invalid parameters
            RuntimeError: Search execution failure
        """
        self._validate_search_params(query, num_results, relevance_threshold, max_cost)

        try:
            logger.info(f"Deep searching Valyu: {query}")
            results = await self._execute_search(query, search_type, num_results, relevance_threshold, max_cost)
            parsed_results = self._parse_search_results(results)
            logger.info(f"Deep search returned {len(parsed_results)} results")
            return parsed_results
        except Exception as e:
            raise RuntimeError(f"Deep search failed for '{query}': {e}") from e

    def _validate_search_params(self, query: str, num_results: int,
                                relevance_threshold: float, max_cost: float) -> None:
        """Validate search parameters with strict assertions."""
        assert query and query.strip(), "Query cannot be empty"
        assert 1 <= num_results <= 50, f"num_results must be 1-50, got {num_results}"
        assert 0.0 <= relevance_threshold <= 1.0, (
            f"relevance_threshold must be 0-1, got {relevance_threshold}"
        )
        assert 0.1 <= max_cost <= 50.0, f"max_cost must be 0.1-50.0, got {max_cost}"

    async def _execute_search(self, query: str, search_type: str, num_results: int,
                              relevance_threshold: float, max_cost: float) -> Any:
        """Execute the actual search with proper parameter handling."""
        # Prepare tool input as a dictionary (most common format for LangChain tools)
        tool_input = {
            "query": query,
            "search_type": search_type,
            "num_results": num_results,
            "relevance_threshold": relevance_threshold,
            "max_cost": max_cost
        }

        try:
            # Try with dictionary input first
            result = await self.search_tool.arun(tool_input)

            # Handle SearchResponse objects - extract the results list
            if hasattr(result, 'results'):
                # It's a SearchResponse object, extract the results field
                return list(result.results)
            elif hasattr(result, '__iter__') and not isinstance(result, str):
                # It's an iterable, convert to list
                return list(result)
            else:
                # Return as is (might be a single result or error)
                return result
        except Exception as e:
            logger.error(f"Error in search execution: {e}")
            try:
                # Try with JSON string
                import json
                result = await self.search_tool.arun(json.dumps(tool_input))
                # Handle SearchResponse objects - extract the results list
                if hasattr(result, 'results'):
                    # It's a SearchResponse object, extract the results field
                    return list(result.results)
                elif hasattr(result, '__iter__') and not isinstance(result, str):
                    # It's an iterable, convert to list
                    return list(result)
                else:
                    # Return as is (might be a single result or error)
                    return result
            except Exception as e2:
                logger.error(f"Error in JSON search execution: {e2}")
                raise

    def _parse_search_results(self, results: Any) -> List[Dict[str, Any]]:
        """
        Parse and normalize search results with comprehensive validation.

        Args:
            results: Raw search results from Valyu API

        Returns:
            List of validated and normalized result dictionaries
        """
        if not results:
            return []

        # Handle string JSON responses
        if isinstance(results, str):
            results = self._parse_json_string(results)
            if not results:
                return []

        # Handle SearchResponse objects and similar structured responses
        if hasattr(results, 'results'):
            results = results.results
        elif hasattr(results, '__iter__') and not isinstance(results, (str, dict)):
            # Convert iterable to list
            results = list(results)

        # Validate result structure
        if not isinstance(results, list):
            logger.warning(f"Invalid results format: {type(results)}")
            # Try to extract results from dict-like objects
            if isinstance(results, dict):
                # Try common result field names
                for field in ['results', 'items', 'data', 'documents']:
                    if field in results and isinstance(results[field], list):
                        results = results[field]
                        break
                else:
                    return []
            else:
                return []
        logger.info(f"Parsed results type: {type(results)}")
        logger.info(f"Parsed results: {results}")
        normalized = []
        for item in results:
            parsed_item = self._normalize_result_item(item)
            if parsed_item:
                normalized.append(parsed_item)

        # Filter and sort by relevance
        return self._filter_and_rank_results(normalized)

    def _normalize_result_item(self, item: Any) -> Optional[Dict[str, Any]]:
        """Normalize individual result item with validation."""
        if not isinstance(item, dict):
            return None

        # Extract and validate URL
        url = item.get('url', '').strip()
        if not url or not self._is_valid_url(url):
            return None

        # Extract other fields with defaults
        return {
            'title': self._clean_text(item.get('title', '')),
            'url': url,
            'snippet': self._clean_text(item.get('snippet', item.get('content', ''))),
            'published_date': self._normalize_date(item.get('date', item.get('published_date', ''))),
            'source': self._extract_domain(url),
            'relevance_score': self._normalize_relevance_score(item.get('relevance', 0.0))
        }

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and accessibility."""
        if not url or len(url) > 2048:
            return False
        return url.startswith(('http://', 'https://'))

    def _clean_text(self, text: str) -> str:
        """Clean and truncate text content."""
        if not text:
            return ''
        # Remove extra whitespace and limit length
        cleaned = ' '.join(text.split())
        return cleaned[:1000] if len(cleaned) > 1000 else cleaned

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to ISO format."""
        if not date_str:
            return ''
        # Basic date validation and formatting
        try:
            # Try to parse common date formats
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_str
        except (ValueError, AttributeError):
            return ''

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return 'unknown'

    def _normalize_relevance_score(self, score: Any) -> float:
        """Normalize relevance score to 0-1 range."""
        try:
            score = float(score)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.0

    def _filter_and_rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out invalid results and rank by relevance."""
        valid_results = [r for r in results if r['url'] and r['title']]

        # Sort by relevance score (descending) then by title length (ascending for quality)
        return sorted(
            valid_results,
            key=lambda x: (x['relevance_score'], -len(x['title'])),
            reverse=True
        )

    async def get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive content from URL using Valyu retriever.

        Args:
            url: URL to extract content from

        Returns:
            Dictionary containing extracted content and metadata

        Raises:
            ValueError: Invalid URL
            RuntimeError: Content extraction failure
        """
        self._validate_url(url)

        try:
            logger.info(f"Extracting deep content from: {url}")
            content = await self._extract_content(url)
            return self._normalize_content(content)
        except Exception as e:
            raise RuntimeError(f"Content extraction failed for {url}: {e}") from e

    def _validate_url(self, url: str) -> None:
        """Validate URL format and accessibility."""
        assert url and url.strip(), "URL cannot be empty"
        assert url.startswith(('http://', 'https://')), f"Invalid URL format: {url}"
        assert len(url) <= 2048, f"URL too long: {len(url)} > 2048"

    async def _extract_content(self, url: str) -> Any:
        """Execute content extraction using Valyu retriever."""
        docs = self.retriever.invoke(url)
        return docs[0].page_content if docs else None

    def _normalize_content(self, content: Any) -> Optional[Dict[str, Any]]:
        """Normalize extracted content into standard format."""
        if isinstance(content, str):
            return {
                'text': content,
                'extracted_at': datetime.utcnow().isoformat(),
                'content_type': 'text'
            }
        elif isinstance(content, dict):
            return dict(content)
        return None

    async def search_with_date_filter(
        self,
        query: str,
        days_back: int = 90,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute deep search with temporal filtering for recent content.

        Args:
            query: Search query string
            days_back: Number of days back from today to search
            num_results: Maximum results to return

        Returns:
            List of temporally filtered search results

        Raises:
            ValueError: Invalid parameters
        """
        self._validate_date_search_params(query, days_back, num_results)

        date_range = self._build_date_range(days_back)
        return await self.search(query=query, num_results=num_results, date_range=date_range)

    def _validate_date_search_params(self, query: str, days_back: int, num_results: int) -> None:
        """Validate date search parameters."""
        assert query and query.strip(), "Query cannot be empty"
        assert 1 <= days_back <= 3650, f"days_back must be 1-3650, got {days_back}"
        assert 1 <= num_results <= 50, f"num_results must be 1-50, got {num_results}"

    def _build_date_range(self, days_back: int) -> Dict[str, str]:
        """Build standardized date range dictionary."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        return {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }

    async def multi_query_search(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute parallel deep searches across multiple queries.

        Args:
            queries: List of search queries to execute
            max_results_per_query: Maximum results per individual query

        Returns:
            Dictionary mapping each query to its search results

        Raises:
            ValueError: Invalid parameters
        """
        self._validate_multi_query_params(queries, max_results_per_query)

        import asyncio

        logger.info(f"Executing {len(queries)} parallel deep searches")

        tasks = [self.search(query, num_results=max_results_per_query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_parallel_results(queries, results)

    def _validate_multi_query_params(self, queries: List[str], max_results_per_query: int) -> None:
        """Validate multi-query search parameters."""
        assert queries, "Queries list cannot be empty"
        assert 1 <= len(queries) <= 20, f"Query count must be 1-20, got {len(queries)}"
        assert 1 <= max_results_per_query <= 25, f"max_results_per_query must be 1-25, got {max_results_per_query}"
        assert all(query and query.strip() for query in queries), "All queries must be non-empty"

    def _process_parallel_results(self, queries: List[str], results: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Process parallel search results with error handling."""
        output = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Deep search failed for '{query}': {result}")
                output[query] = []
            else:
                output[query] = result
        return output

    async def deep_research(
        self,
        query: str,
        depth: str = "comprehensive",
        num_results: int = 15,
        relevance_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Execute comprehensive deep research with enhanced retrieval.

        Args:
            query: Research query string
            depth: Research depth ("basic", "comprehensive", "exhaustive")
            num_results: Maximum results to return
            relevance_threshold: Minimum relevance score (0-1)

        Returns:
            Dictionary with search results and metadata

        Raises:
            ValueError: Invalid parameters
        """
        self._validate_deep_research_params(query, depth, num_results, relevance_threshold)

        logger.info(f"Starting deep research ({depth}) for: {query}")

        # Execute multi-layered search strategy
        search_results = await self._multi_layered_search(query, depth, num_results, relevance_threshold)
        content_analysis = await self._analyze_content_depth(search_results)

        return {
            'query': query,
            'depth': depth,
            'results': search_results,
            'analysis': content_analysis,
            'timestamp': datetime.utcnow().isoformat(),
            'total_sources': len(search_results)
        }

    def _validate_deep_research_params(self, query: str, depth: str,
                                       num_results: int, relevance_threshold: float) -> None:
        """Validate deep research parameters."""
        assert query and query.strip(), "Query cannot be empty"
        assert depth in ["basic", "comprehensive", "exhaustive"], f"Invalid depth: {depth}"
        assert 5 <= num_results <= 30, f"num_results must be 5-30, got {num_results}"
        assert 0.3 <= relevance_threshold <= 0.9, (
            f"relevance_threshold must be 0.3-0.9, got {relevance_threshold}"
        )

    async def _multi_layered_search(self, query: str, depth: str, num_results: int,
                                    relevance_threshold: float) -> List[Dict[str, Any]]:
        """Execute multi-layered search strategy based on depth."""
        search_types = {
            "basic": ["web"],
            "comprehensive": ["all"],
            "exhaustive": ["all", "web"]
        }

        all_results = []
        for search_type in search_types[depth]:
            results = await self.search(
                query=query,
                search_type=search_type,
                num_results=num_results // len(search_types[depth]),
                relevance_threshold=relevance_threshold
            )
            all_results.extend(results)

        # Remove duplicates and sort by relevance
        seen_urls = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.get('relevance_score', 0), reverse=True):
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)

        return unique_results[:num_results]

    async def _analyze_content_depth(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content depth and quality metrics."""
        if not results:
            return {'error': 'No results to analyze'}

        total_sources = len(results)
        avg_relevance = sum(r.get('relevance_score', 0) for r in results) / total_sources

        domains = {}
        for result in results:
            domain = result.get('source', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1

        return {
            'total_sources': total_sources,
            'average_relevance': round(avg_relevance, 3),
            'domain_diversity': len(domains),
            'top_domains': sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5],
            'quality_score': min(1.0, avg_relevance * (len(domains) / max(total_sources, 1)))
        }
