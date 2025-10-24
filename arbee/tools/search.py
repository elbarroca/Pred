"""
Web Search Tools for Research Agents
Provides web search capabilities using Valyu and optionally Tavily
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from arbee.api_clients.valyu import ValyuResearchClient
import logging

logger = logging.getLogger(__name__)


@tool
async def web_search_tool(
    query: str,
    max_results: int = 10,
    date_range_days: Optional[int] = 90
) -> List[Dict[str, Any]]:
    """
    Search the web for information using Valyu Research API.

    Use this tool when you need to find current information, news articles,
    research papers, or any web content related to the market question.

    Args:
        query: Search query string (be specific for better results)
        max_results: Maximum number of results to return (default 10)
        date_range_days: Only return results from last N days (default 90, None for all time)

    Returns:
        List of search results with title, URL, snippet, and published date

    Example:
        >>> results = await web_search_tool("Trump 2024 election polls Arizona", max_results=5)
        >>> print(results[0]['title'])
    """
    try:
        logger.info(f"🔍 Web search: '{query}' (max_results={max_results})")

        client = ValyuResearchClient()

        # Execute search
        results = await client.multi_query_search(
            queries=[query],
            max_results_per_query=max_results
        )

        # Extract results for this query
        search_results = results.get(query, [])

        logger.info(f"✅ Found {len(search_results)} results for '{query}'")

        # Format results consistently
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'title': result.get('title', 'N/A'),
                'url': result.get('url', ''),
                'snippet': result.get('snippet', ''),
                'published_date': result.get('published_date', ''),
                'source': result.get('source', ''),
                'content': result.get('content', result.get('snippet', ''))
            })

        return formatted_results

    except Exception as e:
        logger.error(f"❌ Web search failed for '{query}': {e}")
        return [{"error": str(e), "query": query}]

@tool
async def multi_query_search_tool(
    queries: List[str],
    max_results_per_query: int = 5,
    parallel: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute multiple search queries in parallel for comprehensive research.

    Use this when you need to research multiple angles or subclaims simultaneously.
    More efficient than calling web_search_tool multiple times.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per query
        parallel: Execute queries in parallel (faster)

    Returns:
        Dictionary mapping each query to its results

    Example:
        >>> results = await multi_query_search_tool([
        ...     "Trump Arizona polls 2024",
        ...     "Harris Arizona polls 2024",
        ...     "Arizona swing state trends"
        ... ])
        >>> print(len(results))  # 3 queries
    """
    try:
        logger.info(f"🔍 Multi-query search: {len(queries)} queries")

        client = ValyuResearchClient()

        # Execute all queries
        results = await client.multi_query_search(
            queries=queries,
            max_results_per_query=max_results_per_query
        )

        total_results = sum(len(r) for r in results.values())
        logger.info(f"✅ Multi-query search complete: {total_results} total results")

        return results

    except Exception as e:
        logger.error(f"❌ Multi-query search failed: {e}")
        return {query: [] for query in queries}
