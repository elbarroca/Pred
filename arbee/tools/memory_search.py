"""
Memory Search Tools
Enables agents to search historical analyses and learnings
"""
import logging
import os
from typing import List, Dict, Any, Optional

import httpx
from langchain_core.tools import tool
from langgraph.store.base import SearchItem

from arbee.utils.memory import get_memory_manager

logger = logging.getLogger(__name__)


@tool
async def search_similar_markets_tool(
    market_question: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar market questions analyzed in the past.

    Use this to find analogous cases that can inform your current analysis.
    Helps with setting priors, identifying relevant evidence types, and
    learning from successful strategies.

    Args:
        market_question: Current market question to find similar cases for
        limit: Maximum number of similar markets to return

    Returns:
        List of similar market analyses with question, outcome, prior, posterior, etc.

    Example:
        >>> similar = await search_similar_markets_tool("Will Trump win 2024 election?")
        >>> for market in similar:
        ...     print(f"{market['question']}: prior={market['prior']}, outcome={market['outcome']}")
    """
    try:
        logger.info(f"🔍 Searching for similar markets to: '{market_question[:60]}'")

        limit = max(1, min(limit, 20))

        store_results = await _search_similar_markets_in_store(
            query=market_question,
            limit=limit
        )
        if store_results:
            return store_results

        weaviate_results = await _search_similar_markets_in_weaviate(
            query=market_question,
            limit=limit
        )
        if weaviate_results:
            return weaviate_results

        logger.info("No similar markets found via LangGraph store or Weaviate")
        return []

    except Exception as e:
        logger.error(f"❌ Similar markets search failed: {e}")
        return []


@tool
async def search_historical_evidence_tool(
    topic: str,
    evidence_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search historical evidence database for relevant past findings.

    Use this to check if evidence about this topic has been gathered before,
    avoiding redundant research and building on previous work.

    Args:
        topic: Topic or subclaim to search for
        evidence_type: Optional filter by evidence type (poll, news, study, etc.)
        limit: Maximum results

    Returns:
        List of historical evidence items

    Example:
        >>> evidence = await search_historical_evidence_tool("Arizona swing state polls")
        >>> print(f"Found {len(evidence)} relevant historical evidence items")
    """
    try:
        logger.info(f"🔍 Searching historical evidence: '{topic[:60]}'")

        # TODO: Implement vector/keyword search in evidence database
        # Placeholder for now

        logger.warning("⚠️  Historical evidence search not yet implemented")

        return []

    except Exception as e:
        logger.error(f"❌ Historical evidence search failed: {e}")
        return []


@tool
async def get_base_rates_tool(
    event_category: str,
    time_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve historical base rates for event category.

    Use this to inform prior probability selection with reference class data.

    Args:
        event_category: Category of event (e.g., "US presidential elections", "tech IPOs")
        time_range: Optional time range (e.g., "2000-2020")

    Returns:
        Dict with base_rate, sample_size, confidence, and examples

    Example:
        >>> rates = await get_base_rates_tool("incumbent party wins presidential election")
        >>> print(f"Base rate: {rates['base_rate']:.1%}")
    """
    try:
        logger.info(f"📊 Looking up base rates for: '{event_category}'")

        # TODO: Implement base rate lookup from knowledge base
        # Could use Wikipedia, historical databases, etc.

        logger.warning("⚠️  Base rates lookup not yet implemented")

        return {
            'event_category': event_category,
            'base_rate': 0.5,  # Default
            'sample_size': 0,
            'confidence': 'low',
            'note': 'Base rates lookup not implemented yet'
        }

    except Exception as e:
        logger.error(f"❌ Base rates lookup failed: {e}")
        return {'error': str(e), 'base_rate': 0.5}


@tool
async def store_successful_strategy_tool(
    strategy_type: str,
    description: str,
    effectiveness: float,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Store a successful research or analysis strategy for future reference.

    Use this to record what worked well so future agents can learn from it.

    Args:
        strategy_type: Type of strategy (e.g., "search_strategy", "evidence_filtering")
        description: Description of the strategy
        effectiveness: How effective it was (0.0 to 1.0)
        metadata: Optional additional context

    Returns:
        True if stored successfully

    Example:
        >>> await store_successful_strategy_tool(
        ...     "search_strategy",
        ...     "Combining '538 poll' with state name gives high-quality polling data",
        ...     effectiveness=0.9
        ... )
    """
    try:
        logger.info(f"💾 Storing successful strategy: {strategy_type}")

        # TODO: Store in LangGraph Store for cross-session learning

        logger.warning("⚠️  Strategy storage not yet implemented")

        return False

    except Exception as e:
        logger.error(f"❌ Strategy storage failed: {e}")
        return False


async def _search_similar_markets_in_store(
    query: str,
    limit: int
) -> List[Dict[str, Any]]:
    """
    Search the LangGraph store for similar market analyses using semantic search.

    Args:
        query: Market question to search for
        limit: Maximum number of results to return

    Returns:
        List of similar market analyses dictionaries
    """
    memory_manager = get_memory_manager()
    store = getattr(memory_manager, "store", None)

    if not store:
        logger.debug("No LangGraph store configured; skipping memory search lookup")
        return []

    try:
        # Primary namespace for long-term knowledge
        search_results = await store.asearch(
            ("knowledge_base",),
            query=query,
            filter={"content_type": "market_analysis"},
            limit=limit
        )
    except Exception as err:
        logger.warning(f"LangGraph store search failed: {err}")
        return []

    parsed = [
        result
        for item in search_results
        if (result := _parse_market_search_item(item)) is not None
    ]

    return parsed[:limit]


async def _search_similar_markets_in_weaviate(
    query: str,
    limit: int
) -> List[Dict[str, Any]]:
    """
    Query a Weaviate instance for similar market analyses using hybrid search.
    Falls back silently if Weaviate is not configured.
    """
    endpoint = os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_ENDPOINT")
    if not endpoint:
        return []

    class_name = os.getenv("WEAVIATE_MARKETS_CLASS", "MarketAnalysisMemory")
    api_key = os.getenv("WEAVIATE_API_KEY") or os.getenv("WCS_API_KEY")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "X-OpenAI-Api-Key" not in headers:
        headers["X-OpenAI-Api-Key"] = openai_key

    graphql = (
        "query GetSimilarMarkets($query: String!, $limit: Int!) {{"
        "  Get {{"
        "    {class_name}("
        "      limit: $limit"
        "      hybrid: {{ query: $query, alpha: 0.35 }}"
        "    ) {{"
        "      id"
        "      question"
        "      market_question"
        "      prior"
        "      posterior"
        "      outcome"
        "      summary"
        "      market_url"
        "      metadata"
        "      _additional {{ score }}"
        "    }}"
        "  }}"
        "}}"
    ).format(class_name=class_name)

    payload = {"query": graphql, "variables": {"query": query, "limit": limit}}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1/graphql",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
    except Exception as exc:
        logger.debug(f"Weaviate search failed: {exc}")
        return []

    try:
        data = response.json()
    except ValueError as exc:
        logger.debug(f"Failed to decode Weaviate response: {exc}")
        return []

    hits = data.get("data", {}).get("Get", {}).get(class_name, []) or []
    results: List[Dict[str, Any]] = []

    for node in hits:
        question = node.get("market_question") or node.get("question")
        if not question:
            continue

        item: Dict[str, Any] = {
            "id": node.get("id"),
            "question": question,
            "prior": node.get("prior"),
            "posterior": node.get("posterior"),
            "outcome": node.get("outcome"),
            "summary": node.get("summary"),
            "market_url": node.get("market_url"),
        }

        metadata = node.get("metadata")
        if isinstance(metadata, dict) and metadata:
            item["metadata"] = metadata

        additional = node.get("_additional") or {}
        score = additional.get("score")
        if score is not None:
            item["score"] = score

        results.append(item)

    return results[:limit]


def _parse_market_search_item(item: SearchItem) -> Optional[Dict[str, Any]]:
    """
    Normalize LangGraph search items into the tool output structure.

    Args:
        item: SearchItem returned by LangGraph store

    Returns:
        Parsed dictionary or None if the data is incomplete
    """
    value = item.value or {}
    content = value.get("content")

    if isinstance(content, dict):
        content_dict = content.copy()
    elif isinstance(content, str):
        content_dict = {"analysis": content}
    else:
        content_dict = {}

    question = (
        content_dict.get("market_question")
        or content_dict.get("question")
        or value.get("market_question")
        or value.get("question")
    )

    if not question:
        return None

    metadata = {}
    value_metadata = value.get("metadata")
    if isinstance(value_metadata, dict):
        metadata.update(value_metadata)
    content_metadata = content_dict.get("metadata")
    if isinstance(content_metadata, dict):
        metadata.update(content_metadata)

    result: Dict[str, Any] = {
        "id": value.get("id") or item.key,
        "question": question,
        "analysis": content_dict or value,
        "score": item.score,
        "stored_at": item.updated_at.isoformat(),
    }

    for field in ("prior", "posterior", "outcome", "market_url", "market_id", "workflow_id"):
        if field in content_dict:
            result[field] = content_dict[field]
        elif field in value:
            result[field] = value[field]

    if metadata:
        result["metadata"] = metadata

    return result
