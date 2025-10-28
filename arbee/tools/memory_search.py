"""
Memory Search Tools
Enables agents to search historical analyses and learnings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.store.base import SearchItem

from arbee.utils.memory import get_memory_manager
from config.system_constants import (
    SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
    SEARCH_SIMILAR_MARKETS_LIMIT_MAX,
    SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
    GET_BASE_RATES_LIMIT_DEFAULT,
    NAMESPACE_KNOWLEDGE_BASE,
    NAMESPACE_STRATEGIES,
)


# -----------------------------
# Internal helpers
# -----------------------------
def _clamp_limit(limit: int, default: int, max_val: int) -> int:
    """Clamp user-provided limits to safe bounds."""
    try:
        return max(1, min(int(limit), int(max_val)))
    except Exception:
        return int(default)


def _get_store():
    """Return the LangGraph store if configured, else None."""
    mm = get_memory_manager()
    return getattr(mm, "store", None)


# -----------------------------
# Public tools
# -----------------------------
@tool
async def search_similar_markets_tool(
    market_question: str,
    limit: int = SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Find past market analyses similar to the current market question.

    Args:
        market_question: Market question to match.
        limit: Maximum number of results.

    Returns:
        List of dicts with keys like: id, question, analysis, score, stored_at, and optional prior/posterior/outcome/market_url/market_id/workflow_id.
    """
    assert isinstance(market_question, str) and market_question.strip(), "market_question is required"
    limit = _clamp_limit(limit, SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT, SEARCH_SIMILAR_MARKETS_LIMIT_MAX)
    return await _search_similar_markets_in_store(query=market_question, limit=limit)


@tool
async def search_historical_evidence_tool(
    topic: str,
    evidence_type: Optional[str] = None,
    limit: int = SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Search previously stored evidence items in the knowledge base.

    Args:
        topic: Topic or subclaim to search for.
        evidence_type: Optional evidence subtype (e.g., poll, news, study).
        limit: Maximum results.

    Returns:
        List of simplified evidence dicts.
    """
    assert isinstance(topic, str) and topic.strip(), "topic is required"
    limit = max(1, int(limit))

    store = _get_store()
    if not store:
        return []

    filt: Dict[str, Any] = {"content_type": "evidence_item"}
    if evidence_type:
        filt["evidence_type"] = evidence_type

    try:
        results = await store.asearch((NAMESPACE_KNOWLEDGE_BASE,), query=topic, filter=filt, limit=limit)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for item in results:
        val = item.value or {}
        content = val.get("content")
        if isinstance(content, dict):
            out.append(
                {
                    "id": val.get("id") or item.key,
                    "title": content.get("title", "Unknown"),
                    "url": content.get("url", ""),
                    "llr": content.get("LLR", 0.0),
                    "verifiability": content.get("verifiability_score", 0.5),
                    "independence": content.get("independence_score", 0.8),
                    "recency": content.get("recency_score", 0.7),
                    "support": content.get("support", "neutral"),
                    "claim_summary": content.get("claim_summary", ""),
                    "stored_at": item.updated_at.isoformat() if getattr(item, "updated_at", None) else "",
                    "relevance_score": getattr(item, "score", None),
                }
            )
    return out


@tool
async def get_base_rates_tool(
    event_category: str,
    time_range: Optional[str] = None,
    limit: int = GET_BASE_RATES_LIMIT_DEFAULT,
) -> Dict[str, Any]:
    """
    Retrieve historical base rates for a reference class.

    Args:
        event_category: Reference class label (e.g., "incumbent wins presidential election").
        time_range: Optional time range tag to match stored base rates (e.g., "2000-2020").
        limit: Max stored items to aggregate.

    Returns:
        Dict with base_rate (0..1), sample_size, confidence, and optional sources list.
    """
    assert isinstance(event_category, str) and event_category.strip(), "event_category is required"
    limit = max(1, int(limit))

    store = _get_store()
    if store:
        filt: Dict[str, Any] = {"content_type": "base_rate"}
        if time_range:
            filt["time_range"] = time_range

        try:
            results = await store.asearch((NAMESPACE_KNOWLEDGE_BASE,), query=event_category, filter=filt, limit=limit)
        except Exception:
            results = []

        if results:
            rates: List[float] = []
            sources: List[str] = []
            for it in results:
                val = it.value or {}
                content = val.get("content")
                if isinstance(content, dict):
                    rate = content.get("base_rate")
                    if isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0:
                        rates.append(float(rate))
                        src = content.get("source", "Unknown")
                        if isinstance(src, str):
                            sources.append(src)
            if rates:
                avg = sum(rates) / len(rates)
                return {
                    "event_category": event_category,
                    "base_rate": avg,
                    "sample_size": len(rates),
                    "confidence": "moderate" if len(rates) >= 3 else "low",
                    "sources": sources[:3],
                    "note": f"Aggregated from {len(rates)} stored base-rate items",
                }

    # Fallback neutral prior when no stored base rates available
    return {
        "event_category": event_category,
        "base_rate": 0.5,
        "sample_size": 0,
        "confidence": "low",
        "note": "No stored base-rate data; returning neutral 50% prior",
    }


@tool
async def store_successful_strategy_tool(
    strategy_type: str,
    description: str,
    effectiveness: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Persist a successful research/analysis strategy for future reuse.

    Args:
        strategy_type: Strategy category (e.g., "search_strategy").
        description: Human-readable description.
        effectiveness: Effectiveness in [0.0, 1.0].
        metadata: Optional extra context.

    Returns:
        True if stored successfully, else False.
    """
    assert strategy_type and description, "strategy_type and description are required"
    try:
        eff = float(effectiveness)
    except Exception:
        eff = 0.0
    eff = max(0.0, min(1.0, eff))

    store = _get_store()
    if not store:
        return False

    import hashlib
    import time

    key_src = f"{strategy_type}_{description[:40]}_{time.time()}"
    key = f"strategy_{hashlib.md5(key_src.encode()).hexdigest()[:12]}"

    data = {
        "content_type": "strategy",
        "strategy_type": strategy_type,
        "description": description,
        "effectiveness": eff,
        "stored_at": "current_session",
        "metadata": metadata or {},
    }

    try:
        await store.aput(NAMESPACE_STRATEGIES, key, data)
        return True
    except Exception:
        return False


# -----------------------------
# Store-backed search
# -----------------------------
async def _search_similar_markets_in_store(query: str, limit: int) -> List[Dict[str, Any]]:
    """Semantic search for similar market analyses in the knowledge base."""
    store = _get_store()
    if not store:
        return []
    try:
        results = await store.asearch(
            (NAMESPACE_KNOWLEDGE_BASE,),
            query=query,
            filter={"content_type": "market_analysis"},
            limit=limit,
        )
    except Exception:
        return []
    parsed = [res for item in results if (res := _parse_market_search_item(item)) is not None]
    return parsed[:limit]


def _parse_market_search_item(item: SearchItem) -> Optional[Dict[str, Any]]:
    """Normalize a SearchItem into a lightweight dict."""
    val = item.value or {}
    content = val.get("content")
    if isinstance(content, dict):
        cdict = content.copy()
    elif isinstance(content, str):
        cdict = {"analysis": content}
    else:
        cdict = {}

    question = cdict.get("market_question") or cdict.get("question") or val.get("market_question") or val.get("question")
    if not question:
        return None

    out: Dict[str, Any] = {
        "id": val.get("id") or item.key,
        "question": question,
        "analysis": cdict or val,
        "score": getattr(item, "score", None),
        "stored_at": item.updated_at.isoformat() if getattr(item, "updated_at", None) else "",
    }

    for f in ("prior", "posterior", "outcome", "market_url", "market_id", "workflow_id"):
        if f in cdict:
            out[f] = cdict[f]
        elif f in val:
            out[f] = val[f]

    meta: Dict[str, Any] = {}
    for m in (val.get("metadata"), cdict.get("metadata")):
        if isinstance(m, dict):
            meta.update(m)
    if meta:
        out["metadata"] = meta

    return out


# -----------------------------
# Integrity probe
# -----------------------------
def integrity_report() -> Dict[str, Any]:
    """Compact integrity and coverage snapshot for this module."""
    store = _get_store()
    return {
        "store_available": bool(store),
        "limits": {
            "similar_markets_default": SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
            "similar_markets_max": SEARCH_SIMILAR_MARKETS_LIMIT_MAX,
            "historical_evidence_default": SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
            "base_rates_default": GET_BASE_RATES_LIMIT_DEFAULT,
        },
        "violations": [] if store else ["store_unavailable"],
    }