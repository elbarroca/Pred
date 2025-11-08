"""
Generic Market Pattern Detection Utilities
Provides agnostic pattern detection for any market type.
"""
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


def extract_event_category(market_question: str) -> str:
    """
    Extract event category from market question (generic, not domain-specific).
    
    Args:
        market_question: Market question text
        
    Returns:
        Event category string (e.g., "election outcome", "sports event", etc.)
    """
    question_lower = market_question.lower()
    
    # Generic pattern matching (not domain-specific)
    patterns = {
        "election": ["election", "president", "vote", "ballot", "candidate", "campaign"],
        "sports": ["sports", "race", "game", "match", "championship", "tournament", "athlete"],
        "cryptocurrency": ["crypto", "bitcoin", "ethereum", "blockchain", "token", "coin"],
        "technology": ["tech", "ai", "software", "hardware", "product launch", "release"],
        "finance": ["stock", "market", "price", "earnings", "revenue", "profit"],
        "entertainment": ["movie", "film", "show", "celebrity", "award", "oscar"],
        "politics": ["policy", "law", "bill", "congress", "senate", "legislation"],
        "health": ["health", "medical", "disease", "treatment", "vaccine", "drug"],
    }
    
    for category, keywords in patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            return f"{category} event"
    
    # Fallback: use first few words of question
    words = market_question.split()[:5]
    return " ".join(words).lower()[:50]


def find_similar_market_patterns(
    market_question: str,
    historical_markets: List[Dict[str, Any]],
    similarity_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Find similar markets based on question patterns (generic matching).
    
    Args:
        market_question: Current market question
        historical_markets: List of historical market dicts with 'question' field
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of similar markets with similarity scores
    """
    question_words = set(re.findall(r'\w+', market_question.lower()))
    similar = []
    
    for market in historical_markets:
        hist_question = market.get("question", "")
        hist_words = set(re.findall(r'\w+', hist_question.lower()))
        
        if not question_words or not hist_words:
            continue
        
        # Simple Jaccard similarity
        intersection = len(question_words & hist_words)
        union = len(question_words | hist_words)
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity >= similarity_threshold:
            similar.append({
                **market,
                "similarity_score": similarity,
            })
    
    # Sort by similarity descending
    similar.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    return similar


def extract_key_variables(market_question: str) -> List[str]:
    """
    Extract key variables/factors from market question (generic extraction).
    
    Args:
        market_question: Market question text
        
    Returns:
        List of key variable strings
    """
    # Simple extraction: look for capitalized words, numbers, dates
    variables = []
    
    # Extract capitalized phrases (likely proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', market_question)
    variables.extend(capitalized)
    
    # Extract numbers (dates, amounts, etc.)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', market_question)
    variables.extend(numbers)
    
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', market_question)
    variables.extend(quoted)
    
    # Remove duplicates and return
    return list(set(variables))[:10]  # Limit to top 10


def categorize_market_type(market_question: str) -> Dict[str, Any]:
    """
    Categorize market type generically (not domain-specific).
    
    Args:
        market_question: Market question text
        
    Returns:
        Dict with category, subcategory, and confidence
    """
    category = extract_event_category(market_question)
    
    # Determine if binary or multi-outcome
    question_lower = market_question.lower()
    is_binary = any(word in question_lower for word in ["will", "does", "is", "has", "can"])
    
    # Determine timeframe
    timeframe = "unknown"
    if any(word in question_lower for word in ["2025", "2026", "this year", "next year"]):
        timeframe = "near_term"
    elif any(word in question_lower for word in ["by", "before", "after", "until"]):
        timeframe = "time_bound"
    else:
        timeframe = "open_ended"
    
    return {
        "category": category,
        "is_binary": is_binary,
        "timeframe": timeframe,
        "confidence": 0.7,  # Generic confidence
    }

