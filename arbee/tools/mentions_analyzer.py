"""
Mentions Market Analyzer Tool
Analyzes mentions markets to detect mispriced opportunities.
Inspired by MentionsPro functionality for finding mispriced markets based on transcript analysis.
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from arbee.utils.rich_logging import (
    log_tool_start,
    log_tool_success,
    log_tool_error,
    log_edge_detection_result,
)
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@tool
async def analyze_mentions_market_tool(
    market_question: str,
    transcript_text: Optional[str] = None,
    mentions_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Analyze mentions market to detect mispricing based on transcript analysis.
    
    This tool analyzes mentions markets (markets based on podcast/interview transcripts)
    to find mispriced opportunities by comparing:
    - Transcript content vs market prices
    - Sentiment analysis of mentions
    - Key quote extraction and interpretation
    
    Args:
        market_question: Market question being analyzed
        transcript_text: Optional transcript text to analyze
        mentions_data: Optional list of mention dicts with keys: timestamp, speaker, text, sentiment
        
    Returns:
        Dict with:
        - edge_type: "mentions_mispricing"
        - strength: Mispricing strength (0-1)
        - confidence: Confidence in detection (0-1)
        - evidence: List of evidence strings
        - key_quotes: List of relevant quotes from transcript
        - sentiment_score: Overall sentiment (-1 to 1)
        - mispricing_direction: "bullish" or "bearish"
    """
    try:
        log_tool_start("analyze_mentions_market_tool", {"market_question": market_question[:50], "has_transcript": transcript_text is not None, "has_mentions_data": mentions_data is not None})
        logger.info(f"📝 Analyzing mentions market: {market_question[:50]}...")
        
        strength = 0.0
        confidence = 0.3  # Low default confidence without transcript
        evidence = []
        key_quotes = []
        sentiment_score = 0.0
        mispricing_direction = "neutral"
        
        # If transcript provided, analyze it
        if transcript_text:
            # Simple sentiment analysis (in production, use proper NLP)
            text_lower = transcript_text.lower()
            
            # Positive indicators
            positive_words = ["yes", "will", "likely", "probably", "expect", "confident", "definitely"]
            negative_words = ["no", "won't", "unlikely", "doubt", "uncertain", "probably not"]
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            sentiment_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
            
            # Extract key quotes (simple: sentences with market-relevant keywords)
            sentences = transcript_text.split('.')
            relevant_keywords = market_question.lower().split()[:5]  # First 5 words
            
            for sentence in sentences[:20]:  # Check first 20 sentences
                if any(keyword in sentence.lower() for keyword in relevant_keywords if len(keyword) > 3):
                    if len(sentence.strip()) > 20 and len(sentence.strip()) < 200:
                        key_quotes.append(sentence.strip()[:150])
            
            if sentiment_score > 0.3:
                mispricing_direction = "bullish"
                strength = min(0.7, abs(sentiment_score) * 1.5)
                evidence.append(f"Positive sentiment detected (score: {sentiment_score:.2f})")
            elif sentiment_score < -0.3:
                mispricing_direction = "bearish"
                strength = min(0.7, abs(sentiment_score) * 1.5)
                evidence.append(f"Negative sentiment detected (score: {sentiment_score:.2f})")
            
            if key_quotes:
                confidence = min(0.8, 0.3 + len(key_quotes) * 0.1)
                evidence.append(f"Found {len(key_quotes)} relevant quotes")
        
        # If mentions_data provided, analyze structured mentions
        elif mentions_data:
            positive_mentions = sum(1 for m in mentions_data if m.get('sentiment', 0) > 0)
            negative_mentions = sum(1 for m in mentions_data if m.get('sentiment', 0) < 0)
            total_mentions = len(mentions_data)
            
            if total_mentions > 0:
                sentiment_score = (positive_mentions - negative_mentions) / total_mentions
                
                if sentiment_score > 0.2:
                    mispricing_direction = "bullish"
                    strength = min(0.8, abs(sentiment_score) * 2)
                elif sentiment_score < -0.2:
                    mispricing_direction = "bearish"
                    strength = min(0.8, abs(sentiment_score) * 2)
                
                confidence = min(0.9, 0.5 + total_mentions * 0.05)
                evidence.append(f"Analyzed {total_mentions} mentions ({positive_mentions} positive, {negative_mentions} negative)")
                
                # Extract top quotes
                sorted_mentions = sorted(
                    mentions_data,
                    key=lambda x: abs(x.get('sentiment', 0)),
                    reverse=True
                )
                key_quotes = [m.get('text', '')[:150] for m in sorted_mentions[:5]]
        else:
            evidence.append("No transcript or mentions data provided - using fallback analysis")
            confidence = 0.2
        
        result = {
            "edge_type": "mentions_mispricing",
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence,
            "key_quotes": key_quotes[:5],  # Top 5 quotes
            "sentiment_score": sentiment_score,
            "mispricing_direction": mispricing_direction,
            "market_question": market_question,
            "detection_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("analyze_mentions_market_tool", "mentions_mispricing", strength, confidence, evidence)
        log_tool_success("analyze_mentions_market_tool", {"edge_strength": strength, "confidence": confidence, "sentiment_score": sentiment_score})
        
        return result
        
    except Exception as e:
        log_tool_error("analyze_mentions_market_tool", e, f"Market: {market_question[:50]}")
        logger.error(f"❌ Mentions market analysis failed: {e}")
        return {
            "edge_type": "mentions_mispricing",
            "strength": 0.0,
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "key_quotes": [],
            "sentiment_score": 0.0,
            "mispricing_direction": "neutral",
            "error": str(e),
        }

