"""
Market Matching and Comparison Utilities
Finds similar markets across different prediction platforms for arbitrage analysis
"""
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient
from arbee.api_clients.valyu import ValyuResearchClient


@dataclass
class MarketMatch:
    """Represents a matched pair of markets from different platforms"""
    polymarket_market: Dict[str, Any]
    kalshi_market: Dict[str, Any]
    similarity_score: float
    match_type: str  # "exact", "semantic", "category", "fuzzy"
    key_differences: Dict[str, Any]
    arbitrage_opportunity: Optional[Dict[str, float]] = None


@dataclass
class ArbitrageOpportunity:
    """Represents a potential arbitrage opportunity"""
    poly_price: float
    kalshi_price: float
    price_diff: float
    edge: float
    confidence: float
    recommended_action: str
    risk_factors: List[str]


class MarketMatcher:
    """Matches markets across prediction platforms for arbitrage analysis"""

    def __init__(self):
        self.poly_client = PolymarketClient()
        self.kalshi_client = KalshiClient()
        self.valyu_client = ValyuResearchClient()

    async def find_matching_markets(
        self,
        poly_limit: int = 50,
        kalshi_limit: int = 50,
        min_similarity: float = 0.6
    ) -> List[MarketMatch]:
        """
        Find markets that match between Polymarket and Kalshi

        Args:
            poly_limit: Max Polymarket markets to fetch
            kalshi_limit: Max Kalshi markets to fetch
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of matched market pairs
        """
        print("🔍 Fetching markets from both platforms...")

        # Fetch markets from both platforms
        poly_markets, kalshi_markets = await asyncio.gather(
            self.poly_client.get_event(limit=poly_limit, active=True),
            self.kalshi_client.get_event(limit=kalshi_limit)
        )

        print(f"📊 Retrieved {len(poly_markets)} Polymarket and {len(kalshi_markets)} Kalshi markets")

        # Normalize market data
        normalized_poly = self._normalize_polymarket_data(poly_markets)
        normalized_kalshi = self._normalize_kalshi_data(kalshi_markets)

        # Debug: Show sample normalized data
        if normalized_poly:
            print(f"📋 Sample Polymarket: {normalized_poly[0]['question']} (tags: {normalized_poly[0]['tags']})")
        if normalized_kalshi:
            print(f"📋 Sample Kalshi: {normalized_kalshi[0]['question']} (tags: {normalized_kalshi[0]['tags']})")

        # Find matches
        matches = []
        max_similarity = 0
        for poly_market in normalized_poly:
            for kalshi_market in normalized_kalshi:
                similarity = self._calculate_similarity(poly_market, kalshi_market)
                max_similarity = max(max_similarity, similarity)

                if similarity >= min_similarity:
                    match = MarketMatch(
                        polymarket_market=poly_market,
                        kalshi_market=kalshi_market,
                        similarity_score=similarity,
                        match_type=self._determine_match_type(poly_market, kalshi_market, similarity),
                        key_differences=self._find_differences(poly_market, kalshi_market)
                    )
                    matches.append(match)

        print(f"🎯 Max similarity found: {max_similarity:.3f}")
        print(f"🎯 Found {len(matches)} matching market pairs (threshold: {min_similarity})")

        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)

        return matches

    def _normalize_polymarket_data(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize Polymarket data for comparison"""
        normalized = []

        for market in markets:
            # Fix JSON parsing issues
            if isinstance(market.get('outcomes'), str):
                import json
                try:
                    market['outcomes'] = json.loads(market['outcomes'])
                except:
                    market['outcomes'] = []

            if isinstance(market.get('outcomePrices'), str):
                import json
                try:
                    market['outcomePrices'] = json.loads(market['outcomePrices'])
                    # Convert to floats
                    market['outcomePrices'] = [float(p) for p in market['outcomePrices']]
                except:
                    market['outcomePrices'] = []

            # Convert liquidity to float if string
            if isinstance(market.get('liquidity'), str):
                try:
                    market['liquidity'] = float(market['liquidity'])
                except:
                    market['liquidity'] = 0.0

            # Extract key fields for comparison
            normalized_market = {
                'id': market.get('id', ''),
                'question': market.get('question', '').lower(),
                'category': market.get('category', '').lower(),
                'outcomes': [o.lower() for o in market.get('outcomes', [])],
                'end_date': market.get('endDate', ''),
                'liquidity': market.get('liquidityNum', 0),
                'volume': market.get('volumeNum', 0),
                'active': market.get('active', False),
                'description': market.get('description', '').lower(),
                'tags': self._extract_keywords(market.get('question', '') + ' ' + market.get('description', ''))
            }
            normalized.append(normalized_market)

        return normalized

    def _normalize_kalshi_data(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize Kalshi data for comparison"""
        normalized = []

        for market in markets:
            # Extract key fields for comparison
            normalized_market = {
                'id': market.get('ticker', ''),
                'question': market.get('title', '').lower(),
                'category': market.get('category', '').lower(),
                'outcomes': [
                    market.get('yes_sub_title', '').lower(),
                    market.get('no_sub_title', '').lower()
                ],
                'end_date': market.get('expiration_time', ''),
                'liquidity': market.get('liquidity', 0),
                'volume': market.get('volume', 0),
                'active': market.get('status') == 'active',
                'description': market.get('subtitle', '').lower(),
                'tags': self._extract_keywords(market.get('title', '') + ' ' + market.get('subtitle', ''))
            }
            normalized.append(normalized_market)

        return normalized

    def _calculate_similarity(self, market1: Dict[str, Any], market2: Dict[str, Any]) -> float:
        """Calculate similarity score between two markets (0-1)"""
        score = 0.0
        factors = 0

        # Question similarity (most important)
        question1 = market1.get('question', '')
        question2 = market2.get('question', '')
        question_sim = self._text_similarity(question1, question2)
        score += question_sim * 0.4
        factors += 0.4

        # Category similarity
        cat1 = market1.get('category', '')
        cat2 = market2.get('category', '')
        cat_sim = 1.0 if cat1 == cat2 else 0.0
        score += cat_sim * 0.2
        factors += 0.2

        # Outcome similarity
        outcomes1 = set(market1.get('outcomes', []))
        outcomes2 = set(market2.get('outcomes', []))
        if outcomes1 and outcomes2:
            outcome_overlap = len(outcomes1.intersection(outcomes2)) / max(len(outcomes1.union(outcomes2)), 1)
            score += outcome_overlap * 0.2
            factors += 0.2

        # Keyword similarity
        tags1 = set(market1.get('tags', []))
        tags2 = set(market2.get('tags', []))
        if tags1 and tags2:
            tag_overlap = len(tags1.intersection(tags2)) / max(len(tags1.union(tags2)), 1)
            score += tag_overlap * 0.15
            factors += 0.15

        # Time proximity (bonus for similar end dates) - reduced weight since markets may be from different periods
        time_sim = self._time_similarity(market1.get('end_date', ''), market2.get('end_date', ''))
        score += time_sim * 0.02  # Reduced from 0.05 to 0.02
        factors += 0.02

        return min(score / factors if factors > 0 else 0, 1.0)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        if not text1 or not text2:
            return 0.0

        # Simple word overlap
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0

    def _time_similarity(self, date1: str, date2: str) -> float:
        """Calculate similarity based on time proximity"""
        if not date1 or not date2:
            return 0.5  # Neutral score if dates missing

        try:
            dt1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))

            # Days difference
            diff_days = abs((dt1 - dt2).days)

            # Exponential decay: very similar if < 7 days, drops off after that
            return max(0, 1.0 - (diff_days / 30.0))
        except:
            return 0.5  # Neutral score if parsing fails

    def _determine_match_type(self, market1: Dict[str, Any], market2: Dict[str, Any], similarity: float) -> str:
        """Determine the type of match based on similarity and characteristics"""
        question1 = market1.get('question', '')
        question2 = market2.get('question', '')

        # Exact match
        if question1 == question2:
            return "exact"

        # Very high similarity with same category
        if similarity > 0.8 and market1.get('category') == market2.get('category'):
            return "semantic"

        # Same category with good keyword overlap
        if market1.get('category') == market2.get('category') and similarity > 0.6:
            return "category"

        # Lower similarity but some overlap
        return "fuzzy"

    def _find_differences(self, market1: Dict[str, Any], market2: Dict[str, Any]) -> Dict[str, Any]:
        """Find key differences between matched markets"""
        differences = {}

        # Liquidity difference
        liq1 = market1.get('liquidity', 0)
        liq2 = market2.get('liquidity', 0)
        if liq1 and liq2:
            differences['liquidity_ratio'] = max(liq1, liq2) / min(liq1, liq2) if min(liq1, liq2) > 0 else float('inf')

        # Volume difference
        vol1 = market1.get('volume', 0)
        vol2 = market2.get('volume', 0)
        if vol1 and vol2:
            differences['volume_ratio'] = max(vol1, vol2) / min(vol1, vol2) if min(vol1, vol2) > 0 else float('inf')

        # Time difference
        time_diff = self._time_similarity(market1.get('end_date', ''), market2.get('end_date', ''))
        differences['time_similarity'] = time_diff

        return differences

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []

        # Remove common stop words and extract meaningful terms
        stop_words = {
            'will', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'what', 'when', 'where', 'how', 'why', 'who', 'which', 'yes', 'no', 'not', 'before', 'after',
            'during', 'between', 'among', 'through', 'over', 'under', 'above', 'below', 'around', 'behind'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Also extract important named entities and numbers
        important_terms = []
        for word in words:
            # Keep numbers (years, dates, etc.)
            if word.isdigit() and len(word) == 4:  # Likely a year
                important_terms.append(word)
            # Keep capitalized words (proper nouns)
            elif word.istitle() and len(word) > 3:
                important_terms.append(word.lower())

        keywords.extend(important_terms)
        return list(set(keywords))[:15]  # Remove duplicates and limit to top 15

    async def analyze_arbitrage_opportunities(self, matches: List[MarketMatch]) -> List[ArbitrageOpportunity]:
        """Analyze arbitrage opportunities in matched markets"""
        opportunities = []

        for match in matches:
            try:
                # Get prices from both platforms
                poly_price = self._get_polymarket_price(match.polymarket_market)
                kalshi_price = await self._get_kalshi_price(match.kalshi_market)

                if poly_price is None or kalshi_price is None:
                    continue

                # Calculate arbitrage metrics
                price_diff = poly_price - kalshi_price
                edge = abs(price_diff)

                if edge > 0.02:  # 2% minimum edge
                    opportunity = ArbitrageOpportunity(
                        poly_price=poly_price,
                        kalshi_price=kalshi_price,
                        price_diff=price_diff,
                        edge=edge,
                        confidence=match.similarity_score,
                        recommended_action="BUY" if price_diff > 0 else "SELL",
                        risk_factors=self._assess_risks(match)
                    )
                    opportunities.append(opportunity)

            except Exception as e:
                print(f"Error analyzing arbitrage for match {match.polymarket_market.get('id', 'unknown')}: {e}")
                continue

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge, reverse=True)

        return opportunities

    def _get_polymarket_price(self, market: Dict[str, Any]) -> Optional[float]:
        """Extract price from Polymarket data"""
        prices = market.get('outcomePrices', [])
        if prices and len(prices) >= 2:
            # Use the higher probability outcome as the "main" price
            return max(prices)
        return None

    async def _get_kalshi_price(self, market: Dict[str, Any]) -> Optional[float]:
        """Extract price from Kalshi data"""
        # Use the get_market_price method which handles the price calculation
        return await self.kalshi_client.get_market_price(market['id'])

    def _assess_risks(self, match: MarketMatch) -> List[str]:
        """Assess risks for a potential arbitrage trade"""
        risks = []

        # Liquidity risk
        poly_liq = match.polymarket_market.get('liquidity', 0)
        kalshi_liq = match.kalshi_market.get('liquidity', 0)

        if poly_liq < 1000 or kalshi_liq < 1000:
            risks.append("Low liquidity")

        # Time risk
        time_sim = match.key_differences.get('time_similarity', 1.0)
        if time_sim < 0.8:
            risks.append("Different expiration dates")

        # Platform risk
        if match.similarity_score < 0.7:
            risks.append("Low similarity between markets")

        return risks
