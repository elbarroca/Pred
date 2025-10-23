"""
Market Selection and Prioritization System
Intelligently scores and filters markets for analysis based on:
- Liquidity (can we actually trade?)
- Price differences (arbitrage potential)
- Time urgency (closing soon?)
- Category relevance (domains we understand)
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketSelector:
    """
    Intelligent market selection and prioritization

    Scoring Formula:
    total_score = (liquidity_score * 0.4) +
                  (price_diff_score * 0.3) +
                  (time_urgency_score * 0.2) +
                  (category_match_score * 0.1)
    """

    def __init__(
        self,
        min_liquidity: float = 10000.0,
        min_price_diff: float = 0.05,
        max_days_until_close: int = 90,
        preferred_categories: List[str] = None
    ):
        """
        Initialize market selector

        Args:
            min_liquidity: Minimum liquidity threshold (USD)
            min_price_diff: Minimum price difference for cross-platform (0.05 = 5%)
            max_days_until_close: Only consider markets closing within N days
            preferred_categories: List of category keywords (e.g., ["politics", "sports"])
        """
        self.min_liquidity = min_liquidity
        self.min_price_diff = min_price_diff
        self.max_days_until_close = max_days_until_close
        self.preferred_categories = preferred_categories or [
            "politics", "election", "president", "sports", "nba", "nfl"
        ]

    def score_market(
        self,
        market: Dict[str, Any],
        cross_platform_prices: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate priority score for a market

        Args:
            market: Market metadata dict
            cross_platform_prices: Dict of {provider: price} if available

        Returns:
            Score from 0.0 to 1.0 (higher = higher priority)
        """
        scores = {
            'liquidity': self._score_liquidity(market),
            'price_diff': self._score_price_difference(cross_platform_prices),
            'time_urgency': self._score_time_urgency(market),
            'category': self._score_category_match(market)
        }

        # Weighted combination
        total_score = (
            scores['liquidity'] * 0.4 +
            scores['price_diff'] * 0.3 +
            scores['time_urgency'] * 0.2 +
            scores['category'] * 0.1
        )

        return total_score

    def _score_liquidity(self, market: Dict[str, Any]) -> float:
        """
        Score based on liquidity/volume

        Higher liquidity = higher score
        Uses logarithmic scaling:
        - $10k = 0.5
        - $100k = 0.75
        - $1M+ = 1.0
        """
        liquidity = market.get('liquidity', market.get('volume', 0))

        if liquidity < self.min_liquidity:
            return 0.0

        # Logarithmic scaling
        if liquidity >= 1_000_000:
            return 1.0
        elif liquidity >= 100_000:
            return 0.75
        elif liquidity >= 50_000:
            return 0.6
        else:
            # Scale from min_liquidity to $50k
            normalized = (liquidity - self.min_liquidity) / (50_000 - self.min_liquidity)
            return 0.3 + (normalized * 0.3)  # Scale from 0.3 to 0.6

    def _score_price_difference(
        self,
        cross_platform_prices: Optional[Dict[str, float]]
    ) -> float:
        """
        Score based on price difference between platforms

        Larger difference = higher arbitrage potential

        Args:
            cross_platform_prices: {provider: price}

        Returns:
            Score 0.0-1.0
        """
        if not cross_platform_prices or len(cross_platform_prices) < 2:
            return 0.5  # Neutral score if no comparison available

        prices = list(cross_platform_prices.values())
        max_price = max(prices)
        min_price = min(prices)

        price_diff = max_price - min_price

        # Score based on magnitude of difference
        if price_diff < self.min_price_diff:
            return 0.0  # Below threshold
        elif price_diff >= 0.20:  # 20%+ difference
            return 1.0
        elif price_diff >= 0.15:
            return 0.8
        elif price_diff >= 0.10:
            return 0.6
        else:
            # Scale from min_price_diff to 10%
            normalized = (price_diff - self.min_price_diff) / (0.10 - self.min_price_diff)
            return normalized * 0.6  # Scale from 0 to 0.6

    def _score_time_urgency(self, market: Dict[str, Any]) -> float:
        """
        Score based on time until market closes

        Markets closing soon get higher priority (more time-sensitive)

        Returns:
            Score 0.0-1.0
        """
        # Try multiple date field names
        end_date_str = market.get('end_date_iso') or market.get('close_time') or market.get('end_date')

        if not end_date_str:
            return 0.5  # Neutral if unknown

        try:
            # Parse date (handle multiple formats)
            if isinstance(end_date_str, str):
                if 'T' in end_date_str:
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                else:
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            else:
                return 0.5  # Can't parse

            now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
            days_until_close = (end_date - now).days

            # Filter out markets past our max window
            if days_until_close > self.max_days_until_close:
                return 0.2  # Low priority

            if days_until_close < 0:
                return 0.0  # Already closed

            # Urgency scoring (closer = higher)
            if days_until_close <= 7:
                return 1.0  # Very urgent
            elif days_until_close <= 14:
                return 0.8
            elif days_until_close <= 30:
                return 0.6
            else:
                # Scale from 30 to max_days
                normalized = 1.0 - ((days_until_close - 30) / (self.max_days_until_close - 30))
                return 0.3 + (normalized * 0.3)  # Scale from 0.3 to 0.6

        except Exception as e:
            logger.warning(f"Could not parse end date {end_date_str}: {e}")
            return 0.5

    def _score_category_match(self, market: Dict[str, Any]) -> float:
        """
        Score based on category/topic match with our preferences

        Returns:
            1.0 if matches preferred category, 0.5 otherwise
        """
        question = market.get('question', market.get('title', '')).lower()
        description = market.get('description', market.get('subtitle', '')).lower()
        category = market.get('category', '').lower()

        text_to_check = f"{question} {description} {category}"

        for preferred_cat in self.preferred_categories:
            if preferred_cat.lower() in text_to_check:
                return 1.0

        return 0.5  # Neutral if not in preferred categories

    def filter_and_rank_markets(
        self,
        markets: List[Dict[str, Any]],
        cross_platform_prices: Optional[Dict[str, Dict[str, float]]] = None,
        top_n: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Filter and rank markets by priority score

        Args:
            markets: List of market dicts
            cross_platform_prices: Dict mapping market_slug -> {provider: price}
            top_n: Return only top N markets (None = all)

        Returns:
            List of (market, score) tuples, sorted by score descending
        """
        scored_markets = []

        for market in markets:
            market_slug = market.get('slug', market.get('ticker', market.get('id')))

            # Get cross-platform prices if available
            prices = None
            if cross_platform_prices and market_slug in cross_platform_prices:
                prices = cross_platform_prices[market_slug]

            # Calculate score
            score = self.score_market(market, prices)

            # Apply hard filters
            if score > 0.0:  # Must pass minimum thresholds
                scored_markets.append((market, score))

        # Sort by score descending
        scored_markets.sort(key=lambda x: x[1], reverse=True)

        # Return top N if specified
        if top_n:
            scored_markets = scored_markets[:top_n]

        logger.info(
            f"Filtered {len(scored_markets)} markets from {len(markets)} total "
            f"(avg score: {sum(s for _, s in scored_markets) / len(scored_markets):.2f})"
        )

        return scored_markets

    def explain_score(self, market: Dict[str, Any], cross_platform_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate detailed scoring breakdown for a market

        Useful for debugging and understanding selection logic

        Returns:
            Dict with component scores and explanations
        """
        liquidity_score = self._score_liquidity(market)
        price_diff_score = self._score_price_difference(cross_platform_prices)
        time_score = self._score_time_urgency(market)
        category_score = self._score_category_match(market)

        total_score = (
            liquidity_score * 0.4 +
            price_diff_score * 0.3 +
            time_score * 0.2 +
            category_score * 0.1
        )

        return {
            'total_score': total_score,
            'components': {
                'liquidity': {
                    'score': liquidity_score,
                    'weight': 0.4,
                    'contribution': liquidity_score * 0.4,
                    'value': market.get('liquidity', market.get('volume', 0))
                },
                'price_diff': {
                    'score': price_diff_score,
                    'weight': 0.3,
                    'contribution': price_diff_score * 0.3,
                    'value': cross_platform_prices
                },
                'time_urgency': {
                    'score': time_score,
                    'weight': 0.2,
                    'contribution': time_score * 0.2,
                    'value': market.get('end_date_iso', market.get('close_time'))
                },
                'category': {
                    'score': category_score,
                    'weight': 0.1,
                    'contribution': category_score * 0.1,
                    'matched': any(cat.lower() in market.get('question', '').lower() for cat in self.preferred_categories)
                }
            }
        }


async def fetch_and_score_markets(
    polymarket_markets: List[Dict[str, Any]],
    kalshi_markets: List[Dict[str, Any]],
    selector: Optional[MarketSelector] = None,
    top_n: int = 20
) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
    """
    Convenience function to score markets from multiple platforms

    Args:
        polymarket_markets: List of Polymarket market dicts
        kalshi_markets: List of Kalshi market dicts
        selector: MarketSelector instance (creates default if None)
        top_n: Return top N from each platform

    Returns:
        Dict with 'polymarket' and 'kalshi' lists of (market, score) tuples
    """
    if selector is None:
        selector = MarketSelector()

    # Score each platform separately
    poly_scored = selector.filter_and_rank_markets(polymarket_markets, top_n=top_n)
    kalshi_scored = selector.filter_and_rank_markets(kalshi_markets, top_n=top_n)

    return {
        'polymarket': poly_scored,
        'kalshi': kalshi_scored
    }
