"""
Polymarket API Client
Integrates both Gamma API (metadata) and CLOB API (prices/orderbooks)
"""
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
from py_clob_client.client import ClobClient
from config import settings


class PolymarketGammaClient:
    """Client for Polymarket Gamma API (market metadata)"""

    def __init__(self):
        self.base_url = settings.POLYMARKET_GAMMA_URL
        self.markets_endpoint = f"{self.base_url}/markets"
        self.events_endpoint = f"{self.base_url}/events"

    async def get_market(self, market_slug: str) -> Optional[Dict[str, Any]]:
        """
        Get a single market by slug

        Args:
            market_slug: Market identifier (e.g., 'will-trump-win-2024')

        Returns:
            Market metadata dict or None if not found
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.markets_endpoint,
                    params={"slug": market_slug},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                # Gamma API returns array, get first match
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return None
            except httpx.HTTPError as e:
                print(f"Error fetching market {market_slug}: {e}")
                return None

    async def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Get multiple markets with filters

        Args:
            active: Only active markets (default True)
            limit: Max results to return
            offset: Pagination offset
            **filters: Additional query params (e.g., tag_id, category)

        Returns:
            List of market metadata dicts
        """
        params = {
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
            **filters
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.markets_endpoint,
                    params=params,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"Error fetching markets: {e}")
                return []

    async def search_markets(self, query: str) -> List[Dict[str, Any]]:
        """
        Search markets by text query

        Args:
            query: Search string

        Returns:
            List of matching markets
        """
        # Gamma API doesn't have native search, so we fetch and filter
        markets = await self.get_markets(limit=200)
        query_lower = query.lower()

        return [
            m for m in markets
            if query_lower in m.get('question', '').lower()
            or query_lower in m.get('description', '').lower()
        ]


class PolymarketCLOBClient:
    """Client for Polymarket CLOB API (prices, orderbooks)"""

    def __init__(self):
        """Initialize CLOB client (read-only mode)"""
        # Skip initialization if no valid private key
        if not settings.POLYMARKET_PRIVATE_KEY or settings.POLYMARKET_PRIVATE_KEY == '...':
            self.client = None
            return

        try:
            self.client = ClobClient(
                host=settings.POLYMARKET_CLOB_URL,
                key=settings.POLYMARKET_PRIVATE_KEY,
                chain_id=137  # Polygon
            )
        except ImportError:
            print("Warning: py-clob-client not installed. Install with: pip install py-clob-client")
            self.client = None

    async def get_orderbook(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get orderbook for a specific token

        Args:
            token_id: Token ID from market metadata

        Returns:
            Orderbook dict with bids/asks
        """
        if not self.client:
            return None

        try:
            # py-clob-client methods are sync, not async
            orderbook = self.client.get_order_book(token_id)
            return orderbook
        except Exception as e:
            print(f"Error fetching orderbook for {token_id}: {e}")
            return None

    async def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """
        Get current price for a token

        Args:
            token_id: Token ID
            side: "BUY" or "SELL"

        Returns:
            Price as float (0.0-1.0) or None
        """
        if not self.client:
            return None

        try:
            price = self.client.get_price(token_id, side=side)
            return float(price) if price else None
        except Exception as e:
            print(f"Error fetching price for {token_id}: {e}")
            return None

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """
        Get midpoint price (average of bid/ask)

        Args:
            token_id: Token ID

        Returns:
            Midpoint price or None
        """
        if not self.client:
            return None

        try:
            midpoint = self.client.get_midpoint(token_id)
            return float(midpoint) if midpoint else None
        except Exception as e:
            print(f"Error fetching midpoint for {token_id}: {e}")
            return None

    async def get_simplified_markets(self) -> List[Dict[str, Any]]:
        """
        Get all simplified markets from CLOB

        Returns:
            List of simplified market dicts
        """
        if not self.client:
            return []

        try:
            markets = self.client.get_simplified_markets()
            return markets if markets else []
        except Exception as e:
            print(f"Error fetching simplified markets: {e}")
            return []


class PolymarketClient:
    """Unified Polymarket client combining Gamma and CLOB APIs"""

    def __init__(self):
        self.gamma = PolymarketGammaClient()
        self.clob = PolymarketCLOBClient()

    async def get_market_with_price(self, market_slug: str) -> Optional[Dict[str, Any]]:
        """
        Get market metadata + current price

        Args:
            market_slug: Market identifier

        Returns:
            Combined dict with metadata and pricing
        """
        market = await self.gamma.get_market(market_slug)
        if not market:
            return None

        # Extract token IDs for YES/NO outcomes
        tokens = market.get('tokens', [])
        if len(tokens) >= 2:
            # Usually [NO_token, YES_token]
            yes_token_id = tokens[1].get('token_id')
            no_token_id = tokens[0].get('token_id')

            # Get prices
            yes_price = await self.clob.get_midpoint(yes_token_id)
            no_price = await self.clob.get_midpoint(no_token_id)

            market['prices'] = {
                'yes': yes_price,
                'no': no_price,
                'implied_probability': yes_price,
                'fetched_at': datetime.utcnow().isoformat()
            }

        return market

    async def get_current_price(self, market_slug: str) -> Optional[float]:
        """
        Get just the current implied probability (YES price)

        Args:
            market_slug: Market identifier

        Returns:
            Implied probability as float or None
        """
        market = await self.get_market_with_price(market_slug)
        if market and 'prices' in market:
            return market['prices'].get('implied_probability')
        return None
