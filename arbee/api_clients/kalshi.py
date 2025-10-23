"""
Kalshi API Client
Supports both public (no auth) and authenticated endpoints
"""
import httpx
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings


class KalshiClient:
    """Client for Kalshi prediction market API"""

    def __init__(self):
        self.base_url = settings.KALSHI_API_URL.rstrip('/')  # Remove trailing slash
        self.api_key_id = settings.KALSHI_API_KEY_ID
        self.api_key = settings.KALSHI_API_KEY  # API key for authentication
        self.private_key_path = settings.KALSHI_PRIVATE_KEY_PATH
        self.private_key = None

        # Load private key if available (for authenticated requests)
        if self.private_key_path and Path(self.private_key_path).exists():
            self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key for request signing"""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend

            with open(self.private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
        except Exception as e:
            print(f"Warning: Could not load Kalshi private key: {e}")

    def _sign_request(self, method: str, path: str, timestamp: int) -> Optional[str]:
        """
        Generate RSA-PSS signature for authenticated requests

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (without query params)
            timestamp: Request timestamp in milliseconds

        Returns:
            Base64-encoded signature or None
        """
        if not self.private_key:
            return None

        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64

            # Signature message: timestamp + method + path
            message = f"{timestamp}{method}{path}"

            signature = self.private_key.sign(
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            print(f"Error signing request: {e}")
            return None

    def _get_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Generate authentication headers for Kalshi API

        Args:
            method: HTTP method
            path: Request path (without query string)

        Returns:
            Dict of headers
        """
        headers = {}

        # Try API key authentication first (simpler method)
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        elif self.api_key_id and self.private_key:
            # Fallback to RSA signature method
            timestamp = int(time.time() * 1000)
            signature = self._sign_request(method, path, timestamp)

            if signature:
                headers.update({
                    'KALSHI-ACCESS-KEY': self.api_key_id,
                    'KALSHI-ACCESS-TIMESTAMP': str(timestamp),
                    'KALSHI-ACCESS-SIGNATURE': signature
                })

        return headers

    async def get_events(
        self,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get events from Kalshi

        Args:
            series_ticker: Filter by series (e.g., "PRES2024")
            status: Event status ("open", "closed", "settled")
            limit: Max results

        Returns:
            List of event dicts
        """
        params = {
            "status": status,
            "limit": limit
        }
        if series_ticker:
            params["series_ticker"] = series_ticker

        async with httpx.AsyncClient() as client:
            try:
                # Get auth headers for this request
                auth_headers = self._get_auth_headers("GET", "/events")

                response = await client.get(
                    f"{self.base_url}/events",
                    params=params,
                    headers=auth_headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get('events', [])
            except httpx.HTTPError as e:
                print(f"Error fetching Kalshi events: {e}")
                return []

    async def get_event(self, event_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific event by ticker

        Args:
            event_ticker: Event identifier

        Returns:
            Event dict or None
        """
        async with httpx.AsyncClient() as client:
            try:
                # Get auth headers for this request
                auth_headers = self._get_auth_headers("GET", f"/events/{event_ticker}")

                response = await client.get(
                    f"{self.base_url}/events/{event_ticker}",
                    headers=auth_headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get('event')
            except httpx.HTTPError as e:
                print(f"Error fetching event {event_ticker}: {e}")
                return None

    async def get_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get markets from Kalshi

        Args:
            event_ticker: Filter by event
            series_ticker: Filter by series
            status: Market status
            limit: Max results

        Returns:
            List of market dicts
        """
        params = {
            "status": status,
            "limit": limit
        }
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker

        async with httpx.AsyncClient() as client:
            try:
                # Get auth headers for this request
                auth_headers = self._get_auth_headers("GET", "/markets")

                response = await client.get(
                    f"{self.base_url}/markets",
                    params=params,
                    headers=auth_headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get('markets', [])
            except httpx.HTTPError as e:
                print(f"Error fetching Kalshi markets: {e}")
                return []

    async def get_market(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific market by ticker

        Args:
            market_ticker: Market identifier

        Returns:
            Market dict with pricing info
        """
        async with httpx.AsyncClient() as client:
            try:
                # Get auth headers for this request
                auth_headers = self._get_auth_headers("GET", f"/markets/{market_ticker}")

                response = await client.get(
                    f"{self.base_url}/markets/{market_ticker}",
                    headers=auth_headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get('market')
            except httpx.HTTPError as e:
                print(f"Error fetching market {market_ticker}: {e}")
                return None

    async def get_orderbook(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get orderbook for a market

        Args:
            market_ticker: Market identifier

        Returns:
            Orderbook dict with bids/asks
        """
        async with httpx.AsyncClient() as client:
            try:
                # Get auth headers for this request
                auth_headers = self._get_auth_headers("GET", f"/markets/{market_ticker}/orderbook")

                response = await client.get(
                    f"{self.base_url}/markets/{market_ticker}/orderbook",
                    headers=auth_headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get('orderbook')
            except httpx.HTTPError as e:
                print(f"Error fetching orderbook for {market_ticker}: {e}")
                return None

    async def search_markets(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for markets matching query

        Args:
            query: Search text

        Returns:
            List of matching markets
        """
        # Get all open markets and filter
        markets = await self.get_markets(limit=200)
        query_lower = query.lower()

        return [
            m for m in markets
            if query_lower in m.get('title', '').lower()
            or query_lower in m.get('subtitle', '').lower()
        ]

    async def get_market_price(self, market_ticker: str) -> Optional[float]:
        """
        Get current implied probability for a market

        Args:
            market_ticker: Market identifier

        Returns:
            Implied probability (0.0-1.0) or None
        """
        market = await self.get_market(market_ticker)
        if not market:
            return None

        # Kalshi uses prices in cents (0-100), but the API structure may vary
        # According to API docs, we should use the midpoint of bid/ask or last_price

        # Try to calculate midpoint from best bid/ask (dollars format)
        yes_bid_dollars = market.get('yes_bid_dollars')
        no_ask_dollars = market.get('no_ask_dollars')

        if yes_bid_dollars is not None and no_ask_dollars is not None:
            # Midpoint between yes_bid_dollars and no_ask_dollars gives the implied probability
            midpoint = (float(yes_bid_dollars) + float(no_ask_dollars)) / 2.0
            return midpoint  # Already in dollars (0-1)

        # Try to calculate midpoint from best bid/ask (cents format)
        yes_bid = market.get('yes_bid')
        no_ask = market.get('no_ask')

        if yes_bid is not None and no_ask is not None:
            # Midpoint between yes_bid and no_ask gives the implied probability
            midpoint = (float(yes_bid) + float(no_ask)) / 2.0
            return midpoint / 100.0  # Convert from cents to probability

        # Fallback: try last_price (cents format)
        last_price = market.get('last_price')
        if last_price is not None:
            return float(last_price) / 100.0

        # Fallback: try last_price_dollars (dollars format)
        last_price_dollars = market.get('last_price_dollars')
        if last_price_dollars is not None:
            return float(last_price_dollars)

        # Fallback: try yes_bid or yes_ask individually (cents format)
        yes_bid = market.get('yes_bid')
        if yes_bid is not None:
            return float(yes_bid) / 100.0

        yes_ask = market.get('yes_ask')
        if yes_ask is not None:
            return float(yes_ask) / 100.0

        # Final fallback: try yes_bid_dollars or yes_ask_dollars (dollars format)
        yes_bid_dollars = market.get('yes_bid_dollars')
        if yes_bid_dollars is not None:
            return float(yes_bid_dollars)

        yes_ask_dollars = market.get('yes_ask_dollars')
        if yes_ask_dollars is not None:
            return float(yes_ask_dollars)

        return None
