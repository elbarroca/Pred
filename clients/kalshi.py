"""
Kalshi API Client

Unified API for Kalshi prediction markets (events/markets/orderbooks).

Goals:
- Fetch and normalize all Kalshi events and markets.
- Enrich markets with probabilities and orderbook metrics.
- Provide a simple interface to persist normalized data into a database.
"""

import asyncio
import base64
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config import settings


logger = logging.getLogger(__name__)


class KalshiError(Exception):
    """Base Kalshi exception."""


class MarketNotFoundError(KalshiError):
    """Raised when a specific market or event cannot be found."""


class InvalidOrderbookError(KalshiError):
    """Raised when an orderbook is structurally invalid or empty."""


class AuthenticationError(KalshiError):
    """Raised when authentication fails."""




def _ensure_list(value: Any) -> List[Any]:
    """Return value as list; strings and None become empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safe float conversion with a bounded domain for prices and probabilities."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle dollar strings like "$1,234.56"
        text = value.strip().replace("$", "").replace(",", "")
        if not text:
            return default
        return float(text)
    return default


class KalshiAuth:
    """
    Handle Kalshi RSA signature authentication.

    Kalshi v2 API requires RSA-PSS signatures for authenticated requests.
    Each request needs:
    - KALSHI-ACCESS-KEY: API key ID
    - KALSHI-ACCESS-SIGNATURE: RSA-PSS signed message (base64)
    - KALSHI-ACCESS-TIMESTAMP: Current timestamp in milliseconds
    """

    def __init__(self, api_key_id: str, private_key_path: Optional[str] = None) -> None:
        self.api_key_id = api_key_id
        self.private_key: Optional[rsa.RSAPrivateKey] = None

        if private_key_path and Path(private_key_path).exists():
            self.private_key = self._load_private_key(private_key_path)
            logger.info(f"Loaded Kalshi private key from {private_key_path}")
        elif api_key_id:
            logger.warning("Kalshi API key provided but no private key found - will use public endpoints only")

    def _load_private_key(self, key_path: str) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        try:
            with open(key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise AuthenticationError("Invalid private key type - must be RSA")
            return private_key
        except Exception as e:
            raise AuthenticationError(f"Failed to load private key: {e}")

    def get_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Generate auth headers for request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/trade-api/v2/markets")

        Returns:
            Dict with KALSHI-ACCESS-* headers
        """
        if not self.private_key or not self.api_key_id:
            return {}

        timestamp = str(int(time.time() * 1000))
        signature = self._sign_message(timestamp, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _sign_message(self, timestamp: str, method: str, path: str) -> str:
        """
        Create RSA-PSS signature for request.

        Message format: "{timestamp}{method}{path}"
        Example: "1234567890000GET/trade-api/v2/markets"
        """
        if not self.private_key:
            raise AuthenticationError("No private key available for signing")

        message = f"{timestamp}{method}{path}".encode("utf-8")

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode("utf-8")


class KalshiAPI:
    """
    Low-level Kalshi REST API wrapper.

    Handles:
    - HTTP requests with authentication
    - Cursor-based pagination
    - Rate limiting
    - Error handling
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        rate_limit: float = 15.0,
    ) -> None:
        base = getattr(settings, "KALSHI_API_URL", "").rstrip("/")
        self.base_url = base or "https://api.elections.kalshi.com/trade-api/v2"
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

        # Initialize authentication
        api_key_id = getattr(settings, "KALSHI_API_KEY_ID", "").strip()
        private_key_path = getattr(settings, "KALSHI_PRIVATE_KEY_PATH", "").strip()
        self.auth = KalshiAuth(api_key_id, private_key_path if private_key_path else None)

    async def get_events(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        with_nested_markets: bool = False,
        **filters: Any,
    ) -> Dict[str, Any]:
        """
        Retrieve a single page of events.

        Args:
            limit: Page size (1-1000).
            cursor: Pagination cursor from previous response.
            with_nested_markets: If True, include nested markets in response.
            **filters: Additional query parameters (status, series_ticker, etc.).

        Returns:
            Dict with 'events' list and 'cursor' string.
        """
        params: Dict[str, Any] = {"limit": min(limit, 1000)}
        if cursor:
            params["cursor"] = cursor
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        params.update(filters)

        data = await self._get("/events", params)
        return data

    async def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: str = "active",
        **filters: Any,
    ) -> Dict[str, Any]:
        """
        Retrieve a single page of markets.

        Args:
            limit: Page size (1-1000).
            cursor: Pagination cursor from previous response.
            status: Market status filter (active, closed, settled, etc.).
            **filters: Additional query parameters (event_ticker, series_ticker, etc.).

        Returns:
            Dict with 'markets' list and 'cursor' string.
        """
        params: Dict[str, Any] = {"limit": min(limit, 1000)}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        params.update(filters)

        data = await self._get("/markets", params)
        return data

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve a single market by ticker.

        Args:
            ticker: Market ticker (e.g., "KXBTCD-25JAN18-T104249.99")

        Returns:
            Market object dict.
        """
        data = await self._get(f"/markets/{ticker}", None)
        return data.get("market", data)

    async def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """
        Retrieve a single event by ticker.

        Args:
            event_ticker: Event ticker (e.g., "KXBTCD-25JAN18")

        Returns:
            Event object dict.
        """
        data = await self._get(f"/events/{event_ticker}", None)
        return data.get("event", data)

    async def get_orderbook(self, ticker: str, depth: int = 50) -> Dict[str, Any]:
        """
        Retrieve orderbook for a market.

        Args:
            ticker: Market ticker.
            depth: Maximum depth per side (default 50).

        Returns:
            Dict with 'yes' and 'no' bid arrays.
        """
        params = {"depth": depth}
        data = await self._get(f"/markets/{ticker}/orderbook", params)
        return data.get("orderbook", data)

    async def _get(self, path: str, params: Optional[Dict[str, Any]]) -> Any:
        """Low-level GET helper with rate limiting and auth."""
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        url = f"{self.base_url}{path}"
        headers = self.auth.get_headers("GET", f"/trade-api/v2{path}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            self._last_request_time = time.time()

            if response.status_code == 401:
                raise AuthenticationError("Kalshi authentication failed - check API key and signature")

            response.raise_for_status()
            return response.json()


class KalshiClient:
    """
    High-level Kalshi client.

    Responsibilities:
    - Expose simple methods to fetch events/markets and enrich them with orderbook data.
    - Provide formatting helpers for display.
    - Provide utilities to sync normalized events/markets/orderbooks into a database.
    """

    def __init__(self) -> None:
        self.api = KalshiAPI()

    async def get_events(
        self,
        limit: int = 10,
        with_markets: bool = False,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve multiple events.

        Args:
            limit: Max number of events to return.
            with_markets: If True, include nested markets.
            **filters: Additional filters (status, series_ticker, etc.).

        Returns:
            List of normalized event dicts.
        """
        events: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        while len(events) < limit:
            remaining = limit - len(events)
            page_size = min(remaining, 100)

            response = await self.api.get_events(
                limit=page_size,
                cursor=cursor,
                with_nested_markets=with_markets,
                **filters,
            )

            batch = response.get("events", [])
            if not batch:
                break

            events.extend([self._normalize_event(e) for e in batch])

            cursor = response.get("cursor", "")
            if not cursor:
                break

        return events[:limit]

    async def get_markets(
        self,
        limit: int = 200,
        min_liquidity: float = 100.0,
        max_spread: float = 0.98,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve markets enriched with normalized data.

        Args:
            limit: Maximum number of markets in the final result.
            min_liquidity: Minimum notional liquidity (USD) required to keep a market.
            max_spread: Maximum allowed spread in absolute price terms (0-1).
            **filters: Additional filters (status, event_ticker, etc.).

        Returns:
            List of normalized markets.
        """
        markets: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        # Fetch more than needed to account for filtering
        target = limit * 3

        while len(markets) < target:
            remaining = target - len(markets)
            page_size = min(remaining, 100)

            response = await self.api.get_markets(
                limit=page_size,
                cursor=cursor,
                **filters,
            )

            batch = response.get("markets", [])
            if not batch:
                break

            for market in batch:
                normalized = self._normalize_market(market)

                # Filter by liquidity and spread
                liquidity = _to_float(normalized.get("liquidity_num"))
                spread = _to_float(normalized.get("spread"))

                if liquidity >= min_liquidity and spread <= max_spread:
                    markets.append(normalized)

                if len(markets) >= limit:
                    break

            cursor = response.get("cursor", "")
            if not cursor or len(markets) >= limit:
                break

        logger.info("kalshi.get_markets: %d markets enriched", len(markets))
        return markets[:limit]

    async def sync_all_events_and_markets(
        self,
        save_event: Callable[[Dict[str, Any]], None],
        save_market: Callable[[Dict[str, Any]], None],
        save_orderbook: Optional[Callable[[Dict[str, Any]], None]] = None,
        *,
        only_active: bool = True,
        page_size: int = 100,
        max_events: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Crawl Kalshi API for events + markets and persist them through callbacks.

        Args:
            save_event: Callable that persists a normalized event record.
            save_market: Callable that persists a normalized market record.
            save_orderbook: Optional callable to persist orderbook metrics.
            only_active: If True, restrict to active markets.
            page_size: Page size for crawling.
            max_events: Optional hard cap on events to ingest.

        Returns:
            Summary dict with counts and an integrity report.
        """
        # Fetch events with nested markets
        events_raw: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        logger.info("Starting Kalshi sync: fetching events with nested markets...")

        while True:
            remaining = page_size
            if max_events is not None:
                remaining = min(remaining, max_events - len(events_raw))
                if remaining <= 0:
                    break

            response = await self.api.get_events(
                limit=remaining,
                cursor=cursor,
                with_nested_markets=True,
                status="active" if only_active else None,
            )

            batch = response.get("events", [])
            if not batch:
                break

            events_raw.extend(batch)
            logger.info(f"Fetched {len(batch)} events (total: {len(events_raw)})")

            cursor = response.get("cursor", "")
            if not cursor:
                break

            if max_events and len(events_raw) >= max_events:
                break

        # Process and save events/markets
        market_records: List[Dict[str, Any]] = []
        orderbook_records: List[Dict[str, Any]] = []

        for event_raw in events_raw:
            event = self._normalize_event(event_raw)
            event_id = event.get("event_ticker")

            # Extract market count and liquidity from embedded markets
            markets_list = _ensure_list(event.get("markets", []))
            market_count = len(markets_list)

            total_liquidity = 0.0
            for market in markets_list:
                market_liquidity = _to_float(market.get("liquidity_num"))
                total_liquidity += market_liquidity

            # Extract category for tags
            category = event.get("category", "")
            series_ticker = event.get("series_ticker", "")
            tags = [category, series_ticker] if category else [series_ticker]
            tags = [t for t in tags if t]

            # Save event
            await save_event(
                {
                    "id": event_id,
                    "slug": event_id,  # Kalshi doesn't have slugs, use ticker
                    "title": event.get("title"),
                    "description": "",  # Kalshi events don't have descriptions
                    "category": category,
                    "start_date": None,  # Derive from markets if needed
                    "end_date": None,
                    "active": only_active,
                    "closed": not only_active,
                    "liquidity": total_liquidity,
                    "volume": 0.0,  # Not available at event level
                    "open_interest": 0,
                    "tags": tags,
                    "market_count": market_count,
                    "total_liquidity": total_liquidity,
                    "created_at": None,
                    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "raw": event,
                }
            )

            # Save markets
            for market_raw in markets_list:
                market = self._normalize_market(market_raw)

                yes_price, no_price = self._extract_yes_no_prices(market)

                market_record = {
                    "id": market.get("ticker"),
                    "event_id": event_id,
                    "event_title": event.get("title"),
                    "slug": market.get("ticker"),  # Kalshi has no slug
                    "question": market.get("title"),
                    "description": market.get("rules_primary", ""),
                    "category": category,
                    "tags": tags,
                    "outcomes": ["YES", "NO"],
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "bid": market.get("bid"),
                    "ask": market.get("ask"),
                    "volume_total": market.get("volume_total"),
                    "volume_24h": market.get("volume_24h"),
                    "liquidity_num": market.get("liquidity_num"),
                    "end_date": market.get("close_time"),
                    "active": market.get("status") == "active",
                    "closed": market.get("status") in ["closed", "settled"],
                    "clob_token_ids": [],  # Kalshi doesn't use token IDs
                    "created_at": market.get("created_time"),
                    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "raw": market,
                }

                await save_market(market_record)
                market_records.append(market_record)

        integrity = self.integrity_report(events_raw, market_records, orderbook_records)

        return {
            "events_saved": len(events_raw),
            "markets_saved": len(market_records),
            "orderbooks_saved": len(orderbook_records),
            "integrity": integrity,
        }

    def integrity_report(
        self,
        events: Sequence[Dict[str, Any]],
        markets: Sequence[Dict[str, Any]],
        orderbooks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a simple integrity report over events, markets and orderbooks."""
        total_events = len(events)
        total_markets = len(markets)
        total_orderbooks = len(orderbooks)

        events_with_markets = sum(1 for e in events if _ensure_list(e.get("markets")))
        events_without_category = sum(1 for e in events if not e.get("category"))

        markets_with_category = sum(1 for m in markets if m.get("category"))
        markets_zero_volume = sum(
            1 for m in markets
            if _to_float(m.get("volume_total")) == 0.0 and _to_float(m.get("volume_24h")) == 0.0
        )

        return {
            "events": {
                "total": total_events,
                "with_markets": events_with_markets,
                "without_markets": total_events - events_with_markets,
                "without_category": events_without_category,
            },
            "markets": {
                "total": total_markets,
                "with_category": markets_with_category,
                "without_category": total_markets - markets_with_category,
                "zero_volume": markets_zero_volume,
            },
            "orderbooks": {
                "total": total_orderbooks,
            },
        }

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Kalshi Event payload.

        - Ensures 'markets' is always a list.
        - Normalizes embedded markets.
        """
        markets = _ensure_list(event.get("markets"))
        event["markets"] = [self._normalize_market(m) for m in markets]
        return event

    def _normalize_market(self, market: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Kalshi Market payload.

        - Converts cents (0-100) to probabilities (0-1).
        - Adds convenience aliases for compatibility.
        """
        # Convert prices from cents to 0-1 probabilities
        yes_bid = _to_float(market.get("yes_bid"), 50.0) / 100.0
        yes_ask = _to_float(market.get("yes_ask"), 50.0) / 100.0
        no_bid = _to_float(market.get("no_bid"), 50.0) / 100.0
        no_ask = _to_float(market.get("no_ask"), 50.0) / 100.0
        last_price = _to_float(market.get("last_price"), 50.0) / 100.0

        # Calculate mid-market probabilities
        yes_price = (yes_bid + yes_ask) / 2.0
        no_price = (no_bid + no_ask) / 2.0

        # Spread and liquidity
        spread = yes_ask - yes_bid
        spread_bps = int(spread * 10000)

        # Volume and liquidity
        volume = _to_float(market.get("volume"))
        volume_24h = _to_float(market.get("volume_24h"))
        liquidity = _to_float(market.get("liquidity_dollars", market.get("liquidity")))

        # Add normalized fields
        market["yes_price"] = yes_price
        market["no_price"] = no_price
        market["bid"] = yes_bid
        market["ask"] = yes_ask
        market["mid_price"] = yes_price
        market["spread"] = spread
        market["spread_bps"] = spread_bps
        market["last_price_normalized"] = last_price

        market["volume_total"] = volume
        market["volume_24h"] = volume_24h
        market["liquidity_num"] = liquidity

        # Ensure required fields exist
        market.setdefault("ticker", "")
        market.setdefault("title", "")
        market.setdefault("status", "active")

        return market

    def _extract_yes_no_prices(self, market: Dict[str, Any]) -> Tuple[float, float]:
        """Extract YES/NO prices from normalized market with sane defaults."""
        yes_price = _to_float(market.get("yes_price"), 0.5)
        no_price = _to_float(market.get("no_price"), 0.5)

        # Clamp to [0, 1]
        yes_price = max(0.0, min(1.0, yes_price))
        no_price = max(0.0, min(1.0, no_price))

        return yes_price, no_price
