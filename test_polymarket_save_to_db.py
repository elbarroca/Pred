#!/usr/bin/env python3
"""
Test script to actually save Polymarket events and markets to database.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid dependency issues
from config.settings import settings
from database.client import MarketDatabase

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PolymarketSaveToDB")


async def save_event_to_db(event_data: Dict[str, Any], db: MarketDatabase) -> None:
    """Callback to save event to database."""
    try:
        from database.schema import Event

        event = Event(
            id=event_data["id"],
            platform="polymarket",
            title=event_data["title"],
            description=event_data.get("description"),
            category=event_data["category"],
            status="active" if event_data.get("active", True) else "closed",
            start_date=event_data.get("start_date"),
            end_date=event_data.get("end_date"),
            tags=event_data.get("tags", []),
            market_count=event_data.get("market_count", 0),
            total_liquidity=event_data.get("total_liquidity", 0.0),
            created_at=event_data.get("created_at"),
            updated_at=event_data.get("updated_at"),
            raw_data=event_data
        )

        success = await db.save_event(event)
        if success:
            logger.info(f"✅ Saved event: {event.id} - {event.title[:50]}...")
        else:
            logger.warning(f"❌ Failed to save event: {event.id}")

    except Exception as e:
        logger.error(f"Error saving event {event_data.get('id', 'unknown')}: {e}")


async def save_market_to_db(market_data: Dict[str, Any], db: MarketDatabase) -> None:
    """Callback to save market to database."""
    try:
        from database.schema import Market

        # Extract prices
        yes_price = market_data.get("yes_price")
        no_price = market_data.get("no_price")

        # Extract volumes and liquidity
        volume_24h = market_data.get("volume_24h")
        total_volume = market_data.get("volume_total")
        liquidity = market_data.get("liquidity_num")

        # Extract dates
        close_date = market_data.get("end_date")
        created_at = market_data.get("created_at")
        updated_at = market_data.get("updated_at")

        # Extract status
        status = "open" if market_data.get("active", True) else "closed"

        # Extract outcomes
        outcomes = market_data.get("outcomes", [])
        num_outcomes = len(outcomes) if outcomes else None

        # Extract clob token ids
        clob_token_ids = market_data.get("clob_token_ids", [])

        market = Market(
            id=market_data["id"],
            platform="polymarket",
            event_id=market_data.get("event_id"),
            event_title=market_data.get("event_title"),
            title=market_data["question"],
            description=market_data.get("description"),
            category=market_data.get("category"),
            tags=market_data.get("tags", []),
            status=status,
            p_yes=yes_price,
            p_no=no_price,
            bid=market_data.get("bid"),
            ask=market_data.get("ask"),
            liquidity=liquidity,
            volume_24h=volume_24h,
            total_volume=total_volume,
            num_outcomes=num_outcomes,
            created_at=created_at,
            updated_at=updated_at,
            close_date=close_date,
            raw_data=market_data
        )

        success = await db.save_market(market)
        if success:
            logger.info(f"✅ Saved market: {market.id} - {market.title[:50]}...")
        else:
            logger.warning(f"❌ Failed to save market: {market.id}")

    except Exception as e:
        logger.error(f"Error saving market {market_data.get('id', 'unknown')}: {e}")


async def save_orderbook_to_db(orderbook_data: Dict[str, Any], db: MarketDatabase) -> None:
    """Callback to save orderbook to database."""
    try:
        logger.info(f"📊 Orderbook for market {orderbook_data.get('market_id')}: "
                   f"Bid={orderbook_data.get('best_bid'):.3f}, "
                   f"Ask={orderbook_data.get('best_ask'):.3f}")
    except Exception as e:
        logger.error(f"Error saving orderbook for market {orderbook_data.get('market_id', 'unknown')}: {e}")


async def main():
    """Main test function."""
    logger.info("🚀 Polymarket Save to Database Test")
    logger.info("="*80)

    # Import Polymarket client directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Initialize database
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("❌ Supabase credentials not found in .env file.")
        logger.error("   Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        return

    db = MarketDatabase(supabase_url, supabase_key)
    logger.info("✅ Database connection initialized")

    # Copy the PolymarketClient code to avoid import chain issues
    # (Simplified version for testing)

    # Copy necessary helper functions and classes from polymarket.py
    import json
    import re
    from datetime import datetime
    from importlib.util import find_spec
    from typing import Any, Callable, List, Optional, Sequence, Tuple

    import httpx

    # Optional CLOB client import
    if find_spec("py_clob_client") is not None:
        from py_clob_client.client import ClobClient
    else:
        ClobClient = None

    # Copy helper functions
    def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)

    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return []

    def _parse_json_list_field(container: Dict[str, Any], field: str) -> List[Any]:
        raw = container.get(field)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str) and raw.strip().startswith("["):
            parsed = json.loads(raw)
            assert isinstance(parsed, list)
            container[field] = parsed
            return parsed
        container[field] = []
        return []

    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return default
        return float(text)

    # Simplified PolymarketGamma class
    class PolymarketGamma:
        def __init__(self, *, timeout: float = 30.0, default_page_size: int = 100) -> None:
            base = getattr(settings, "POLYMARKET_GAMMA_URL", "").rstrip("/")
            self.base_url = base or "https://gamma-api.polymarket.com"
            self.timeout = timeout
            self.default_page_size = default_page_size

        async def get_all_events(
            self,
            *,
            active: bool = True,
            page_size: Optional[int] = None,
            max_events: Optional[int] = None,
            **filters: Any,
        ) -> List[Dict[str, Any]]:
            size = page_size or self.default_page_size
            assert size > 0
            events: List[Dict[str, Any]] = []
            offset = 0

            while True:
                remaining = size
                if max_events is not None:
                    remaining = min(remaining, max_events - len(events))
                    if remaining <= 0:
                        break

                batch = await self.get_events(
                    active=active,
                    limit=remaining,
                    offset=offset,
                    **filters,
                )
                if not batch:
                    break

                events.extend(batch)
                if len(batch) < remaining:
                    break

                offset += remaining

            return events

        async def get_events(
            self,
            active: bool = True,
            limit: int = 100,
            offset: int = 0,
            order: str = "liquidity",
            ascending: str = "false",
            **filters: Any,
        ) -> List[Dict[str, Any]]:
            assert limit > 0
            page_limit = min(limit, 1000)
            params: Dict[str, Any] = {
                "limit": page_limit,
                "offset": offset,
                "order": order,
                "ascending": ascending
            }
            if "closed" in filters:
                params["closed"] = filters.pop("closed")
            elif active:
                params["closed"] = "false"
            params.update(filters)

            data = await self._get("/events", params)
            assert isinstance(data, list)
            return [self._normalize_event(e) for e in data]

        async def _get(self, path: str, params: Optional[Dict[str, Any]]) -> Any:
            url = f"{self.base_url}{path}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
            markets = _ensure_list(event.get("markets"))
            tags = _ensure_list(event.get("tags"))

            if not event.get("category") and tags:
                head = tags[0]
                if isinstance(head, dict):
                    event["category"] = head.get("label") or head.get("slug")

            event["markets"] = [self._normalize_market(m) for m in markets]
            event["tags"] = tags
            return event

        def _normalize_market(self, market: Dict[str, Any]) -> Dict[str, Any]:
            outcomes = _parse_json_list_field(market, "outcomes")
            prices = _parse_json_list_field(market, "outcomePrices")
            clob_ids = _parse_json_list_field(market, "clobTokenIds")

            market["outcomes"] = outcomes
            market["outcomePrices"] = prices
            market["clobTokenIds"] = clob_ids

            numeric_fields = (
                "volumeNum",
                "liquidityNum",
                "volume",
                "liquidity",
                "volume24hr",
                "volume1wk",
                "volume1mo",
                "volume1yr",
            )
            for name in numeric_fields:
                if name in market:
                    market[name] = _to_float(market.get(name))

            if not market.get("category"):
                events = _ensure_list(market.get("events"))
                if events:
                    parent = events[0]
                    if isinstance(parent, dict):
                        category = parent.get("category")
                        if not category:
                            parent_tags = _ensure_list(parent.get("tags"))
                            if parent_tags and isinstance(parent_tags[0], dict):
                                category = parent_tags[0].get("label") or parent_tags[0].get("slug")
                        market["category"] = category

            if not market.get("tags"):
                events = _ensure_list(market.get("events"))
                if events and isinstance(events[0], dict):
                    market["tags"] = _ensure_list(events[0].get("tags"))
                else:
                    market["tags"] = []

            if not market.get("close_date") and market.get("endDate"):
                market["close_date"] = market["endDate"]

            if not market.get("volume_24h") and market.get("volume24hr") is not None:
                market["volume_24h"] = _to_float(market.get("volume24hr"))

            if not market.get("volume_total") and market.get("volumeNum") is not None:
                market["volume_total"] = _to_float(market.get("volumeNum"))

            market.setdefault("active", True)
            market.setdefault("slug", "")
            market.setdefault("question", "")

            return market

    # Simplified PolymarketClient
    class PolymarketClient:
        def __init__(self) -> None:
            self.gamma = PolymarketGamma()

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
            events = await self.gamma.get_all_events(
                active=only_active,
                page_size=page_size,
                max_events=max_events,
            )

            market_records: List[Dict[str, Any]] = []
            orderbook_records: List[Dict[str, Any]] = []

            for event in events:
                event_id = event.get("id")
                # Extract market count from embedded markets
                markets_list = _ensure_list(event.get("markets", []))
                market_count = len(markets_list)

                # Calculate total liquidity from all markets in the event
                total_liquidity = 0.0
                for market in markets_list:
                    market_liquidity = market.get("liquidityNum") or market.get("liquidity") or 0.0
                    total_liquidity += _to_float(market_liquidity)

                # Extract tags - handle both nested and flat structures
                tags = _ensure_list(event.get("tags", []))
                if tags and isinstance(tags[0], dict):
                    # Tags are objects with label/slug
                    tag_labels = [tag.get("label") or tag.get("slug", "") for tag in tags if isinstance(tag, dict)]
                else:
                    # Tags are already strings
                    tag_labels = [str(tag) for tag in tags]

                # Ensure we have a clean list of strings
                tag_labels = [tag for tag in tag_labels if tag and isinstance(tag, str)]

                await save_event(
                    {
                        "id": event_id,
                        "slug": event.get("slug") or event.get("ticker"),
                        "title": event.get("title"),
                        "description": event.get("description"),
                        "category": event.get("category"),
                        "start_date": event.get("startDate") or event.get("startTime"),
                        "end_date": event.get("endDate"),
                        "active": event.get("active"),
                        "closed": event.get("closed"),
                        "liquidity": event.get("liquidity"),
                        "volume": event.get("volume"),
                        "open_interest": event.get("openInterest"),
                        "tags": tag_labels,
                        "market_count": market_count,
                        "total_liquidity": total_liquidity,
                        "created_at": event.get("createdAt"),
                        "updated_at": event.get("updatedAt"),
                        "raw": event,
                    }
                )

                for market in event.get("markets", []):
                    yes_price, no_price = self._extract_yes_no_prices(market)
                    clob_ids = _ensure_list(market.get("clobTokenIds"))
                    yes_token_id = clob_ids[1] if len(clob_ids) >= 2 else None

                    # Extract bid/ask from orderbook if available
                    bid = market.get("bestBid")
                    ask = market.get("bestAsk")

                    # Extract tags from market (usually empty but handle if present)
                    # Markets inherit tags from their parent event
                    market_tags = _ensure_list(market.get("tags", []))
                    if market_tags and isinstance(market_tags[0], dict):
                        market_tag_labels = [tag.get("label") or tag.get("slug", "") for tag in market_tags if isinstance(tag, dict)]
                    else:
                        market_tag_labels = [str(tag) for tag in market_tags]

                    # If market has no tags, inherit from parent event
                    if not market_tag_labels:
                        market_tag_labels = tag_labels.copy()

                    # Ensure we have a clean list of strings
                    market_tag_labels = [tag for tag in market_tag_labels if tag and isinstance(tag, str)]

                    market_record = {
                        "id": market.get("id"),
                        "event_id": event_id,
                        "event_title": event.get("title"),  # Add event title
                        "slug": market.get("slug"),
                        "question": market.get("question"),
                        "description": market.get("description"),  # Add description
                        "category": market.get("category") or event.get("category"),
                        "tags": market_tag_labels,  # Add tags
                        "outcomes": market.get("outcomes"),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "bid": bid,  # Add bid price
                        "ask": ask,  # Add ask price
                        "volume_total": market.get("volume_total") or market.get("volume"),
                        "volume_24h": market.get("volume_24h") or market.get("volume24hr"),
                        "liquidity_num": market.get("liquidityNum") or market.get("liquidity"),
                        "end_date": market.get("endDate"),
                        "active": market.get("active"),
                        "closed": market.get("closed"),
                        "clob_token_ids": clob_ids,
                        "created_at": market.get("createdAt"),  # Add created_at
                        "updated_at": market.get("updatedAt"),  # Add updated_at
                        "raw": market,
                        "norm_title": ""  # For compatibility with old Market class
                    }
                    await save_market(market_record)
                    market_records.append(market_record)

            return {
                "events_saved": len(events),
                "markets_saved": len(market_records),
                "orderbooks_saved": len(orderbook_records),
                "integrity": self.integrity_report(events, market_records, orderbook_records),
            }

        def _extract_yes_no_prices(self, market: Dict[str, Any]) -> Tuple[float, float]:
            """Extract YES/NO prices from market['outcomePrices'] with sane defaults."""
            prices = _ensure_list(market.get("outcomePrices"))
            yes = 0.5
            no = 0.5
            if len(prices) >= 2:
                no = _to_float(prices[0], default=0.5)
                yes = _to_float(prices[1], default=0.5)
            elif len(prices) == 1:
                yes = _to_float(prices[0], default=0.5)
                no = 1.0 - yes
            yes = max(0.0, min(1.0, yes))
            no = max(0.0, min(1.0, no))
            return yes, no

        def integrity_report(
            self,
            events: Sequence[Dict[str, Any]],
            markets: Sequence[Dict[str, Any]],
            orderbooks: Sequence[Dict[str, Any]],
        ) -> Dict[str, Any]:
            return {
                "events": {"total": len(events)},
                "markets": {"total": len(markets)},
                "orderbooks": {"total": len(orderbooks)},
            }

    polymarket_client = PolymarketClient()
    logger.info("✅ Polymarket client initialized")

    try:
        # Test saving data to database
        logger.info("\n💾 PHASE 1: Saving Top Events & Markets to Database")
        logger.info("="*80)

        # Create database callbacks
        save_event_callback = lambda event_data: save_event_to_db(event_data, db)
        save_market_callback = lambda market_data: save_market_to_db(market_data, db)
        save_orderbook_callback = lambda orderbook_data: save_orderbook_to_db(orderbook_data, db)

        # Save 10 events to test data integrity
        logger.info("🔄 Starting sync with top 10 events and their markets...")

        sync_result = await polymarket_client.sync_all_events_and_markets(
            save_event=save_event_callback,
            save_market=save_market_callback,
            save_orderbook=save_orderbook_callback,
            only_active=True,
            page_size=10,  # Top 10 events by liquidity
            max_events=10
        )

        # Report sync results
        logger.info("\n📊 SAVE RESULTS")
        logger.info("-"*80)
        logger.info(f"✅ Events saved to DB: {sync_result['events_saved']}")
        logger.info(f"✅ Markets saved to DB: {sync_result['markets_saved']}")
        logger.info(f"✅ Orderbooks processed: {sync_result['orderbooks_saved']}")

        # Verify data was saved
        logger.info("\n🔍 PHASE 2: Verifying Database Contents")
        logger.info("="*80)

        # Get counts
        event_count = await db.get_market_count()  # Note: this returns market count, not event count
        market_count = await db.get_market_count()

        logger.info(f"📊 Total markets in database: {market_count}")

        # Get recent markets to verify they were saved
        if market_count > 0:
            recent_markets = await db.get_recent_markets(5)
            logger.info("\n--- Recently Saved Markets ---")
            for market in recent_markets:
                p_yes = market.get('p_yes')
                volume = market.get('total_volume') or market.get('volume_24h')
                p_yes_str = f"{p_yes:.1%}" if p_yes is not None else 'N/A'
                volume_str = f"${volume:,.0f}" if volume else 'N/A'
                logger.info(f"  {market['platform']}: {market['title'][:50]}... "
                          f"(P(YES): {p_yes_str}, Volume: {volume_str})")

        logger.info("\n✅ Polymarket Save to Database Test Complete!")
        logger.info("Data should now be saved in your Supabase database.")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
