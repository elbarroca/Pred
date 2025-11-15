"""
Async database client for Supabase with prediction market data persistence.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from supabase import create_client, Client
from .schema import Event, Market, ScanSession


class MarketDatabase:
    """Async database client for storing prediction market data.
    
    Provides high-level async interface for persisting events, markets, and scan sessions
    to Supabase with batch support and thread-safe operations.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize database client with Supabase credentials.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            
        Raises:
            AssertionError: If URL or key is empty
        """
        assert supabase_url, "supabase_url must not be empty"
        assert supabase_key, "supabase_key must not be empty"
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._lock = asyncio.Lock()

    @staticmethod
    def _serialize(obj: Any) -> Optional[str]:
        """Serialize object to JSON if not None.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON string or None
        """
        return json.dumps(obj) if obj else None

    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert Event schema to database record.
        
        Args:
            event: Event object
            
        Returns:
            Dictionary ready for database insertion
        """
        assert event.id, "Event must have id"
        assert event.platform, "Event must have platform"
        
        return {
            "id": event.id,
            "platform": event.platform,
            "title": event.title,
            "description": event.description,
            "category": event.category,
            "status": event.status,
            "start_date": event.start_date,
            "end_date": event.end_date,
            "tags": self._serialize(event.tags),
            "market_count": event.market_count,
            "total_liquidity": event.total_liquidity,
            "created_at": event.created_at,
            "updated_at": event.updated_at,
            "raw_data": self._serialize(event.raw_data),
        }

    def _market_to_dict(self, market: Market) -> Dict[str, Any]:
        """Convert Market schema to database record.
        
        Args:
            market: Market object
            
        Returns:
            Dictionary ready for database insertion
        """
        assert market.id, "Market must have id"
        assert market.platform, "Market must have platform"
        
        return {
            "id": market.id,
            "platform": market.platform,
            "event_id": market.event_id,
            "event_title": market.event_title,
            "title": market.title,
            "description": market.description,
            "category": market.category,
            "tags": self._serialize(market.tags),
            "status": market.status,
            "p_yes": market.p_yes,
            "p_no": market.p_no,
            "bid": market.bid,
            "ask": market.ask,
            "liquidity": market.liquidity,
            "volume_24h": market.volume_24h,
            "total_volume": market.total_volume,
            "num_outcomes": market.num_outcomes,
            "created_at": market.created_at,
            "updated_at": market.updated_at,
            "close_date": market.close_date,
            "raw_data": self._serialize(market.raw_data),
        }

    async def save_event(self, event: Event) -> bool:
        """Save a single event to the database.
        
        Args:
            event: Event object to persist
            
        Returns:
            True if upsert succeeded, False otherwise
        """
        async with self._lock:
            data = self._event_to_dict(event)
            result = self.supabase.table("events").upsert(data).execute()
            return len(result.data) > 0

    async def save_market(self, market: Market) -> bool:
        """Save a single market to the database.
        
        Args:
            market: Market object to persist
            
        Returns:
            True if upsert succeeded, False otherwise
        """
        async with self._lock:
            data = self._market_to_dict(market)
            result = self.supabase.table("markets").upsert(data).execute()
            return len(result.data) > 0

    async def save_events_batch(self, events: List[Event]) -> int:
        """Save multiple events in a batch.
        
        Args:
            events: List of Event objects to persist
            
        Returns:
            Count of successfully saved events
        """
        assert events, "events list must not be empty"
        
        saved_count = 0
        for event in events:
            if await self.save_event(event):
                saved_count += 1
        return saved_count

    async def save_markets_batch(self, markets: List[Market]) -> int:
        """Save multiple markets in a batch.
        
        Args:
            markets: List of Market objects to persist
            
        Returns:
            Count of successfully saved markets
        """
        assert markets, "markets list must not be empty"
        
        saved_count = 0
        for market in markets:
            if await self.save_market(market):
                saved_count += 1
        return saved_count

    async def create_scan_session(self, platforms: List[str]) -> Optional[str]:
        """Create a new scan session record.
        
        Args:
            platforms: List of platforms scanned
            
        Returns:
            Session ID if created, None otherwise
        """
        assert platforms, "platforms list must not be empty"
        
        data = {
            "started_at": datetime.utcnow().isoformat(),
            "platforms_scanned": json.dumps(platforms),
            "status": "running"
        }
        result = self.supabase.table("scan_sessions").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    async def update_scan_session(self, session_id: str, **updates) -> None:
        """Update a scan session with completion stats.
        
        Args:
            session_id: ID of scan session to update
            **updates: Fields to update
        """
        assert session_id, "session_id must not be empty"
        assert updates, "updates must contain at least one field"
        
        updates["db_updated_at"] = datetime.utcnow().isoformat()
        self.supabase.table("scan_sessions").update(updates).eq("id", session_id).execute()

    async def get_recent_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recently added markets for verification.
        
        Args:
            limit: Maximum number of markets to retrieve (default: 100)
            
        Returns:
            List of market records ordered by creation date (newest first)
        """
        assert limit > 0, "limit must be positive"
        
        result = (self.supabase.table("markets")
                 .select("*")
                 .order("created_at", desc=True)
                 .limit(limit)
                 .execute())
        return result.data if result.data else []

    async def get_market_count(self) -> int:
        """Get total count of markets in database.
        
        Returns:
            Total number of market records
        """
        result = self.supabase.table("markets").select("id", count="exact").execute()
        return result.count or 0