"""Open Positions Collector for Elite Traders."""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketDataAPI

logger = logging.getLogger(__name__)

class WalletOpenPositionsCollector:
    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.api = PolymarketDataAPI()

    async def run(self, max_wallets: int = 100, force_refresh: bool = False) -> Dict[str, Any]:
        """Main execution pipeline."""
        start = datetime.now(timezone.utc)

        # 0. Cleanup expired positions
        await self._cleanup_expired_positions()

        # 1. Fetch Context (Wallets + Metadata + Exclusions)
        targets, context = await self._get_processing_context(max_wallets, force_refresh)
        if not targets:
            return self._stats(0, 0, start)

        logger.info(f"Processing {len(targets)} wallets")
        
        # 2. Process in Batches
        stats = {"found": 0, "stored": 0}
        batch_size = 50
        
        for i in range(0, len(targets), batch_size):
            batch = targets[i:i + batch_size]
            
            # Fetch all API data concurrently
            tasks = [self.api.get_positions(w, limit=500) for w in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Transform valid results
            write_queue = []
            for wallet, positions in zip(batch, results):
                if isinstance(positions, list) and positions:
                    meta = context.get(wallet, {})
                    write_queue.extend(
                        self._transform(p, wallet, meta) for p in positions if self._is_valid(p)
                    )

            # Bulk Write
            if write_queue:
                self.db.supabase.table("elite_open_positions").upsert(write_queue).execute()
                stats["found"] += len(write_queue)
                stats["stored"] += len(write_queue)
                logger.info(f"Batch {i//batch_size + 1}: Upserted {len(write_queue)} positions")

        return self._stats(len(targets), stats["stored"], start)

    async def _get_processing_context(self, limit: int, force: bool) -> tuple[List[str], Dict[str, Any]]:
        """Aggressively fetches targets and metadata - no mercy, no fallbacks."""
        # Execute queries assertively
        traders_res = (self.db.supabase.table("elite_traders")
                      .select("*")
                      .order("composite_score", desc=True)
                      .limit(limit)
                      .execute())

        recent_res = None if force else (self.db.supabase.table("elite_open_positions")
                                        .select("proxy_wallet")
                                        .gte("updated_at", (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat())
                                        .execute())

        # Build elite trader context - check data exists
        traders = traders_res.data
        if not traders:
            logger.warning("No elite traders found - database empty or query failed")
            return [], {}

        context_map = {
            t["proxy_wallet"]: {
                "wallet_rank": t.get("rank_in_tier"),
                "composite_score": t.get("composite_score"),
                "roi": t.get("roi"),
                "win_rate": t.get("win_rate"),
                "total_volume": t.get("total_volume"),
                "n_positions": t.get("n_positions"),
                "n_markets": t.get("n_markets"),
                "trader_tier": t.get("tier")
            } for t in traders
        }

        # Filter targets aggressively
        targets = list(context_map.keys())
        if not force and recent_res and recent_res.data:
            recent_wallets = {r["proxy_wallet"] for r in recent_res.data}
            targets = [t for t in targets if t not in recent_wallets]

        return targets, context_map

    async def _cleanup_expired_positions(self) -> None:
        """Remove positions for events that have already ended."""
        today = datetime.now(timezone.utc).date().isoformat()
        result = (self.db.supabase.table("elite_open_positions")
                 .delete()
                 .lt("event_end_date", today)
                 .execute())

        deleted_count = len(result.data) if result.data else 0
        if deleted_count > 0:
            logger.info(f"🧹 Cleaned up {deleted_count} expired positions")
        else:
            logger.debug("No expired positions found to clean up")

    def _is_valid(self, p: Dict) -> bool:
        """Fast validation check - only include positions for future events."""
        # Check required fields exist
        if not p.get("conditionId") or p.get("outcomeIndex") is None:
            logger.debug(f"Position missing critical fields: {p.get('conditionId')}")
            return False

        # Check event is still active (end date is today or future)
        end_date_str = p.get("endDate")
        if not end_date_str:
            logger.debug(f"Position missing end date: {p.get('conditionId')}")
            return False

        end_date = datetime.fromisoformat(end_date_str).date()
        today = datetime.now(timezone.utc).date()
        return end_date >= today

    def _transform(self, p: Dict, wallet: str, meta: Dict) -> Dict:
        """Pure data transformation - assertive and direct."""
        # Extract values assertively
        size = float(p["size"])
        cur_price = float(p["curPrice"])
        entry_price = float(p["avgPrice"])

        # Generate ID
        pid = hashlib.sha256(f"{wallet}{p['conditionId']}{p['outcomeIndex']}".encode()).hexdigest()[:32]
        now = datetime.now(timezone.utc).isoformat()

        # Core position data - no fallbacks, no mercy
        return {
            "id": pid,
            "proxy_wallet": wallet,
            "event_id": p.get("eventId"),
            "condition_id": p["conditionId"],
            "asset": p.get("asset"),
            "outcome": p["outcome"],
            "outcome_index": p["outcomeIndex"],
            "size": size,
            "avg_entry_price": entry_price,
            "current_price": cur_price,
            "unrealized_pnl": size * (cur_price - entry_price),
            "position_value": size * cur_price,
            "cash_pnl": float(p["cashPnl"]),
            "initial_value": float(p["initialValue"]),
            "title": p["market"]["title"] if p.get("market") else p["title"],
            "slug": p["market"]["slug"] if p.get("market") else p["slug"],
            "event_slug": p["event"]["slug"] if p.get("event") else p["eventSlug"],
            "event_end_date": p["endDate"],
            "redeemable": bool(p["redeemable"]),
            "mergeable": bool(p["mergeable"]),
            "negative_risk": bool(p["negativeRisk"]),
            "raw_data": p,
            "created_at": now,
            "updated_at": now
        } | meta

    def _stats(self, count: int, stored: int, start: datetime) -> Dict:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "wallets_processed": count,
            "positions_stored": stored,
            "duration": duration,
            "rate": stored / duration if duration > 0 else 0
        }

# Direct Entry Point
async def collect_elite_trader_open_positions(max_wallets: int = 100, force_refresh: bool = False):
    return await WalletOpenPositionsCollector().run(max_wallets, force_refresh)