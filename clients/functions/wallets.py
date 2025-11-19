"""
Elite Wallet Enrichment & Historical Performance Orchestration Engine.
Assertive, High-Performance, and Clean.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.functions.retrieve_wallets import RetrieveTradesWalletsCollector

# Configure logging to fail loud and clear
logger = logging.getLogger(__name__)

class WalletTracker:
    """
    High-performance, in-memory tracker for wallet position fetching and normalization.
    """

    def __init__(
        self,
        api: Optional[PolymarketDataAPI] = None,
        gamma: Optional[PolymarketGamma] = None,
    ):
        self.api = api or PolymarketDataAPI()
        self.gamma = gamma or PolymarketGamma()
        self._event_metadata_cache: Dict[str, Dict[str, Any]] = {}

    async def sync_wallet_closed_positions(
        self,
        proxy_wallet: str,
        save_position_batch: Callable[[List[Dict[str, Any]]], Any],
        save_event_batch: Callable[[List[Dict[str, Any]]], Any],
        last_synced_timestamp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Syncs closed positions for a specific wallet.
        Stops fetching if timestamps go below the last_synced_timestamp.
        """
        assert proxy_wallet, "Proxy wallet address is required"
        assert save_position_batch, "Position save callback required"

        logger.info(f"Syncing wallet: {proxy_wallet} (Cutoff: {last_synced_timestamp})")
        
        all_fetched_positions = []
        offset = 0
        events_to_resolve: Set[str] = set()
        
        # Fetch Loop
        while True:
            batch = await self.api.get_closed_positions(user=proxy_wallet, limit=50, offset=offset)
            if not batch:
                break

            valid_positions_in_batch = []
            
            for pos in batch:
                # 1. Check Cutoff
                ts = float(pos.get("timestamp") or 0)
                if last_synced_timestamp > 0 and ts <= last_synced_timestamp:
                    logger.debug(f"Reached timestamp cutoff for {proxy_wallet}")
                    # We can return immediately here as API returns desc order
                    return self._compile_report(proxy_wallet, all_fetched_positions)

                # 2. Normalize
                normalized = self._normalize_closed_position(pos, proxy_wallet)
                valid_positions_in_batch.append(normalized)

                # 3. Queue Event for Resolution
                if normalized.get("event_slug") and not normalized.get("event_id"):
                    events_to_resolve.add(normalized["event_slug"])

            if not valid_positions_in_batch:
                break

            # 4. Batch Metadata Fetch (Gamma)
            if events_to_resolve:
                await self._prefetch_event_metadata(list(events_to_resolve))
                events_to_resolve.clear()

            # 5. Apply Metadata & Save Events
            unique_events_to_save = []
            seen_event_ids = set()

            for pos in valid_positions_in_batch:
                self._apply_metadata_inplace(pos)
                
                # Collect unique event data for saving
                e_id = pos.get("event_id")
                if e_id and e_id not in seen_event_ids:
                    cached = self._event_metadata_cache.get(e_id)
                    if cached:
                        unique_events_to_save.append(cached)
                        seen_event_ids.add(e_id)

            # 6. Persist Batch
            if unique_events_to_save:
                await save_event_batch(unique_events_to_save)
            
            await save_position_batch(valid_positions_in_batch)
            
            all_fetched_positions.extend(valid_positions_in_batch)

            if len(batch) < 50:
                break
            offset += 50

        return self._compile_report(proxy_wallet, all_fetched_positions)

    def _compile_report(self, wallet: str, positions: List[Dict]) -> Dict[str, Any]:
        event_ids = list(set(p.get("event_id") for p in positions if p.get("event_id")))
        return {
            "wallet": wallet,
            "positions_fetched": len(positions),
            "total_volume": sum(float(p.get("total_bought") or 0) for p in positions),
            "realized_pnl": sum(float(p.get("realized_pnl") or 0) for p in positions),
            "event_ids": event_ids
        }

    async def _prefetch_event_metadata(self, slugs: List[str]) -> None:
        """Fetches event metadata in parallel chunks."""
        missing = [s for s in slugs if f"slug:{s}" not in self._event_metadata_cache]
        if not missing:
            return

        async def fetch(slug: str):
            try:
                data = await self.gamma.get_event(slug)
                if data:
                    self._process_and_cache_event(data, slug)
            except Exception as e:
                logger.warning(f"Failed to fetch metadata for slug {slug}: {e}")

        # Concurrency limit of 10
        chunk_size = 10
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i : i + chunk_size]
            await asyncio.gather(*[fetch(s) for s in chunk])

    def _process_and_cache_event(self, data: Dict, slug_key: str) -> Dict:
        """Normalizes and caches event data."""
        metrics = self._compute_metrics(data)
        
        metadata = {
            "id": str(data.get("id") or ""),
            "slug": data.get("slug") or slug_key,
            "title": data.get("title"),
            "description": data.get("description"),
            "category": data.get("category"),
            "tags": self._extract_tags(data.get("tags")),
            "status": "closed" if data.get("closed") else "active",
            "start_date": data.get("startDate"),
            "end_date": data.get("endDate"),
            "market_count": metrics["market_count"],
            "total_liquidity": metrics["liquidity"],
            "total_volume": metrics["volume"],
            "raw_data": data,
        }

        # Cache by Slug AND ID
        self._event_metadata_cache[f"slug:{slug_key}"] = metadata
        if metadata["id"]:
            self._event_metadata_cache[metadata["id"]] = metadata
        
        return metadata

    def _apply_metadata_inplace(self, position: Dict[str, Any]) -> None:
        """Enriches position dict with cached event metadata."""
        meta = self._event_metadata_cache.get(f"slug:{position.get('event_slug')}")
        
        # Fallback to ID lookup
        if not meta and position.get("event_id"):
            meta = self._event_metadata_cache.get(position.get("event_id"))
            
        if meta:
            position["event_id"] = meta["id"]
            position["event_slug"] = meta["slug"]
            position["event_category"] = meta["category"]
            position["event_tags"] = meta["tags"]

    def _normalize_closed_position(self, pos: Dict[str, Any], wallet: str) -> Dict[str, Any]:
        """Assertive normalization of raw API data."""
        cond_id = pos.get("conditionId") or ""
        idx = pos.get("outcomeIndex", 0)
        
        # Deterministic ID generation
        pos_id = hashlib.sha256(f"{wallet}{cond_id}{idx}".encode()).hexdigest()[:32]

        # Smart field extraction
        slug = pos.get("eventSlug") or pos.get("event_slug") or pos.get("slug")
        
        # Try to find event_id immediately from cache if possible
        event_id = None
        if slug:
            cached = self._event_metadata_cache.get(f"slug:{slug}")
            if cached:
                event_id = cached["id"]

        return {
            "id": pos_id,
            "proxy_wallet": wallet,
            "event_id": event_id,
            "condition_id": cond_id,
            "asset": pos.get("asset"),
            "outcome": pos.get("outcome"),
            "outcome_index": idx,
            "total_bought": float(pos.get("totalBought") or 0),
            "avg_price": float(pos.get("avgPrice") or 0),
            "cur_price": float(pos.get("curPrice") or 0),
            "realized_pnl": float(pos.get("realizedPnl") or 0),
            "timestamp": int(pos.get("timestamp") or 0),
            "end_date": pos.get("endDate"),
            "title": pos.get("title"),
            "slug": pos.get("slug"),
            "event_slug": slug,
            "event_category": pos.get("category") or pos.get("eventCategory"),
            "event_tags": self._extract_tags(pos.get("tags") or pos.get("eventTags")),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": pos
        }

    @staticmethod
    def _extract_tags(raw: Any) -> List[str]:
        """Safe tag extraction."""
        if not raw: return []
        if isinstance(raw, str):
            if raw.startswith("["):
                try: return json.loads(raw)
                except json.JSONDecodeError: return [raw]
            return [raw]
        if isinstance(raw, list):
            return [t if isinstance(t, str) else t.get("label", "") for t in raw]
        return []

    @staticmethod
    def _compute_metrics(event: Dict) -> Dict:
        markets = event.get("markets", []) or []
        vol = float(event.get("volume") or 0)
        if vol == 0 and markets:
            vol = sum(float(m.get("volume") or 0) for m in markets if isinstance(m, dict))
        return {
            "market_count": max(len(markets), int(event.get("marketCount", 0))),
            "liquidity": float(event.get("liquidity") or 0),
            "volume": vol
        }


class PolymarketWalletCollector:
    """
    Orchestrates the pipeline: Database -> API -> Normalization -> Database.
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        
        # Components
        self.wallet_tracker = WalletTracker()
        self.retrieve_collector = RetrieveTradesWalletsCollector()
        
        self._saved_events_cache: Set[str] = set()

    async def enrich_wallets_positions(self, max_wallets: int = 50, skip_existing_positions: bool = True) -> Dict[str, Any]:
        """
        Main Entry Point: Enriches top unenriched wallets.
        """
        start = datetime.now(timezone.utc)
        
        # 1. Fetch Candidates
        candidates = await self.db.get_wallets_needing_enrichment(max_wallets)
        if not candidates:
            logger.info("No wallets needing enrichment found.")
            return {}

        wallets = [w["proxy_wallet"] for w in candidates if w.get("proxy_wallet")]
        
        # 2. Fetch Last Sync Timestamps (Optimization)
        timestamps = await self._get_wallet_timestamps(wallets)

        logger.info(f"Processing {len(wallets)} wallets...")
        
        results = []
        for wallet in wallets:
            try:
                # 3. Sync Logic
                res = await self.wallet_tracker.sync_wallet_closed_positions(
                    proxy_wallet=wallet,
                    save_position_batch=lambda rows: self._bulk_upsert("wallet_closed_positions", rows),
                    save_event_batch=self._save_unique_events,
                    last_synced_timestamp=timestamps.get(wallet, 0.0)
                )
                
                # 4. Discovery (Optional)
                if res["event_ids"]:
                    new_wallets = await self._discover_secondary_wallets(res["event_ids"])
                    res["discovered_wallets"] = len(new_wallets)
                
                # 5. Mark Complete
                await self._mark_enriched(wallet)
                results.append(res)
                
            except Exception as e:
                logger.error(f"❌ Critical failure processing wallet {wallet}: {e}")
                # Continue to next wallet instead of crashing entire batch
                continue

        summary = {
            "wallets_processed": len(results),
            "total_positions": sum(r["positions_fetched"] for r in results),
            "duration": (datetime.now(timezone.utc) - start).total_seconds()
        }
        logger.info(f"✅ Enrichment Batch Complete: {summary}")
        return summary

    async def enrich_unenriched_events(self, max_events: int = 10, is_closed: bool = False, skip_existing_positions: bool = True) -> Dict[str, Any]:
        """
        Two-Step Flow: 
        1. Find wallets participating in specific events (RetrieveTradesWalletsCollector).
        2. Enrich those discovered wallets fully.
        """
        logger.info(f"Starting Event-Based Enrichment (Events: {max_events})")
        
        # Step 1: Discovery
        discovery_report = await self.retrieve_collector.retrieve_wallets_from_events(max_events, is_closed)
        discovered_count = discovery_report.get("wallets_discovered", 0)
        
        # Step 2: Enrichment
        if discovered_count > 0:
            logger.info(f"Triggering enrichment for {discovered_count} discovered wallets...")
            enrichment_report = await self.enrich_wallets_positions(
                max_wallets=max(discovered_count, 1),
                skip_existing_positions=skip_existing_positions
            )
            discovery_report["enrichment_stats"] = enrichment_report
            
        return discovery_report

    # --- Internal Helpers ---

    async def _save_unique_events(self, events: List[Dict[str, Any]]):
        """Filters duplicates before saving events."""
        to_save = []
        for e in events:
            eid = e.get("id")
            if eid and eid not in self._saved_events_cache:
                to_save.append(e)
                self._saved_events_cache.add(eid)
        
        if to_save:
            await self._bulk_upsert("events_closed", to_save)

    async def _bulk_upsert(self, table: str, rows: List[Dict[str, Any]]) -> None:
        """Assertive bulk upsert with exponential backoff."""
        if not rows: return
        
        batch_size = 100
        max_retries = 3
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    self.db.supabase.table(table).upsert(batch).execute()
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"FINAL FAILURE upserting to {table}: {e}")
                        raise e # Fail loud after retries exhausted
                    
                    delay = 2 ** attempt
                    logger.warning(f"Upsert failed ({e}), retrying in {delay}s...")
                    await asyncio.sleep(delay)

    async def _get_wallet_timestamps(self, wallets: List[str]) -> Dict[str, float]:
        """Gets max timestamp for wallets to allow incremental sync."""
        if not wallets: return {}
        try:
            res = self.db.supabase.rpc("get_wallets_max_timestamp", {"wallets": wallets}).execute()
            if res.data:
                return {row['wallet']: float(row['max_ts']) for row in res.data}
        except Exception as e:
            logger.warning(f"Timestamp fetch failed ({e}). Defaulting to full sync.")
        return {w: 0.0 for w in wallets}

    async def _mark_enriched(self, wallet: str):
        """Marks wallet as enriched in DB."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.supabase.table("wallets").update({
            "enriched": True,
            "enriched_at": now,
            "updated_at": now
        }).eq("proxy_wallet", wallet).execute()

    async def _discover_secondary_wallets(self, event_ids: List[str]) -> Set[str]:
        """Uses retrieve collector to find new wallets from event IDs."""
        new_wallets = set()
        
        # Process in small chunks to avoid overloading API
        chunk_size = 5
        for i in range(0, len(event_ids), chunk_size):
            chunk = event_ids[i:i+chunk_size]
            tasks = [self.retrieve_collector._discover_wallets_only(eid) for eid in chunk]
            results = await asyncio.gather(*tasks)
            
            for r in results:
                new_wallets.update(r)

        # Filter against DB
        if new_wallets:
            existing = await self.retrieve_collector._check_existing_wallets(list(new_wallets))
            truly_new = new_wallets - existing
            
            if truly_new:
                # Save placeholders
                placeholder_rows = [{
                    "proxy_wallet": w, 
                    "enriched": False,
                    "created_at": datetime.now(timezone.utc).isoformat()
                } for w in truly_new]
                await self._bulk_upsert("wallets", placeholder_rows)
                return truly_new

        return set()

# --- Exported Functions ---

async def enrich_unenriched_events(max_events: int = 10, is_closed: bool = False, skip_existing_positions: bool = True) -> Dict[str, Any]:
    collector = PolymarketWalletCollector()
    return await collector.enrich_unenriched_events(max_events, is_closed, skip_existing_positions)

async def enrich_wallets_positions(max_wallets: int = 50, skip_existing_positions: bool = True) -> Dict[str, Any]:
    collector = PolymarketWalletCollector()
    return await collector.enrich_wallets_positions(max_wallets, skip_existing_positions)