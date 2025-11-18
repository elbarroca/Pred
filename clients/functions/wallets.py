"""Elite Wallet Enrichment & Historical Performance Orchestration Engine.

This module focuses on the slow process of enriching wallets with positions, stats, and scores.
For the fast process of retrieving trades/wallets from events, see retrieve_trades_wallets.py
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.wallet_tracker import WalletTracker
from clients.functions.retrieve_trades_wallets import RetrieveTradesWalletsCollector

logger = logging.getLogger(__name__)


class PolymarketWalletCollector:
    """Elite wallet enrichment orchestration engine - focused on positions, stats, and scores."""

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        self.wallet_tracker = WalletTracker(api=self.data_api, gamma=self.gamma)
        self.max_concurrent = 20  # Increased for better async performance

    async def enrich_wallets_positions(
        self,
        max_wallets: int = 50,
        skip_existing_positions: bool = True
    ) -> Dict[str, Any]:
        """Slow process: Enrich wallets with positions, stats, and scores where enriched=false.
        
        This function:
        1. Gets wallets where enriched=false
        2. Syncs closed positions for each wallet
        3. Computes stats (global, tag, market)
        4. Computes scores
        5. Marks wallets as enriched=true
        
        This is the slow process that can run independently/asynchronously.
        
        Args:
            max_wallets: Maximum number of wallets to process
            skip_existing_positions: Whether to skip wallets that already have positions synced
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info(f"Starting slow process: enrich positions/stats for {max_wallets} wallets")
        start_time = datetime.now(timezone.utc)
        report = {
            "wallets_processed": 0,
            "wallets_enriched": 0,
            "positions_synced": 0,
            "stats_computed": 0,
            "scores_computed": 0,
            "total_volume": 0.0,
            "total_pnl": 0.0,
            "wallets_skipped": 0,
            "duration_seconds": 0.0
        }

        # Get wallets needing enrichment
        wallets = await self.db.get_wallets_needing_enrichment(max_wallets)
        if not wallets:
            logger.info("No wallets needing enrichment found")
            return report

        wallet_addresses = [w["proxy_wallet"] for w in wallets if w.get("proxy_wallet")]
        if not wallet_addresses:
            logger.info("No valid wallet addresses found")
            return report

        logger.info(f"Found {len(wallet_addresses)} wallets needing enrichment")

        # Filter wallets that need position sync
        wallets_to_sync = wallet_addresses
        if skip_existing_positions:
            wallets_to_sync = await self._filter_wallets_needing_sync(set(wallet_addresses))
            report["wallets_skipped"] = len(wallet_addresses) - len(wallets_to_sync)

        # Step 1: Sync closed positions (optimized async)
        if wallets_to_sync:
            logger.info(f"Syncing positions for {len(wallets_to_sync)} wallets (max_concurrency={self.max_concurrent})")
            position_saver = self._save_to_db_async("wallet_closed_positions", "proxy_wallet")
            pos_result = await self.wallet_tracker.sync_wallets_closed_positions_batch(
                proxy_wallets=wallets_to_sync,
                save_position=position_saver,
                save_event=self._save_event_metadata,
                max_concurrency=self.max_concurrent  # Use increased concurrency
            )
            logger.info(f"Positions synced: {pos_result.get('total_positions', 0)} positions, ${pos_result.get('total_volume', 0):,.2f} volume")
            report["positions_synced"] += pos_result.get("total_positions", 0)
            report["total_volume"] += pos_result.get("total_volume", 0.0)
            report["total_pnl"] += pos_result.get("total_pnl", 0.0)
        else:
            logger.info(f"All {len(wallet_addresses)} wallets already have positions synced")
            report["wallets_skipped"] = len(wallet_addresses)

        # Step 2: Delegate stats + scores to database functions
        logger.info(f"DB: running wallet metrics for {len(wallet_addresses)} wallets")
        await self._run_wallet_metrics(wallet_addresses)
        report["stats_computed"] = len(wallet_addresses)
        report["scores_computed"] = len(wallet_addresses)

        # Step 4: Mark wallets as enriched (parallel batch update)
        async def mark_wallet_enriched(wallet_address: str) -> bool:
            return await self.db.update_wallet_enriched_status(wallet_address, True)
        
        # Process in batches for better performance
        enriched_results = await self._process_batches(
            wallet_addresses, 
            mark_wallet_enriched, 
            batch_size=self.max_concurrent
        )
        enriched_count = sum(1 for r in enriched_results if r)

        report["wallets_processed"] = len(wallet_addresses)
        report["wallets_enriched"] = enriched_count
        report["duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Slow process complete: {report['wallets_processed']} wallets enriched")
        return report

    async def enrich_unenriched_events(
        self,
        max_events: int = 10,
        is_closed: bool = False,
        skip_existing_positions: bool = True
    ) -> Dict[str, Any]:
        """Run the fast (trades) and slow (wallet enrichment) flows in sequence."""
        logger.info(f"Running full enrichment for {max_events} events")
        start_time = datetime.now(timezone.utc)
        retriever = RetrieveTradesWalletsCollector()
        fast_report = await retriever.retrieve_trades_and_wallets_from_events(max_events, is_closed)
        target_wallets = max(fast_report.get("wallets_discovered", 0), 1)
        slow_report = await self.enrich_wallets_positions(
            max_wallets=target_wallets,
            skip_existing_positions=skip_existing_positions
        )
        report = {
            "events_processed": fast_report.get("events_processed", 0),
            "wallets_discovered": fast_report.get("wallets_discovered", 0),
            "wallets_new": fast_report.get("wallets_new", 0),
            "wallets_existing": fast_report.get("wallets_existing", 0),
            "trades_saved": fast_report.get("trades_saved", 0),
            "wallets_processed": slow_report.get("wallets_processed", 0),
            "wallets_enriched": slow_report.get("wallets_enriched", 0),
            "positions_synced": slow_report.get("positions_synced", 0),
            "stats_computed": slow_report.get("stats_computed", 0),
            "scores_computed": slow_report.get("scores_computed", 0),
            "wallets_skipped": slow_report.get("wallets_skipped", 0),
            "total_volume": slow_report.get("total_volume", 0.0),
            "total_pnl": slow_report.get("total_pnl", 0.0),
            "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
        }
        logger.info(
            f"Full enrichment complete: {report['events_processed']} events processed, "
            f"{report['wallets_enriched']} wallets enriched"
        )
        return report

    async def _get_unenriched_events(self, limit: int, is_closed: bool) -> List[Dict[str, Any]]:
        """Get top N unenriched events ordered by liquidity (backward compatibility)."""
        logger.info(f"DB: Querying unenriched events (limit={limit}, closed={is_closed})")
        table = "events_closed" if is_closed else "events"
        query = (self.db.supabase.table(table)
                .select("*")
                .eq("enriched", False)
                .gt("market_count", 0)
                .gt("total_liquidity", 0))
        if not is_closed:
            query = query.eq("status", "active")
        query = query.order("total_liquidity", desc=True).order("market_count", desc=True).limit(limit)
        result = query.execute()
        logger.info(f"DB: Found {len(result.data) if result.data else 0} unenriched events")
        return result.data if result.data else []

    async def _filter_wallets_needing_sync(self, wallets: Set[str]) -> List[str]:
        """Filter wallets that don't have positions synced yet (optimized batch query)."""
        if not wallets:
            return []
        wallet_list = list(wallets)
        logger.info(f"DB: Checking {len(wallet_list)} wallets for existing positions")
        
        # Query in batches of 1000 (Supabase limit) for better performance
        synced = set()
        batch_size = 1000
        for i in range(0, len(wallet_list), batch_size):
            batch = wallet_list[i:i + batch_size]
            result = (
                self.db.supabase.table("wallet_closed_positions")
                .select("proxy_wallet")
                .in_("proxy_wallet", batch)
                .execute()
            )
            if result.data:
                synced.update(p.get("proxy_wallet") for p in result.data if p.get("proxy_wallet"))
        
        needing_sync = [w for w in wallet_list if w not in synced]
        logger.info(f"Filter: {len(needing_sync)} need sync, {len(synced)} already synced")
        return needing_sync

    async def _mark_event_enriched(self, event_id: str, is_closed: bool) -> None:
        """Mark event as enriched (backward compatibility)."""
        if is_closed:
            await self.db.update_event_closed_enriched_status(event_id, True)
        else:
            await self.db.update_event_enriched_status(event_id, True)

    async def _run_wallet_metrics(self, wallets: List[str]) -> None:
        """Trigger database-side wallet metrics aggregation."""
        unique_wallets = list({w for w in wallets if w})
        if not unique_wallets:
            return
        self.db.supabase.rpc("run_wallet_metrics", {"p_wallets": unique_wallets}).execute()


    async def _save_to_db(self, table_name: str, data: Dict[str, Any], required_field: Optional[str] = None) -> None:
        """Unified database save using upsert."""
        assert data, f"Data cannot be empty for table {table_name}"
        if required_field:
            assert data.get(required_field), f"Missing required field '{required_field}'"
        self.db.supabase.table(table_name).upsert(data).execute()

    def _save_to_db_async(self, table_name: str, required_field: str):
        """Create async save callback for batch operations."""
        async def save_func(data: Dict[str, Any]) -> None:
            await self._save_to_db(table_name, data, required_field)
        return save_func

    async def _save_event_metadata(self, metadata: Dict[str, Any]) -> None:
        event_id = metadata.get("id")
        if not event_id:
            return

        status = (metadata.get("status") or "").lower()
        table = "events_closed" if status == "closed" else "events"

        def _coerce_tags(raw: Any) -> Any:
            if raw is None:
                return []
            if isinstance(raw, list):
                return raw
            return [raw]

        record = {
            "id": event_id,
            "platform": metadata.get("platform") or "polymarket",
            "slug": metadata.get("slug"),
            "title": metadata.get("title"),
            "description": metadata.get("description"),
            "category": metadata.get("category"),
            "tags": _coerce_tags(metadata.get("tags")),
            "status": metadata.get("status") or ("closed" if table == "events_closed" else "active"),
            "start_date": metadata.get("start_date"),
            "end_date": metadata.get("end_date"),
            "market_count": metadata.get("market_count") or 0,
            "total_liquidity": metadata.get("total_liquidity") or 0.0,
            "total_volume": metadata.get("total_volume") or 0.0,
            "raw_data": metadata.get("raw"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.db.supabase.table(table).upsert(record).execute()
        logger.info(
            "Slow path | upserted event metadata id=%s table=%s market_count=%s total_liquidity=%.2f total_volume=%.2f",
            event_id,
            table,
            record["market_count"],
            record["total_liquidity"],
            record["total_volume"],
        )

    async def _process_batches(self, items: List, process_func, batch_size: int = None) -> List:
        """Execute batch processing with parallelism."""
        batch_size = batch_size or self.max_concurrent
        return [
            result
            for i in range(0, len(items), batch_size)
            for result in await asyncio.gather(*[process_func(item) for item in items[i:i + batch_size]])
        ]


# Standalone functions for easy import
async def enrich_unenriched_events(
    max_events: int = 10,
    is_closed: bool = False,
    skip_existing_positions: bool = True
) -> Dict[str, Any]:
    """Master function to enrich unenriched events with wallet intelligence (backward compatibility)."""
    collector = PolymarketWalletCollector()
    return await collector.enrich_unenriched_events(max_events, is_closed, skip_existing_positions)


async def enrich_wallets_positions(
    max_wallets: int = 50,
    skip_existing_positions: bool = True
) -> Dict[str, Any]:
    """Slow process: Enrich wallets with positions, stats, and scores where enriched=false."""
    collector = PolymarketWalletCollector()
    return await collector.enrich_wallets_positions(max_wallets, skip_existing_positions)
