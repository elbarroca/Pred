"""Elite Wallet Discovery & Historical Performance Orchestration Engine."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.wallet_tracker import WalletTracker

logger = logging.getLogger(__name__)


class PolymarketWalletCollector:
    """Elite wallet intelligence orchestration engine."""

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        self.wallet_tracker = WalletTracker(api=self.data_api, gamma=self.gamma)
        self.max_concurrent = 10

    async def enrich_unenriched_events(
        self,
        max_events: int = 10,
        is_closed: bool = False,
        skip_existing_positions: bool = True
    ) -> Dict[str, Any]:
        """Master function: Get unenriched events → discover wallets → sync trades/positions → compute stats/scores → mark enriched."""
        logger.info(f"Starting enrichment for {max_events} events")
        start_time = datetime.now(timezone.utc)
        report = {
            "events_processed": 0,
            "wallets_discovered": 0,
            "trades_saved": 0,
            "positions_synced": 0,
            "stats_computed": 0,
            "scores_computed": 0,
            "total_volume": 0.0,
            "total_pnl": 0.0,
            "wallets_skipped": 0
        }

        # Get unenriched events
        events = await self._get_unenriched_events(max_events, is_closed)
        if not events:
            logger.info("No unenriched events found")
            return report

        # Load events lookup cache once
        all_events = await self._fetch_events_from_db(500, is_closed=True)
        events_by_id, events_by_slug = self.wallet_tracker.build_events_lookup(all_events)
        self.wallet_tracker.populate_event_slug_cache(events_by_slug)

        # Process each event sequentially
        for event in events:
            event_id = event["id"]
            logger.info(f"Processing event {event_id[:12]}...")

            # Step 1: Discover wallets from event trades + save trades
            wallets, trades_count = await self._discover_wallets_and_trades(event_id)
            if not wallets:
                logger.info(f"No wallets found for event {event_id[:8]}... - marking enriched")
                await self._mark_event_enriched(event_id, is_closed)
                report["events_processed"] += 1
                continue

            report["wallets_discovered"] += len(wallets)
            report["trades_saved"] += trades_count

            # Step 2: Sync closed positions (skip if exists)
            wallets_to_sync = await self._filter_wallets_needing_sync(wallets) if skip_existing_positions else list(wallets)
            if wallets_to_sync:
                logger.info(f"Syncing positions for {len(wallets_to_sync)} wallets")
                pos_result = await self.wallet_tracker.sync_wallets_closed_positions_batch(
                    proxy_wallets=wallets_to_sync,
                    save_position=self._save_to_db_async("wallet_closed_positions", "proxy_wallet"),
                    max_concurrency=self.max_concurrent
                )
                logger.info(f"Positions synced: {pos_result.get('total_positions', 0)} positions, ${pos_result.get('total_volume', 0):,.2f} volume")
                report["positions_synced"] += pos_result.get("total_positions", 0)
                report["total_volume"] += pos_result.get("total_volume", 0.0)
                report["total_pnl"] += pos_result.get("total_pnl", 0.0)
                report["wallets_skipped"] += len(wallets) - len(wallets_to_sync)
            else:
                logger.info(f"All {len(wallets)} wallets already have positions synced")
                report["wallets_skipped"] += len(wallets)

            # Step 3: Compute stats
            logger.info(f"Computing stats for {len(wallets)} wallets")
            stats_result = await self._compute_stats_batch(wallets, events_by_slug, events_by_id)
            logger.info(f"Stats computed: {stats_result.get('stats_computed', 0)} wallets")
            report["stats_computed"] += stats_result.get("stats_computed", 0)

            # Step 4: Compute scores
            logger.info(f"Computing scores for {len(wallets)} wallets")
            scores_count = await self._compute_scores_batch(wallets)
            logger.info(f"Scores computed: {scores_count} wallets")
            report["scores_computed"] += scores_count

            # Step 5: Mark event as enriched
            logger.info(f"DB: Marking event {event_id[:12]}... as enriched")
            await self._mark_event_enriched(event_id, is_closed)
            report["events_processed"] += 1

        report["duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Enrichment complete: {report['events_processed']} events, {report['wallets_discovered']} wallets")
        return report

    async def _get_unenriched_events(self, limit: int, is_closed: bool) -> List[Dict[str, Any]]:
        """Get top N unenriched events ordered by liquidity."""
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

    async def _discover_wallets_and_trades(self, event_id: str) -> tuple[Set[str], int]:
        """Discover wallets from event trades and save both wallets and trades.
        
        API Optimization:
        - Uses maximum batch_size=10000 (Polymarket API limit) to minimize API calls
        - Rate limit: 75 requests/10s for /trades endpoint (7.5 req/s)
        - Parallelizes API calls to maximize throughput (up to 8 concurrent)
        
        Deduplication:
        - ~50% unique trades is normal (wash trading, pagination duplicates, HFT)
        - Trades are deduplicated globally by ID before saving (upsert handles duplicates)
        - Uses API's native trade ID when available for accurate matching
        - Unique wallets tracked via set() to ensure no duplicates
        
        """
        logger.info(f"Fetching trades for event {event_id[:12]}...")
        batch_size = 10000  # Maximum allowed by Polymarket API
        max_concurrent_requests = 8  # Parallelize up to 8 API calls (within 75/10s limit)
        progress_log_interval = 50000  # Log progress every 50k trades
        
        # Early termination configuration
        DUPLICATE_THRESHOLD = 0.95  # 95% duplicate rate triggers early stop
        CONSECUTIVE_DUPLICATE_BATCHES = 3  # Need 3 consecutive batches of duplicates
        
        # Step 1: Fetch all trades in parallel with early termination detection
        async def fetch_batch(offset_val: int) -> List[Dict[str, Any]]:
            """Fetch a single batch of trades."""
            try:
                trades = await self.data_api.get_trades(event_id=[event_id], limit=batch_size, offset=offset_val)
                return trades or []
            except Exception as e:
                logger.error(f"Error fetching batch at offset {offset_val}: {e}")
                return []

        # Fetch first batch to check if trades exist
        first_batch = await fetch_batch(0)
        if not first_batch:
            logger.info("No trades found")
            return set(), 0
        
        # Track unique trade IDs during fetching for early termination
        seen_trade_ids_during_fetch: Set[str] = set()
        consecutive_duplicate_batches = 0
        early_terminated = False
        
        # Process first batch to initialize tracking
        for trade in first_batch:
            normalized = self.wallet_tracker._normalize_trade(trade, event_id)
            trade_id = normalized.get("id")
            if trade_id:
                seen_trade_ids_during_fetch.add(trade_id)
        
        # Collect all trade batches in parallel with early termination
        all_raw_trades = list(first_batch)  # Start with first batch
        offset = batch_size
        active_tasks = set()
        fetch_errors = 0
        
        logger.info(f"Fetching trades in parallel (max {max_concurrent_requests} concurrent)...")
        
        while True:
            # Stop fetching if we detected too many consecutive duplicate batches
            if early_terminated:
                logger.info(f"Early termination: Stopped fetching after {consecutive_duplicate_batches} consecutive batches with >95% duplicates")
                break
            
            # Start new parallel requests up to concurrency limit
            while len(active_tasks) < max_concurrent_requests and offset < 10000000 and not early_terminated:
                task = asyncio.create_task(fetch_batch(offset))
                active_tasks.add(task)
                offset += batch_size

            if not active_tasks:
                break

            # Wait for at least one to complete
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            active_tasks = pending

            # Process completed batches and check for duplicates
            for task in done:
                try:
                    trades = await task
                    if not trades:
                        # Empty batch means we've reached the end - stop starting new requests
                        offset = 10000000  # Set to safety limit to stop fetching
                        continue
                    
                    # Check for duplicates in this batch
                    new_unique_count = 0
                    batch_trade_ids = []
                    
                    for trade in trades:
                        normalized = self.wallet_tracker._normalize_trade(trade, event_id)
                        trade_id = normalized.get("id")
                        if trade_id:
                            batch_trade_ids.append(trade_id)
                            if trade_id not in seen_trade_ids_during_fetch:
                                seen_trade_ids_during_fetch.add(trade_id)
                                new_unique_count += 1
                    
                    # Calculate duplicate rate for this batch
                    batch_size_actual = len(batch_trade_ids)
                    if batch_size_actual > 0:
                        duplicate_rate = 1.0 - (new_unique_count / batch_size_actual)
                        
                        # Check if this batch is mostly duplicates
                        if duplicate_rate >= DUPLICATE_THRESHOLD:
                            consecutive_duplicate_batches += 1
                            if consecutive_duplicate_batches >= CONSECUTIVE_DUPLICATE_BATCHES:
                                early_terminated = True
                                logger.info(f"Early termination triggered: {consecutive_duplicate_batches} consecutive batches with {duplicate_rate*100:.1f}% duplicates")
                                # Cancel pending tasks
                                for pending_task in active_tasks:
                                    pending_task.cancel()
                                active_tasks.clear()
                                break
                        else:
                            # Reset counter if we got new unique trades
                            consecutive_duplicate_batches = 0
                    
                    # Add trades to collection
                    all_raw_trades.extend(trades)
                    
                    # Progress logging
                    if len(all_raw_trades) % progress_log_interval == 0:
                        unique_so_far = len(seen_trade_ids_during_fetch)
                        logger.info(f"Fetched: {len(all_raw_trades):,} trades | {unique_so_far:,} unique | {consecutive_duplicate_batches} consecutive duplicate batches")
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    fetch_errors += 1
        
        total_fetched = len(all_raw_trades)
        unique_during_fetch = len(seen_trade_ids_during_fetch)
        logger.info(f"Fetch complete: {total_fetched:,} trades fetched | {unique_during_fetch:,} unique IDs detected ({fetch_errors} errors)")
        if early_terminated:
            logger.info(f"Early termination saved ~{total_fetched - unique_during_fetch:,} duplicate API calls")
        
        # Step 2: Extract wallets and normalize trades sequentially (no race conditions)
        wallets = set()
        wallet_data_list = []
        normalized_trades = []
        processed_trade_ids: Set[str] = set()  # Track which trades we've added to normalized_trades
        
        logger.info("Processing trades: extracting wallets and normalizing...")
        
        for trade in all_raw_trades:
            # Extract wallet
            proxy_wallet = trade.get("proxyWallet") or trade.get("proxy_wallet")
            if proxy_wallet and proxy_wallet not in wallets:
                wallets.add(proxy_wallet)
                wallet_data_list.append(self.wallet_tracker._normalize_wallet_from_trade(trade))
            
            # Normalize trade
            normalized_trade = self.wallet_tracker._normalize_trade(trade, event_id)
            trade_id = normalized_trade.get("id")
            
            # Global deduplication: only add if we haven't added this trade ID before
            if trade_id and trade_id not in processed_trade_ids:
                processed_trade_ids.add(trade_id)
                normalized_trades.append(normalized_trade)
        
        unique_trades_count = len(normalized_trades)
        duplicates_removed = total_fetched - unique_trades_count
        
        logger.info(f"Processing complete: {unique_trades_count:,} unique trades | {duplicates_removed:,} duplicates removed | {len(wallets):,} wallets")
        
        # Step 3: Save wallets first (ensures FK constraint satisfied)
        if wallet_data_list:
            logger.info(f"Saving {len(wallet_data_list)} wallets...")
            await self._batch_upsert("wallets", wallet_data_list, "proxy_wallet", check_existing=False)
            logger.info(f"Saved {len(wallet_data_list)} wallets")
        
        # Step 4: Save trades in batches
        if normalized_trades:
            logger.info(f"Saving {unique_trades_count:,} unique trades...")
            saved_count = await self._batch_upsert_with_retry("trades", normalized_trades, "id")
            logger.info(f"Saved {saved_count:,} trades to database")
        else:
            saved_count = 0

        # Step 5: Final verification
        try:
            trade_count_result = self.db.supabase.table("trades").select("id", count="exact").eq("event_id", event_id).execute()
            db_trade_count = trade_count_result.count if hasattr(trade_count_result, 'count') else len(trade_count_result.data) if trade_count_result.data else 0
            
            logger.info(f"Verification: {total_fetched:,} fetched → {unique_trades_count:,} unique → {saved_count:,} saved → {db_trade_count:,} in DB | {len(wallets):,} wallets")
            
            if db_trade_count < unique_trades_count * 0.9:  # Less than 90% saved
                logger.warning(f"WARNING: Only {db_trade_count:,}/{unique_trades_count:,} unique trades in DB ({db_trade_count/unique_trades_count*100:.1f}%)")
            elif db_trade_count == unique_trades_count:
                logger.info(f"✓ All {unique_trades_count:,} unique trades successfully saved")
        except Exception as e:
            logger.warning(f"Could not verify DB counts: {e}")

        return wallets, unique_trades_count

    async def _filter_wallets_needing_sync(self, wallets: Set[str]) -> List[str]:
        """Filter wallets that don't have positions synced yet."""
        if not wallets:
            return []
        wallet_list = list(wallets)
        logger.info(f"DB: Checking {len(wallet_list)} wallets for existing positions")
        result = (self.db.supabase.table("wallet_closed_positions")
                 .select("proxy_wallet")
                 .in_("proxy_wallet", wallet_list)
                 .execute())
        synced = set(p.get("proxy_wallet") for p in result.data if p.get("proxy_wallet"))
        needing_sync = [w for w in wallet_list if w not in synced]
        logger.info(f"Filter: {len(needing_sync)} need sync, {len(synced)} already synced")
        return needing_sync

    async def _compute_stats_batch(
        self,
        wallets: Set[str],
        events_by_slug: Dict[str, Dict[str, Any]],
        events_by_id: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute stats for wallets: global, tag-specific, and market-specific."""
        if not wallets:
            return {"stats_computed": 0}

        wallet_list = list(wallets)
        stats_computed = 0

        async def compute_stats(wallet: str) -> bool:
            positions = await self._fetch_positions(wallet)
            if not positions:
                return False

            # Global stats
            global_stats = self.wallet_tracker.compute_wallet_stats_from_positions(positions)
            global_stats["proxy_wallet"] = wallet
            await self._save_to_db("wallet_stats", global_stats, "proxy_wallet")

            # Tag stats
            tag_stats = self.wallet_tracker.compute_wallet_tag_stats_with_ids(
                wallet, positions, events_by_slug, events_by_id
            )
            for tag_stat in tag_stats:
                await self._save_to_db("wallet_tag_stats", tag_stat, "proxy_wallet")

            # Market stats
            market_stats = self.wallet_tracker.compute_wallet_market_stats(wallet, positions)
            for market_stat in market_stats:
                await self._save_to_db("wallet_market_stats", market_stat, "proxy_wallet")

            return True

        results = await self._process_batches(wallet_list, compute_stats, batch_size=20)
        stats_computed = sum(results)
        return {"stats_computed": stats_computed}

    async def _compute_scores_batch(self, wallets: Set[str]) -> int:
        """Compute and save scores for wallets."""
        scores_computed = 0
        for wallet in wallets:
            stats_result = self.db.supabase.table("wallet_stats").select("*").eq("proxy_wallet", wallet).execute()
            if not stats_result.data:
                continue

            wallet_stats = stats_result.data[0]
            tag_result = (self.db.supabase.table("wallet_tag_stats")
                         .select("*")
                         .eq("proxy_wallet", wallet)
                         .order("roi", desc=True)
                         .limit(1)
                         .execute())
            tag_stats = tag_result.data[0] if tag_result.data else None

            score = self.wallet_tracker.score_wallet(wallet_stats, tag_stats)
            score["proxy_wallet"] = wallet
            await self._save_to_db("wallet_scores", score, "proxy_wallet")
            scores_computed += 1

        return scores_computed

    async def _mark_event_enriched(self, event_id: str, is_closed: bool) -> None:
        """Mark event as enriched."""
        if is_closed:
            await self.db.update_event_closed_enriched_status(event_id, True)
        else:
            await self.db.update_event_enriched_status(event_id, True)

    async def _fetch_events_from_db(self, limit: int, is_closed: bool = False) -> List[Dict[str, Any]]:
        """Fetch events from database."""
        if is_closed:
            result = self.db.supabase.table("events_closed").select("*").limit(limit).execute()
        else:
            result = self.db.supabase.table("events").select("*").eq("status", "active").limit(limit).execute()
        return result.data if result.data else []

    async def _fetch_positions(self, wallet: str) -> List[Dict[str, Any]]:
        """Fetch closed positions for a wallet."""
        result = self.db.supabase.table("wallet_closed_positions").select("*").eq("proxy_wallet", wallet).execute()
        return result.data if result.data else []

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

    async def _check_existing_ids(self, table_name: str, ids: List[str], id_field: str = "id") -> Set[str]:
        """Check which IDs already exist in the database."""
        if not ids:
            return set()
        try:
            # Query in chunks of 1000 (Supabase limit)
            existing = set()
            for i in range(0, len(ids), 1000):
                chunk = ids[i:i + 1000]
                result = self.db.supabase.table(table_name).select(id_field).in_(id_field, chunk).execute()
                if result.data:
                    existing.update(row.get(id_field) for row in result.data if row.get(id_field))
            return existing
        except Exception as e:
            logger.warning(f"Error checking existing IDs in {table_name}: {e}")
            return set()

    async def _batch_upsert(self, table_name: str, data_list: List[Dict[str, Any]], key_field: str, check_existing: bool = False) -> None:
        """Batch upsert data to database - simplified and fast."""
        if not data_list:
            return
        
        # Batch upsert with error handling
        for i in range(0, len(data_list), 1000):
            batch = data_list[i:i + 1000]
            try:
                self.db.supabase.table(table_name).upsert(batch).execute()
            except Exception as e:
                logger.error(f"Failed to upsert {table_name}: {e}")
                if "foreign key" in str(e).lower():
                    logger.error(f"FK constraint violation - wallets not saved before trades")
                raise

    async def _batch_upsert_with_retry(
        self, 
        table_name: str, 
        data_list: List[Dict[str, Any]], 
        key_field: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> int:
        """Batch upsert with retry logic and detailed error tracking.
        
        Returns:
            int: Number of successfully saved records
        """
        if not data_list:
            return 0
        
        saved_count = 0
        failed_batches = []
        total_batches = (len(data_list) + 999) // 1000
        
        for batch_idx, i in enumerate(range(0, len(data_list), 1000), 1):
            batch = data_list[i:i + 1000]
            batch_saved = False
            
            # Retry logic for transient errors
            for attempt in range(max_retries):
                try:
                    self.db.supabase.table(table_name).upsert(batch).execute()
                    saved_count += len(batch)
                    batch_saved = True
                    
                    # Log progress for large batches
                    if batch_idx % 10 == 0 or batch_idx == total_batches:
                        logger.debug(f"Saved batch {batch_idx}/{total_batches} ({saved_count:,} records)")
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check error type
                    if "foreign key" in error_msg or "fk_" in error_msg:
                        logger.error(f"FK constraint violation in batch {batch_idx}/{total_batches}: {e}")
                        # Don't retry FK errors - they indicate a logic issue
                        failed_batches.append({
                            "batch_idx": batch_idx,
                            "size": len(batch),
                            "error": "FK constraint violation",
                            "details": str(e)
                        })
                        break
                    elif "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
                        # Transient error - retry
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Transient error in batch {batch_idx}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Failed to save batch {batch_idx} after {max_retries} attempts: {e}")
                            failed_batches.append({
                                "batch_idx": batch_idx,
                                "size": len(batch),
                                "error": "Network/timeout error",
                                "details": str(e)
                            })
                    else:
                        # Other error - log and don't retry
                        logger.error(f"Failed to save batch {batch_idx}/{total_batches}: {e}")
                        failed_batches.append({
                            "batch_idx": batch_idx,
                            "size": len(batch),
                            "error": "Other error",
                            "details": str(e)
                        })
                        break
            
            if not batch_saved and batch_idx <= 3:
                # Log sample of failed batch for debugging
                logger.error(f"Sample failed batch {batch_idx}: {batch[0] if batch else 'empty'}")
        
        # Summary logging
        if failed_batches:
            failed_count = sum(b["size"] for b in failed_batches)
            logger.warning(f"Saved {saved_count:,}/{len(data_list):,} records to {table_name} ({len(failed_batches)} batches failed, {failed_count:,} records lost)")
        else:
            logger.info(f"Successfully saved all {saved_count:,} records to {table_name}")
        
        return saved_count

    async def _process_batches(self, items: List, process_func, batch_size: int = None) -> List:
        """Execute batch processing with parallelism."""
        batch_size = batch_size or self.max_concurrent
        return [
            result
            for i in range(0, len(items), batch_size)
            for result in await asyncio.gather(*[process_func(item) for item in items[i:i + batch_size]])
        ]


# Standalone function for easy import
async def enrich_unenriched_events(
    max_events: int = 10,
    is_closed: bool = False,
    skip_existing_positions: bool = True
) -> Dict[str, Any]:
    """Master function to enrich unenriched events with wallet intelligence."""
    collector = PolymarketWalletCollector()
    return await collector.enrich_unenriched_events(max_events, is_closed, skip_existing_positions)
