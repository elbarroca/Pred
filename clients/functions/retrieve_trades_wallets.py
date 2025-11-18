"""Retrieve trades and wallets for events that still need raw data."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketDataAPI
from clients.wallet_tracker import WalletTracker
from postgrest import APIError

logger = logging.getLogger(__name__)


class RetrieveTradesWalletsCollector:
    """Fast path: fetch trades, persist wallets, mark events."""

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.data_api = PolymarketDataAPI()
        self.wallet_tracker = WalletTracker(api=self.data_api)
        self.trade_batch = 10_000
        self.max_requests = 8

    async def retrieve_trades_and_wallets_from_events(
        self,
        max_events: int = 10,
        is_closed: bool = False
    ) -> Dict[str, Any]:
        """Return a summary after syncing trades/wallets for pending events."""
        logger.info(f"Starting fast process: retrieve trades and wallets for {max_events} events")
        start_time = datetime.now(timezone.utc)
        report = {
            "events_processed": 0,
            "wallets_discovered": 0,
            "wallets_new": 0,
            "wallets_existing": 0,
            "trades_saved": 0,
            "duration_seconds": 0.0
        }

        # Get events where retrieve_data=false
        events = await self._get_events_needing_data_retrieval(max_events, is_closed)
        if not events:
            logger.info("No events needing data retrieval found")
            return report

        total_events = len(events)
        for idx, event in enumerate(events, start=1):
            event_id = event["id"]
            event_slug = event.get("slug")
            logger.info(
                "Fast path | [%d/%d] event=%s slug=%s | starting trade/wallet discovery",
                idx,
                total_events,
                event_id,
                event_slug,
            )

            # Discover wallets and trades
            wallets, trades_count = await self._discover_wallets_and_trades(event_id)
            if not wallets:
                logger.info(f"No wallets found for event {event_id[:8]}... - marking retrieve_data=true")
                await self._mark_event_retrieve_data(event_id, is_closed)
                report["events_processed"] += 1
                continue

            # Check which wallets are new vs existing
            existing_wallets = await self._check_existing_wallets(list(wallets))
            new_wallets = wallets - existing_wallets

            report["wallets_discovered"] += len(wallets)
            report["wallets_new"] += len(new_wallets)
            report["wallets_existing"] += len(existing_wallets)
            report["trades_saved"] += trades_count
            logger.info(
                "Fast path | event=%s | wallets total=%d new=%d existing=%d trades_saved=%d",
                event_id,
                len(wallets),
                len(new_wallets),
                len(existing_wallets),
                trades_count,
            )

            # Mark event as retrieve_data=true
            logger.info("Fast path | marking event=%s retrieve_data=true", event_id)
            await self._mark_event_retrieve_data(event_id, is_closed)
            report["events_processed"] += 1

        report["duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Fast process complete: {report['events_processed']} events, {report['wallets_discovered']} wallets, {report['trades_saved']} trades")
        return report

    async def _get_events_needing_data_retrieval(self, limit: int, is_closed: bool) -> List[Dict[str, Any]]:
        """Get top N events where retrieve_data=false, ordered by liquidity."""
        logger.info(f"DB: Querying events needing data retrieval (limit={limit}, closed={is_closed})")
        table = "events_closed" if is_closed else "events"
        query = (self.db.supabase.table(table)
                .select("*")
                .eq("retrieve_data", False)
                .gt("market_count", 0)
                .gt("total_liquidity", 0))
        if not is_closed:
            query = query.eq("status", "active")
        query = query.order("total_liquidity", desc=True).order("market_count", desc=True).limit(limit)
        result = query.execute()
        logger.info(f"DB: Found {len(result.data) if result.data else 0} events needing data retrieval")
        return result.data if result.data else []

    async def _discover_wallets_and_trades(self, event_id: str) -> tuple[Set[str], int]:
        trades = await self._fetch_all_trades(event_id)
        if not trades:
            return set(), 0
        wallets, wallet_rows, trade_rows = self._normalize_trades(event_id, trades)
        logger.info(
            "Fast path | event=%s | fetched trades=%d unique_wallets=%d",
            event_id,
            len(trades),
            len(wallets),
        )
        if wallet_rows:
            await self._bulk_upsert("wallets", wallet_rows)
        if trade_rows:
            await self._bulk_upsert("trades", trade_rows)
        self._verify_trade_counts(event_id, len(trades), len(trade_rows), len(wallets))
        return wallets, len(trade_rows)

    async def _fetch_all_trades(self, event_id: str) -> List[Dict[str, Any]]:
        trades, pending, offset = [], set(), 0

        def schedule() -> None:
            nonlocal offset
            pending.add(asyncio.create_task(self.data_api.get_trades(event_id=[event_id], limit=self.trade_batch, offset=offset)))
            offset += self.trade_batch

        for _ in range(self.max_requests):
            schedule()

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                batch = task.result() or []
                if not batch:
                    pending.clear()
                    break
                trades.extend(batch)
                if len(batch) == self.trade_batch:
                    schedule()
        return trades

    def _normalize_trades(
        self, event_id: str, raw_trades: List[Dict[str, Any]]
    ) -> tuple[Set[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        wallets, wallet_rows, trade_rows, seen_ids = set(), [], [], set()
        for trade in raw_trades:
            wallet = trade.get("proxyWallet") or trade.get("proxy_wallet")
            if wallet and wallet not in wallets:
                wallets.add(wallet)
                wallet_rows.append(self.wallet_tracker._normalize_wallet_from_trade(trade))
            normalized = self.wallet_tracker._normalize_trade(trade, event_id)
            trade_id = normalized.get("id")
            if trade_id and trade_id not in seen_ids:
                seen_ids.add(trade_id)
                trade_rows.append(normalized)
        return wallets, wallet_rows, trade_rows

    async def _bulk_upsert(self, table: str, rows: List[Dict[str, Any]], chunk: int = 1000) -> None:
        for batch in self._chunk(rows, chunk):
            self.db.supabase.table(table).upsert(batch).execute()
            logger.info("Fast path | upserted %d rows into %s", len(batch), table)

    def _verify_trade_counts(self, event_id: str, fetched: int, unique: int, wallets: int) -> None:
        count_result = (
            self.db.supabase.table("trades")
            .select("id", count="exact")
            .eq("event_id", event_id)
            .execute()
        )
        stored = count_result.count if hasattr(count_result, "count") else len(count_result.data or [])
        logger.info(
            f"Event {event_id[:12]}: fetched {fetched:,}, unique {unique:,}, stored {stored:,}, wallets {wallets:,}"
        )

    async def _mark_event_retrieve_data(self, event_id: str, is_closed: bool) -> None:
        """Mark event as retrieve_data=true."""
        await self.db.update_event_retrieve_data_status(event_id, True, is_closed)


    @staticmethod
    def _chunk(items: List[Any], size: int) -> List[Any]:
        for i in range(0, len(items), size):
            yield items[i:i + size]

    async def _check_existing_wallets(self, wallet_addresses: List[str]) -> Set[str]:
        """Check which wallets already exist in the database."""
        if not wallet_addresses:
            return set()
        existing = set()
        for batch in self._chunk(wallet_addresses, 200):
            try:
                result = (
                    self.db.supabase.table("wallets")
                    .select("proxy_wallet")
                    .in_("proxy_wallet", batch)
                    .execute()
                )
                existing.update(
                    row["proxy_wallet"]
                    for row in (result.data or [])
                    if row.get("proxy_wallet")
                )
                logger.info(
                    "Fast path | existence check chunk size=%d found=%d",
                    len(batch),
                    len(result.data or []),
                )
            except APIError as err:
                logger.warning(
                    "Fast path | wallet existence check failed for chunk size=%d: %s",
                    len(batch),
                    err,
                )
        return existing



# Standalone function for easy import
async def retrieve_trades_and_wallets(
    max_events: int = 10,
    is_closed: bool = False
) -> Dict[str, Any]:
    """Fast process: Retrieve trades and wallets from events where retrieve_data=false."""
    collector = RetrieveTradesWalletsCollector()
    return await collector.retrieve_trades_and_wallets_from_events(max_events, is_closed)

