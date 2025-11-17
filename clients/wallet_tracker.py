"""Polymarket Wallet Tracker & Copy-Trading Analysis Client."""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from clients.polymarket import PolymarketDataAPI, PolymarketGamma

logger = logging.getLogger(__name__)


class WalletTracker:
    """High-level wallet tracking and copy-trading analysis orchestrator."""

    def __init__(
        self,
        api: Optional[PolymarketDataAPI] = None,
        gamma: Optional[PolymarketGamma] = None,
        min_volume: float = 10000.0,
        min_markets: int = 20,
        min_win_rate: float = 0.40
    ):
        self.api = api or PolymarketDataAPI()
        self.gamma = gamma or PolymarketGamma()
        self.min_volume = min_volume
        self.min_markets = min_markets
        self.min_win_rate = min_win_rate
        self._event_slug_to_id_cache: Dict[str, str] = {}

    def compute_wallet_stats_from_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute global wallet statistics from closed positions."""
        if not positions:
            return {
                "total_volume": 0.0, "realized_pnl": 0.0, "roi": 0.0,
                "n_positions": 0, "n_wins": 0, "n_losses": 0, "win_rate": 0.0,
                "n_markets": 0, "n_events": 0, "is_eligible": False
            }

        total_volume = sum(p.get("totalBought") or p.get("total_bought") or 0 for p in positions)
        realized_pnl = sum(p.get("realizedPnl") or p.get("realized_pnl") or 0 for p in positions)
        roi = (realized_pnl / total_volume) if total_volume > 0 else 0.0

        n_positions = len(positions)
        n_wins = sum(1 for p in positions if (p.get("realizedPnl") or p.get("realized_pnl") or 0) > 0)
        win_rate = n_wins / n_positions if n_positions > 0 else 0.0

        n_markets = len(set(p.get("conditionId") or p.get("condition_id") for p in positions if p.get("conditionId") or p.get("condition_id")))
        n_events = len(set(p.get("eventSlug") or p.get("event_slug") for p in positions if p.get("eventSlug") or p.get("event_slug")))

        timestamps = [p.get("timestamp") for p in positions if p.get("timestamp")]
        first_trade_at = last_trade_at = None
        if timestamps:
            first_trade_at = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat()
            last_trade_at = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat()

        is_eligible = total_volume >= self.min_volume and n_positions >= self.min_markets and win_rate >= self.min_win_rate

        return {
            "total_volume": total_volume,
            "avg_position_size": total_volume / n_positions if n_positions > 0 else 0.0,
            "realized_pnl": realized_pnl,
            "roi": roi,
            "n_positions": n_positions,
            "n_wins": n_wins,
            "n_losses": n_positions - n_wins,
            "win_rate": win_rate,
            "n_markets": n_markets,
            "n_events": n_events,
            "first_trade_at": first_trade_at,
            "last_trade_at": last_trade_at,
            "is_eligible": is_eligible,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

    def compute_wallet_tag_stats_with_ids(
        self,
        proxy_wallet: str,
        positions: List[Dict[str, Any]],
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        events_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Compute wallet statistics grouped by event tags with proper IDs."""
        if not events_by_slug and not events_by_id:
            return []

        tag_positions: Dict[str, List[Dict[str, Any]]] = {}

        for position in positions:
            event = None
            event_slug = position.get("eventSlug") or position.get("event_slug")
            if events_by_slug and event_slug:
                event = events_by_slug.get(event_slug)
            elif events_by_id and position.get("event_id"):
                event = events_by_id.get(position.get("event_id"))

            if not event:
                continue

            tags = set(event.get("tags", []))
            if category := event.get("category"):
                tags.add(category)

            for tag in tags:
                tag_positions.setdefault(tag, []).append(position)

        tag_stats = []
        for tag, tag_pos in tag_positions.items():
            stats = self.compute_wallet_stats_from_positions(tag_pos)
            tag_id = hashlib.sha256(f"{proxy_wallet}_{tag}".encode()).hexdigest()[:32]
            stats["id"] = tag_id
            stats["proxy_wallet"] = proxy_wallet
            stats["tag"] = tag
            stats.pop("is_eligible", None)
            tag_stats.append(stats)

        return tag_stats

    def compute_wallet_market_stats(
        self,
        proxy_wallet: str,
        positions: List[Dict[str, Any]],
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Compute wallet statistics per market (condition_id)."""
        market_positions: Dict[str, List[Dict[str, Any]]] = {}
        for pos in positions:
            condition_id = pos.get("condition_id") or pos.get("conditionId")
            if condition_id:
                market_positions.setdefault(condition_id, []).append(pos)

        market_stats = []
        for condition_id, market_pos in market_positions.items():
            wallet_volume = sum(p.get("total_bought") or p.get("totalBought") or 0 for p in market_pos)
            realized_pnl = sum(p.get("realized_pnl") or p.get("realizedPnl") or 0 for p in market_pos)

            timestamps = [p.get("timestamp", 0) for p in market_pos if p.get("timestamp")]
            first_trade_ts = min(timestamps) if timestamps else None
            last_trade_ts = max(timestamps) if timestamps else None

            event_id = event_slug = market_title = None
            for p in market_pos:
                if not event_id:
                    event_id = p.get("event_id")
                if not event_slug:
                    event_slug = p.get("event_slug") or p.get("eventSlug")
                if not market_title:
                    market_title = p.get("title")

            market_stat_id = hashlib.sha256(f"{proxy_wallet}_{condition_id}".encode()).hexdigest()[:32]

            market_stats.append({
                "id": market_stat_id,
                "proxy_wallet": proxy_wallet,
                "event_id": event_id,
                "condition_id": condition_id,
                "wallet_volume": wallet_volume,
                "market_volume": 0.0,
                "volume_share": 0.0,
                "n_trades": len(market_pos),
                "realized_pnl": realized_pnl,
                "first_trade_ts": first_trade_ts,
                "last_trade_ts": last_trade_ts,
                "market_title": market_title,
                "event_slug": event_slug,
                "computed_at": datetime.now(timezone.utc).isoformat()
            })

        return market_stats

    def score_wallet(
        self,
        wallet_stats: Dict[str, Any],
        tag_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute copy-trading score for a wallet."""
        roi = wallet_stats.get("roi", 0)
        win_rate = wallet_stats.get("win_rate", 0)
        total_volume = wallet_stats.get("total_volume", 0)
        last_trade_at = wallet_stats.get("last_trade_at")

        roi_score = max(0, min(1, (roi + 1) / 2))
        win_rate_score = win_rate
        volume_score = min(1, total_volume / 100000.0)

        recency_score = 0.5
        if last_trade_at:
            last_trade = datetime.fromisoformat(last_trade_at.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - last_trade).days
            recency_score = max(0, 1 - (days_ago / 90))

        roi_tag_score = max(0, min(1, (tag_stats.get("roi", 0) + 1) / 2)) if tag_stats else 0.0

        composite_score = (
            0.4 * roi_score +
            0.3 * win_rate_score +
            0.2 * (roi_tag_score or roi_score) +
            0.1 * recency_score
        )

        meets_thresholds = wallet_stats.get("is_eligible", False)
        tier = None
        if meets_thresholds:
            if composite_score >= 0.7:
                tier = "A"
            elif composite_score >= 0.5:
                tier = "B"
            else:
                tier = "C"

        return {
            "roi_score": roi_score,
            "win_rate_score": win_rate_score,
            "volume_score": volume_score,
            "recency_score": recency_score,
            "roi_tag_score": roi_tag_score,
            "composite_score": composite_score,
            "meets_thresholds": meets_thresholds,
            "tier": tier,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

    def build_events_lookup(self, events: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Build lookup dicts for events by ID and slug."""
        events_by_id, events_by_slug = {}, {}
        for event in events:
            if event_id := event.get("id"):
                events_by_id[event_id] = event
            if event_slug := event.get("slug"):
                events_by_slug[event_slug] = event
        return events_by_id, events_by_slug

    def populate_event_slug_cache(self, events_by_slug: Dict[str, Dict[str, Any]]) -> None:
        """Populate the internal event_slug → event_id cache."""
        for slug, event in events_by_slug.items():
            if event_id := event.get("id"):
                self._event_slug_to_id_cache[slug] = str(event_id)

    async def resolve_event_id_from_slug(self, event_slug: str) -> Optional[str]:
        """Resolve event_id from event_slug, using cache or fetching from API."""
        if not event_slug:
            return None

        if event_slug in self._event_slug_to_id_cache:
            return self._event_slug_to_id_cache[event_slug]

        event = await self.gamma.get_event(event_slug)
        if event_id := event.get("id"):
            self._event_slug_to_id_cache[event_slug] = str(event_id)
            return str(event_id)

        return None

    async def sync_wallet_closed_positions_with_enrichment(
        self,
        proxy_wallet: str,
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        event_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Sync closed positions with automatic event_id enrichment and early termination."""
        assert proxy_wallet, "proxy_wallet required"

        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)

        positions = []
        offset = 0
        seen_position_ids: Set[str] = set()  # Track seen positions for early termination
        POSITION_DUPLICATE_THRESHOLD = 1.0  # 100% duplicates = stop

        while True:
            logger.debug(f"API: get_closed_positions(user={proxy_wallet[:12]}..., offset={offset})")
            batch = await self.api.get_closed_positions(
                user=proxy_wallet, event_id=event_ids, limit=50, offset=offset
            )
            if not batch:
                break
            logger.debug(f"API: Received {len(batch)} positions")

            # Check for duplicates in this batch
            batch_duplicate_count = 0
            new_positions_in_batch = []
            
            for position in batch:
                normalized = self._normalize_closed_position(position, proxy_wallet)
                
                # Create position ID: proxy_wallet + condition_id + outcome_index
                condition_id = normalized.get("condition_id") or ""
                outcome_index = normalized.get("outcome_index", 0)
                position_id = f"{proxy_wallet}_{condition_id}_{outcome_index}"
                
                # Check if we've seen this position before
                if position_id in seen_position_ids:
                    batch_duplicate_count += 1
                else:
                    seen_position_ids.add(position_id)
                    new_positions_in_batch.append(normalized)

            # Early termination: if batch is 100% duplicates, stop fetching
            if len(batch) > 0 and batch_duplicate_count == len(batch):
                logger.info(f"Early termination: Batch at offset {offset} is 100% duplicates ({batch_duplicate_count}/{len(batch)}), stopping fetch")
                break

            # Process new positions
            for normalized in new_positions_in_batch:
                if event_slug := normalized.get("event_slug"):
                    event_id = await self.resolve_event_id_from_slug(event_slug)
                    if event_id:
                        normalized["event_id"] = event_id

                if save_position:
                    await save_position(normalized)
                positions.append(normalized)

            if len(batch) < 50:
                break
            offset += 50

        total_volume = sum(p.get("total_bought", 0) for p in positions)
        realized_pnl = sum(p.get("realized_pnl", 0) for p in positions)
        enriched_count = sum(1 for p in positions if p.get("event_id"))

        logger.info(f"Wallet {proxy_wallet[:12]}...: {len(positions)} positions, ${total_volume:,.2f} volume, ${realized_pnl:,.2f} PnL, {enriched_count} enriched")

        return {
            "wallet": proxy_wallet,
            "positions_fetched": len(positions),
            "positions_enriched": enriched_count,
            "total_volume": total_volume,
            "realized_pnl": realized_pnl
        }

    async def sync_wallets_closed_positions_batch(
        self,
        proxy_wallets: List[str],
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        max_concurrency: int = 5,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """Sync closed positions for multiple wallets concurrently."""
        logger.info(f"Batch sync: {len(proxy_wallets)} wallets, concurrency={max_concurrency}")
        if not proxy_wallets:
            return {
                "wallets_processed": 0,
                "wallets_with_positions": 0,
                "total_positions": 0,
                "total_enriched": 0,
                "total_volume": 0.0,
                "total_pnl": 0.0,
                "results": {}
            }

        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)
            logger.debug(f"Cache: Populated {len(events_by_slug)} event slugs")

        semaphore = asyncio.Semaphore(max_concurrency)
        total_wallets = len(proxy_wallets)

        async def process_wallet(wallet: str, idx: int) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                if progress_callback:
                    progress_callback(wallet, idx, total_wallets)

                result = await self.sync_wallet_closed_positions_with_enrichment(
                    proxy_wallet=wallet,
                    save_position=save_position,
                    events_by_slug=None
                )
                return wallet, result

        tasks = [process_wallet(wallet, idx) for idx, wallet in enumerate(proxy_wallets, 1)]
        results = await asyncio.gather(*tasks)

        aggregated = {
            "wallets_processed": 0,
            "wallets_with_positions": 0,
            "total_positions": 0,
            "total_enriched": 0,
            "total_volume": 0.0,
            "total_pnl": 0.0,
            "results": {}
        }

        for wallet, wallet_result in results:
            aggregated["wallets_processed"] += 1
            aggregated["results"][wallet] = wallet_result

            if wallet_result.get("positions_fetched", 0) > 0:
                aggregated["wallets_with_positions"] += 1
                aggregated["total_positions"] += wallet_result["positions_fetched"]
                aggregated["total_enriched"] += wallet_result.get("positions_enriched", 0)
                aggregated["total_volume"] += wallet_result.get("total_volume", 0.0)
                aggregated["total_pnl"] += wallet_result.get("realized_pnl", 0.0)

        logger.info(f"Batch complete: {aggregated['wallets_with_positions']}/{aggregated['wallets_processed']} wallets with positions, {aggregated['total_positions']} total positions")
        return aggregated

    def _normalize_wallet_from_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Extract wallet profile from trade data."""
        timestamp = trade.get("timestamp", 0)
        timestamp_iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if timestamp else None
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "proxy_wallet": trade.get("proxyWallet"),
            "name": trade.get("name") or "",
            "pseudonym": trade.get("pseudonym") or "",
            "bio": trade.get("bio") or "",
            "profile_image": trade.get("profileImage") or "",
            "profile_image_optimized": trade.get("profileImageOptimized") or "",
            "display_username_public": trade.get("displayUsernamePublic", False),
            "first_seen_at": timestamp_iso,
            "last_seen_at": timestamp_iso,
            "last_sync_at": now_iso,
            "total_trades": 0,
            "total_markets": 0,
            "total_volume": 0.0,
            "created_at": now_iso,
            "updated_at": now_iso,
            "raw_data": trade
        }

    def _normalize_trade(self, trade: Dict[str, Any], event_id: Optional[str] = None) -> Dict[str, Any]:
        """Normalize trade data.
        
        Uses API's native trade ID if available, otherwise generates from unique fields.
        This ensures DB records match API responses exactly.
        """
        # Use API's native ID if available (most reliable)
        trade_id = trade.get("id") or trade.get("tradeId")
        
        # Fallback: generate from unique fields if API doesn't provide ID
        if not trade_id:
            tx_hash = trade.get("transactionHash", "")
            asset = trade.get("asset", "")
            timestamp = trade.get("timestamp", 0)
            condition_id = trade.get("conditionId", "")
            outcome_index = trade.get("outcomeIndex", 0)
            # More comprehensive ID generation to ensure uniqueness
            trade_id = hashlib.sha256(
                f"{tx_hash}{asset}{condition_id}{outcome_index}{timestamp}".encode()
            ).hexdigest()[:32]
        
        size, price = trade.get("size", 0), trade.get("price", 0)
        return {
            "id": str(trade_id),  # Ensure string type
            "proxy_wallet": trade.get("proxyWallet"),
            "event_id": event_id,
            "condition_id": trade.get("conditionId"),
            "side": trade.get("side"),
            "asset": asset,
            "size": size,
            "price": price,
            "notional": size * price,
            "timestamp": timestamp,
            "transaction_hash": tx_hash,
            "title": trade.get("title"),
            "slug": trade.get("slug"),
            "event_slug": trade.get("eventSlug"),
            "outcome": trade.get("outcome"),
            "outcome_index": trade.get("outcomeIndex"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": trade
        }

    def _normalize_closed_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """Normalize closed position data."""
        condition_id = position.get("conditionId", "")
        outcome_index = position.get("outcomeIndex", 0)
        pos_id = hashlib.sha256(f"{proxy_wallet}{condition_id}{outcome_index}".encode()).hexdigest()[:32]
        return {
            "id": pos_id,
            "proxy_wallet": proxy_wallet,
            "event_id": None,
            "condition_id": condition_id,
            "asset": position.get("asset"),
            "outcome": position.get("outcome"),
            "outcome_index": outcome_index,
            "total_bought": position.get("totalBought", 0),
            "avg_price": position.get("avgPrice", 0),
            "cur_price": position.get("curPrice", 0),
            "realized_pnl": position.get("realizedPnl", 0),
            "timestamp": position.get("timestamp", 0),
            "end_date": position.get("endDate"),
            "title": position.get("title"),
            "slug": position.get("slug"),
            "event_slug": position.get("eventSlug"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": position
        }
