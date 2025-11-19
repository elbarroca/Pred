#!/usr/bin/env python3
"""Asynchronously enrich all wallets in batches of 8-10."""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from clients.functions.wallets import PolymarketWalletCollector
from database.client import MarketDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_wallet_enrichment")


class BatchWalletEnricher:
    """Batch enrich wallets asynchronously."""

    def __init__(self, batch_size: int = 50):  # Increased for better throughput with 63k wallets
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.collector = PolymarketWalletCollector()
        self.batch_size = batch_size
        self.checkpoint_file = Path(__file__).parent / "batch_enrichment_checkpoint.json"

    def _save_checkpoint(self, processed_wallets: int, successful_wallets: int, last_wallet_batch: List[str]):
        """Save progress checkpoint for crash recovery."""
        checkpoint = {
            "processed_wallets": processed_wallets,
            "successful_wallets": successful_wallets,
            "last_wallet_batch": last_wallet_batch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_size": self.batch_size
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"💾 Checkpoint saved: {processed_wallets} processed, {successful_wallets} successful")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> Dict:
        """Load progress checkpoint for crash recovery."""
        if not self.checkpoint_file.exists():
            return {}

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"📂 Checkpoint loaded: {checkpoint.get('processed_wallets', 0)} processed, {checkpoint.get('successful_wallets', 0)} successful")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return {}

    def _clear_checkpoint(self):
        """Clear checkpoint file after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("🗑️  Checkpoint cleared - process completed successfully")
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")

    async def get_unenriched_wallets(self) -> List[str]:
        """Get all unenriched wallets using pagination."""
        wallets = []
        page_size = 2000  # Larger batch size for efficiency
        offset = 0

        while True:
            result = self.db.supabase.table("wallets").select("proxy_wallet").eq("enriched", False).range(offset, offset + page_size - 1).execute()
            page_wallets = result.data or []

            if not page_wallets:
                break

            wallets.extend([row["proxy_wallet"] for row in page_wallets])
            offset += page_size

            # Safety break if too many pages (shouldn't happen normally)
            if offset > 100000:  # Emergency break at 100k
                logger.warning("Emergency break: too many wallets, stopping at offset 100k")
                break

        logger.info(f"Retrieved {len(wallets)} unenriched wallets using pagination")
        return wallets

    async def enrich_wallet_batch(self, wallet_batch: List[str]) -> Dict[str, int]:
        """Enrich a batch of wallets concurrently."""
        logger.info(f"Processing batch of {len(wallet_batch)} wallets")

        async def enrich_single_wallet(wallet_addr: str) -> Dict[str, any]:
            """Enrich single wallet and return result."""
            try:
                # Get timestamp for this wallet
                timestamps = await self.collector._get_wallet_last_sync_timestamps([wallet_addr])
                last_ts = timestamps.get(wallet_addr, 0)

                # Create event batch saver
                async def save_event_batch_filtered(events: List[Dict]):
                    new_events = [e for e in events if e.get("id") not in self.collector._saved_events_cache]
                    if not new_events: return
                    await self.collector._bulk_upsert("events_closed", new_events)
                    for event in new_events:
                        if event.get("id"):
                            self.collector._saved_events_cache.add(event["id"])

                # Track sample IDs for data integrity verification
                saved_position_ids = []
                saved_event_ids = []

                async def verified_position_batch(positions):
                    await self.collector._bulk_upsert("wallet_closed_positions", positions)
                    # Collect sample IDs for verification
                    for pos in positions[:5]:  # Sample first 5 positions
                        if pos.get("id"):
                            saved_position_ids.append(pos["id"])

                async def verified_event_batch(events):
                    await self.collector._bulk_upsert("events_closed", events)
                    # Collect sample IDs for verification
                    for event in events[:3]:  # Sample first 3 events
                        if event.get("id"):
                            saved_event_ids.append(event["id"])

                # Enrich wallet with verified saving
                result = await self.collector.wallet_tracker.sync_wallet_closed_positions_with_enrichment(
                    proxy_wallet=wallet_addr,
                    save_position_batch=verified_position_batch,
                    save_event_batch=verified_event_batch,
                    last_synced_timestamp=last_ts
                )

                # Verify data integrity
                positions_integrity = await self.collector._verify_data_integrity(
                    "wallet_closed_positions", result.get("positions_fetched", 0), saved_position_ids
                )
                events_integrity = await self.collector._verify_data_integrity(
                    "events_closed", len(saved_event_ids), saved_event_ids
                )

                if not positions_integrity or not events_integrity:
                    logger.error(f"❌ DATA INTEGRITY FAILURE for wallet {wallet_addr[:10]}... - will retry next run")
                    return {"wallet": wallet_addr, "success": False, "error": "Data integrity check failed"}

                # Discover additional wallets
                event_ids = result.get("event_ids", [])
                additional_discovered = 0
                if event_ids:
                    discovered = await self.collector._discover_wallets_from_events_during_enrichment(event_ids)
                    additional_discovered = len(discovered)

                # Mark as enriched
                await self.collector._mark_wallet_enriched_single(wallet_addr)

                logger.info(f"✅ {wallet_addr[:10]}...: {result.get('positions_fetched', 0)} positions, ${result.get('total_volume', 0):,.0f} vol, +{additional_discovered} wallets")
                return {"wallet": wallet_addr, "success": True, "result": result, "additional_discovered": additional_discovered}

            except Exception as e:
                logger.error(f"❌ Failed {wallet_addr[:10]}...: {e}")
                # DO NOT mark as enriched - allow retry on next run with improved error handling
                # Only mark as enriched if the failure is due to permanent issues (like invalid wallet)
                if "invalid" in str(e).lower() or "not found" in str(e).lower():
                    try:
                        await self.collector._mark_wallet_enriched_single(wallet_addr)
                        logger.warning(f"⚠️  Marked permanently failed wallet {wallet_addr[:10]}... as enriched")
                    except Exception as mark_e:
                        logger.error(f"Failed to mark permanently failed wallet {wallet_addr[:10]}...: {mark_e}")
                return {"wallet": wallet_addr, "success": False, "error": str(e)}

        # Process batch concurrently
        batch_results = await asyncio.gather(*[enrich_single_wallet(wallet) for wallet in wallet_batch])

        # Aggregate results
        successful = sum(1 for r in batch_results if r["success"])
        total_positions = sum(r.get("result", {}).get("positions_fetched", 0) for r in batch_results if r["success"])
        total_volume = sum(r.get("result", {}).get("total_volume", 0) for r in batch_results if r["success"])
        total_discovered = sum(r.get("additional_discovered", 0) for r in batch_results)

        return {
            "batch_size": len(wallet_batch),
            "successful": successful,
            "failed": len(wallet_batch) - successful,
            "total_positions": total_positions,
            "total_volume": total_volume,
            "additional_wallets_discovered": total_discovered
        }

    async def enrich_all_wallets(self) -> Dict[str, any]:
        """Enrich all unenriched wallets in batches with checkpoint recovery."""
        start_time = time.time()

        # Load checkpoint for crash recovery
        checkpoint = self._load_checkpoint()
        total_processed = checkpoint.get("processed_wallets", 0)
        total_successful = checkpoint.get("successful_wallets", 0)
        last_processed_batch = checkpoint.get("last_wallet_batch", [])

        total_positions = 0
        total_volume = 0.0
        total_discovered = 0

        logger.info("Starting batch wallet enrichment process")
        if checkpoint:
            logger.info(f"🔄 Resuming from checkpoint: {total_processed} already processed, {total_successful} successful")
            # Skip wallets that were already processed in the last batch
            if last_processed_batch:
                logger.info(f"⏭️  Skipping last incomplete batch: {len(last_processed_batch)} wallets")

        while True:
            # Get current unenriched wallets
            unenriched_wallets = await self.get_unenriched_wallets()

            # If resuming from checkpoint, skip wallets already processed
            if checkpoint and last_processed_batch:
                # Remove wallets that were in the last incomplete batch
                unenriched_wallets = [w for w in unenriched_wallets if w not in last_processed_batch]
                last_processed_batch = []  # Clear after first use

            if not unenriched_wallets:
                logger.info("No more unenriched wallets found")
                break

            remaining = len(unenriched_wallets)
            logger.info(f"Found {remaining} unenriched wallets, processing next batch of {min(self.batch_size, remaining)}")

            # Process next batch
            current_batch = unenriched_wallets[:self.batch_size]
            batch_result = await self.enrich_wallet_batch(current_batch)

            # Save checkpoint after each batch
            self._save_checkpoint(
                total_processed + batch_result["batch_size"],
                total_successful + batch_result["successful"],
                current_batch
            )

            # Update totals
            total_processed += batch_result["batch_size"]
            total_successful += batch_result["successful"]
            total_positions += batch_result["total_positions"]
            total_volume += batch_result["total_volume"]
            total_discovered += batch_result["additional_wallets_discovered"]

            # Progress logging
            elapsed = time.time() - start_time
            rate_per_min = (total_processed / elapsed * 60) if elapsed > 0 else 0
            eta_hours = remaining / (total_processed / elapsed) / 3600 if elapsed > 0 and total_processed > 0 else 0

            success_rate = total_successful / total_processed if total_processed > 0 else 0

            logger.info(f"Progress: {total_processed} processed, {total_successful} successful ({success_rate:.1%}), {remaining} remaining")
            logger.info(f"Stats: {total_positions} positions, ${total_volume:,.0f} volume, +{total_discovered} wallets discovered")
            logger.info(f"Rate: {rate_per_min:.1f} wallets/min, ETA: {eta_hours:.1f} hours")

            # Checkpoint every 1000 wallets
            if total_processed % 1000 == 0 and total_processed > 0:
                logger.info(f"🎯 CHECKPOINT: {total_processed} wallets processed - progress saved")

            # Small delay between batches
            await asyncio.sleep(1)

        elapsed_total = time.time() - start_time

        # Clear checkpoint on successful completion
        self._clear_checkpoint()

        return {
            "total_processed": total_processed,
            "total_successful": total_successful,
            "total_failed": total_processed - total_successful,
            "total_positions": total_positions,
            "total_volume": total_volume,
            "total_discovered_wallets": total_discovered,
            "elapsed_seconds": elapsed_total,
            "elapsed_hours": elapsed_total / 3600,
            "success_rate": total_successful / total_processed if total_processed > 0 else 0
        }


async def main():
    """Run batch wallet enrichment."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch wallet enrichment with crash recovery")
    parser.add_argument("--restart", action="store_true", help="Force restart from beginning (ignore checkpoint)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing (default: 50)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BATCH WALLET ENRICHMENT PROCESS")
    print("=" * 60)
    print("Asynchronously enriching all wallets in batches")
    print("Features: Crash recovery, data integrity checks, retry logic")
    print("Designed for large-scale processing (63k+ wallets)")
    print("This process may take several hours to complete")
    if args.restart:
        print("🔄 FORCE RESTART: Ignoring any existing checkpoint")
    print("=" * 60)

    enricher = BatchWalletEnricher(batch_size=args.batch_size)

    # Handle force restart
    if args.restart and enricher.checkpoint_file.exists():
        enricher._clear_checkpoint()
        print("🗑️  Cleared existing checkpoint for fresh start")

    result = await enricher.enrich_all_wallets()

    print("\n📊 Final Results:")
    print(f"   Total processed: {result['total_processed']}")
    print(f"   Successful: {result['total_successful']}")
    print(f"   Failed: {result['total_failed']}")
    print(f"   Success rate: {result['success_rate']:.1%}")
    print(f"   Total positions saved: {result['total_positions']}")
    print(f"   Total volume: ${result['total_volume']:,.2f}")
    print(f"   Additional wallets discovered: {result['total_discovered_wallets']}")
    print(f"   Elapsed time: {result['elapsed_hours']:.2f} hours")
    print(f"   Average rate: {result['total_processed'] / max(result['elapsed_hours'], 0.001):.1f} wallets/hour")
    print("\n" + "=" * 60)

    if result["total_successful"] > 0:
        print("✅ Batch enrichment completed successfully")
        exit(0)
    else:
        print("⚠️  No wallets were successfully enriched")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
