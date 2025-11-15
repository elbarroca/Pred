#!/usr/bin/env python3
"""
Step-by-Step Trader Flow Validation

Complete flow validation:
1. Fetch events from database
2. Discover traders from those events
3. Fetch trader closed positions (historical P&L)
4. Filter traders by credibility (ROI, Win Rate, # trades)
5. Score traders with composite score
6. Backfill historical events from their trades
7. Save everything to database with validation

Usage:
    python test_trader_flow_step_by_step.py
    python test_trader_flow_step_by_step.py --step 3  # Start from step 3
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from database.client import MarketDatabase
from clients.wallet_tracker import WalletTracker, PolymarketDataAPI


async def run_flow_step_by_step(start_step: int = 1, skip_discovery: bool = False):
    """Execute the complete trader discovery and validation flow."""

    print("\n" + "=" * 80)
    print("TRADER FLOW - STEP BY STEP VALIDATION")
    print("=" * 80)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80 + "\n")

    settings = Settings()
    db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    api = PolymarketDataAPI()
    tracker = WalletTracker(api=api)

    # ========================================================================
    # FETCH EVENTS FROM DATABASE (needed for steps 2+)
    # ========================================================================
    events = []
    event_ids = []
    events_by_id = {}
    events_by_slug = {}

    if start_step <= 1 or start_step == 2:
        if start_step <= 1:
            print("STEP 1: FETCH EVENTS FROM DATABASE")
            print("-" * 80)

        events_result = db.supabase.table("events")\
            .select("id, title, tags, category, platform, raw_data")\
            .eq("platform", "polymarket")\
            .execute()

        events = []
        for event in events_result.data:
            event_data = {
                "id": event["id"],
                "title": event["title"],
                "tags": event["tags"],
                "category": event["category"],
                "platform": event["platform"]
            }
            # Extract slug from raw_data
            if event.get("raw_data"):
                try:
                    raw = json.loads(event["raw_data"]) if isinstance(event["raw_data"], str) else event["raw_data"]
                    event_data["slug"] = raw.get("slug", "")
                except:
                    event_data["slug"] = ""
            else:
                event_data["slug"] = ""
            events.append(event_data)

        if start_step <= 1:
            print(f"  Found {len(events)} Polymarket events\n")

            for i, event in enumerate(events[:5], 1):
                print(f"  {i}. {event['title'][:60]}...")
                print(f"     ID: {event['id']}, Category: {event['category']}")

            if len(events) > 5:
                print(f"  ... and {len(events) - 5} more events")

        event_ids = [e["id"] for e in events]
        events_by_id = {e["id"]: e for e in events}
        events_by_slug = {e["slug"]: e for e in events if e.get("slug")}

        if start_step <= 1:
            print(f"\n  STEP 1 COMPLETE")

    # ========================================================================
    # STEP 2: DISCOVER TRADERS FROM EVENTS
    # ========================================================================
    skip_step_2 = skip_discovery
    if start_step > 2 and not skip_discovery:
        # Check if wallets already exist, if so skip step 2
        try:
            wallets_check = db.supabase.table("wallets").select("*", count="exact").limit(1).execute()
            if wallets_check.count and wallets_check.count > 0:
                print(f"STEP 2: SKIPPED (found {wallets_check.count:,} existing wallets)")
                skip_step_2 = True
        except:
            pass

    if start_step <= 2 and not skip_step_2:
        print("\n\nSTEP 2: DISCOVER TRADERS FROM EVENTS")
        print("-" * 80)
        print("Fetching trades from events to discover wallets...")

        async def save_wallet(wallet_data):
            try:
                db.supabase.table("wallets").upsert(wallet_data).execute()
            except:
                pass

        async def save_trade(trade_data):
            try:
                db.supabase.table("trades").upsert(trade_data).execute()
            except:
                pass

        result = await tracker.discover_wallets_from_events(
            event_ids=event_ids,
            save_wallet=save_wallet,
            save_trade=save_trade,
            max_trades_per_event=1000  # More trades = more wallets
        )

        discovered_wallets = result["wallets"]
        print(f"\n  Wallets discovered: {len(discovered_wallets):,}")
        print(f"  Trades fetched: {result['trades_fetched']:,}")

        # Verify database
        db_wallets = db.supabase.table("wallets").select("*", count="exact").limit(1).execute()
        db_trades = db.supabase.table("trades").select("*", count="exact").limit(1).execute()

        print(f"\n  Database verification:")
        print(f"    wallets table: {db_wallets.count} records")
        print(f"    trades table: {db_trades.count} records")

        print(f"\n  STEP 2 COMPLETE")

    # ========================================================================
    # STEP 3: FETCH TRADER CLOSED POSITIONS (HISTORICAL P&L)
    # ========================================================================
    if start_step <= 3:
        print("\n\nSTEP 3: FETCH TRADER CLOSED POSITIONS")
        print("-" * 80)
        print("Fetching closed positions for each trader (their historical P&L)...")

        # Get wallets from database
        wallets_result = db.supabase.table("wallets").select("proxy_wallet").execute()
        all_wallets = [w["proxy_wallet"] for w in wallets_result.data]

        print(f"  Total wallets to process: {len(all_wallets)}")

        # Limit for testing (process all in production)
        wallets_to_process = all_wallets[:10]  # First 10 wallets for testing
        print(f"  Processing first {len(wallets_to_process)} wallets...\n")

        total_positions = 0
        wallets_with_positions = 0

        async def save_position(position_data):
            try:
                db.supabase.table("wallet_closed_positions").upsert(position_data).execute()
            except:
                pass

        for i, wallet in enumerate(wallets_to_process, 1):
            print(f"    [{i}/{len(wallets_to_process)}] Processing wallet: {wallet[:10]}...")

            try:
                result = await tracker.sync_wallet_closed_positions(
                    proxy_wallet=wallet,
                    save_position=save_position
                )

                positions_fetched = result["positions_fetched"]
                total_volume = result["total_volume"]
                realized_pnl = result["realized_pnl"]

                if positions_fetched > 0:
                    wallets_with_positions += 1
                    total_positions += positions_fetched
                    print(f"      ✅ Found {positions_fetched} positions | Volume: ${total_volume:,.2f} | PnL: ${realized_pnl:,.2f}")
                else:
                    print(f"      ⚠️  No positions found")

            except Exception as e:
                print(f"      ❌ Error: {str(e)[:100]}...")
                pass

        print(f"\n  Wallets with positions: {wallets_with_positions}")
        print(f"  Total positions fetched: {total_positions:,}")

        # Verify
        db_positions = db.supabase.table("wallet_closed_positions").select("*", count="exact").limit(1).execute()
        print(f"\n  Database verification:")
        print(f"    wallet_closed_positions: {db_positions.count:,} records")

        print(f"\n  STEP 3 COMPLETE")

    # ========================================================================
    # STEP 4: COMPUTE TRADER STATISTICS
    # ========================================================================
    if start_step <= 4:
        print("\n\nSTEP 4: COMPUTE TRADER STATISTICS")
        print("-" * 80)
        print("Computing ROI, Win Rate, Volume for each trader...")

        # Get all positions (paginate)
        all_positions = []
        offset = 0
        batch_size = 1000

        while True:
            result = db.supabase.table("wallet_closed_positions")\
                .select("proxy_wallet, total_bought, realized_pnl, event_slug, condition_id")\
                .range(offset, offset + batch_size - 1)\
                .execute()

            if not result.data:
                break

            all_positions.extend(result.data)
            offset += batch_size

            if len(result.data) < batch_size:
                break

        print(f"  Total positions to analyze: {len(all_positions):,}")

        # Compute stats per wallet
        wallet_stats = defaultdict(lambda: {
            "total_volume": 0.0,
            "realized_pnl": 0.0,
            "n_positions": 0,
            "n_wins": 0,
            "event_slugs": set(),
            "condition_ids": set()
        })

        for pos in all_positions:
            wallet = pos["proxy_wallet"]
            wallet_stats[wallet]["total_volume"] += pos.get("total_bought", 0)
            wallet_stats[wallet]["realized_pnl"] += pos.get("realized_pnl", 0)
            wallet_stats[wallet]["n_positions"] += 1
            if pos.get("realized_pnl", 0) > 0:
                wallet_stats[wallet]["n_wins"] += 1
            if pos.get("event_slug"):
                wallet_stats[wallet]["event_slugs"].add(pos["event_slug"])
            if pos.get("condition_id"):
                wallet_stats[wallet]["condition_ids"].add(pos["condition_id"])

        # Save stats to database
        print(f"  Saving stats for {len(wallet_stats)} wallets...")

        for wallet, stats in wallet_stats.items():
            n_pos = stats["n_positions"]
            if n_pos == 0:
                continue

            roi = stats["realized_pnl"] / stats["total_volume"] if stats["total_volume"] > 0 else 0
            win_rate = stats["n_wins"] / n_pos
            n_events = len(stats["event_slugs"])
            n_markets = len(stats["condition_ids"])

            is_eligible = (
                stats["total_volume"] >= 10000 and
                n_pos >= 20 and
                win_rate >= 0.60
            )

            wallet_stat = {
                "proxy_wallet": wallet,
                "total_volume": stats["total_volume"],
                "realized_pnl": stats["realized_pnl"],
                "roi": roi,
                "win_rate": win_rate,
                "n_positions": n_pos,
                "n_wins": stats["n_wins"],
                "n_losses": n_pos - stats["n_wins"],
                "n_markets": n_markets,
                "n_events": n_events,
                "is_eligible": is_eligible,
                "computed_at": datetime.now(timezone.utc).isoformat()
            }

            try:
                db.supabase.table("wallet_stats").upsert(wallet_stat).execute()
            except:
                pass

        # Count eligible
        eligible_result = db.supabase.table("wallet_stats")\
            .select("*", count="exact")\
            .eq("is_eligible", True)\
            .execute()

        print(f"\n  Eligible wallets (≥$10k, ≥20 pos, ≥60% WR): {eligible_result.count}")

        print(f"\n  STEP 4 COMPLETE")

    # ========================================================================
    # STEP 5: SCORE TRADERS (COMPOSITE SCORE)
    # ========================================================================
    if start_step <= 5:
        print("\n\nSTEP 5: SCORE TRADERS")
        print("-" * 80)
        print("Computing composite score: ROI (40%) + Win Rate (30%) + # Trades (30%)")

        # Get all wallet stats
        all_stats = []
        offset = 0
        batch_size = 1000

        while True:
            result = db.supabase.table("wallet_stats")\
                .select("proxy_wallet, total_volume, realized_pnl, roi, win_rate, n_positions, n_markets, is_eligible")\
                .range(offset, offset + batch_size - 1)\
                .execute()

            if not result.data:
                break

            all_stats.extend(result.data)
            offset += batch_size

            if len(result.data) < batch_size:
                break

        print(f"  Total wallets: {len(all_stats)}")

        # Compute normalized scores
        if all_stats:
            # Get max values for normalization
            max_roi = max(s["roi"] for s in all_stats) if all_stats else 1
            max_wr = max(s["win_rate"] for s in all_stats) if all_stats else 1
            max_trades = max(s["n_positions"] for s in all_stats) if all_stats else 1

            # Avoid division by zero
            if max_roi <= 0:
                max_roi = 1
            if max_wr <= 0:
                max_wr = 1
            if max_trades <= 0:
                max_trades = 1

            scored_traders = []
            for stat in all_stats:
                # Normalize components (0-1 scale)
                roi_score = stat["roi"] / max_roi if stat["roi"] > 0 else 0
                wr_score = stat["win_rate"] / max_wr
                trades_score = stat["n_positions"] / max_trades

                # Composite score (weighted)
                composite_score = (
                    0.4 * roi_score +
                    0.3 * wr_score +
                    0.3 * trades_score
                )

                scored_traders.append({
                    "wallet": stat["proxy_wallet"],
                    "total_volume": stat["total_volume"],
                    "roi": stat["roi"],
                    "win_rate": stat["win_rate"],
                    "n_positions": stat["n_positions"],
                    "n_markets": stat["n_markets"],
                    "composite_score": composite_score,
                    "is_eligible": stat["is_eligible"]
                })

            # Sort by composite score
            scored_traders.sort(key=lambda x: x["composite_score"], reverse=True)

            # Display top 10
            print(f"\n  TOP 10 TRADERS BY COMPOSITE SCORE:")
            print(f"  {'Rank':<6}{'Wallet':<15}{'Volume':<15}{'ROI':<10}{'WR':<10}{'Trades':<10}{'Score':<10}{'Eligible':<10}")
            print("  " + "-" * 95)

            for i, trader in enumerate(scored_traders[:10], 1):
                wallet_short = trader["wallet"][:12]
                volume = f"${trader['total_volume']:,.0f}"
                roi = f"{trader['roi']*100:.2f}%"
                wr = f"{trader['win_rate']*100:.1f}%"
                trades = str(trader["n_positions"])
                score = f"{trader['composite_score']:.4f}"
                eligible = "YES" if trader["is_eligible"] else "NO"

                print(f"  {i:<6}{wallet_short:<15}{volume:<15}{roi:<10}{wr:<10}{trades:<10}{score:<10}{eligible:<10}")

            # Save to wallet_scores table
            print(f"\n  Saving scores to database...")

            for trader in scored_traders:
                try:
                    score_data = {
                        "proxy_wallet": trader["wallet"],
                        "roi_score": trader["roi"],
                        "win_rate_score": trader["win_rate"],
                        "volume_score": trader["total_volume"],
                        "composite_score": trader["composite_score"],
                        "meets_thresholds": trader["is_eligible"],
                        "computed_at": datetime.now(timezone.utc).isoformat()
                    }
                    db.supabase.table("wallet_scores").upsert(score_data).execute()
                except:
                    pass

        print(f"\n  STEP 5 COMPLETE")

    # ========================================================================
    # STEP 6: BACKFILL HISTORICAL EVENTS
    # ========================================================================
    if start_step <= 6:
        print("\n\nSTEP 6: BACKFILL HISTORICAL EVENTS")
        print("-" * 80)
        print("Finding events from top traders' history and fetching from Polymarket API...")

        # Get top 5 traders
        top_traders = db.supabase.table("wallet_stats")\
            .select("proxy_wallet")\
            .order("roi", desc=True)\
            .limit(5)\
            .execute()

        top_wallet_addresses = [t["proxy_wallet"] for t in top_traders.data]
        print(f"  Top 5 traders: {[w[:12] for w in top_wallet_addresses]}")

        # Get all event slugs from their positions
        all_event_slugs = set()
        for wallet in top_wallet_addresses:
            positions = db.supabase.table("wallet_closed_positions")\
                .select("event_slug")\
                .eq("proxy_wallet", wallet)\
                .execute()

            for pos in positions.data:
                if pos.get("event_slug"):
                    all_event_slugs.add(pos["event_slug"])

        print(f"  Unique events from top traders: {len(all_event_slugs)}")

        # Get existing events in database
        existing_events = db.supabase.table("events").select("raw_data").execute()
        existing_slugs = set()
        for event in existing_events.data:
            if event.get("raw_data"):
                try:
                    raw = json.loads(event["raw_data"]) if isinstance(event["raw_data"], str) else event["raw_data"]
                    if raw.get("slug"):
                        existing_slugs.add(raw["slug"])
                except:
                    pass

        print(f"  Events already in database: {len(existing_slugs)}")

        # Find missing events
        missing_slugs = all_event_slugs - existing_slugs
        print(f"  MISSING events to fetch: {len(missing_slugs)}")

        if missing_slugs:
            print(f"\n  Fetching missing events from Polymarket API...")

            # Limit to first 50 for testing
            slugs_to_fetch = list(missing_slugs)[:50]
            print(f"  Fetching first {len(slugs_to_fetch)} events...\n")

            fetched_count = 0
            async with httpx.AsyncClient(timeout=30.0) as client:
                for slug in slugs_to_fetch:
                    try:
                        url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
                        response = await client.get(url)

                        if response.status_code == 200:
                            event_data = response.json()

                            # Save to events table
                            new_event = {
                                "id": str(event_data.get("id")),
                                "title": event_data.get("title", ""),
                                "description": event_data.get("description", "")[:1000] if event_data.get("description") else "",
                                "platform": "polymarket",
                                "category": event_data.get("category", ""),
                                "tags": json.dumps(event_data.get("tags", [])),
                                "status": "resolved" if event_data.get("closed") else "active",
                                "start_date": event_data.get("startDate"),
                                "end_date": event_data.get("endDate"),
                                "raw_data": json.dumps(event_data),
                                "created_at": datetime.now(timezone.utc).isoformat(),
                                "updated_at": datetime.now(timezone.utc).isoformat()
                            }

                            db.supabase.table("events").upsert(new_event).execute()
                            fetched_count += 1

                            if fetched_count <= 5:
                                print(f"    Added: {event_data.get('title', 'Unknown')[:60]}")

                        await asyncio.sleep(0.1)  # Rate limiting

                    except Exception as e:
                        pass

            print(f"\n  Successfully added {fetched_count} new events to database")

        # Verify
        total_events = db.supabase.table("events").select("*", count="exact").execute()
        print(f"\n  Database verification:")
        print(f"    events table: {total_events.count} records")

        print(f"\n  STEP 6 COMPLETE")

    # ========================================================================
    # STEP 7: FINAL VALIDATION
    # ========================================================================
    print("\n\nSTEP 7: FINAL VALIDATION")
    print("-" * 80)

    tables = [
        ("events", "Event catalog"),
        ("markets", "Market catalog"),
        ("wallets", "Discovered traders"),
        ("trades", "Trades from events"),
        ("wallet_closed_positions", "Historical P&L"),
        ("wallet_stats", "Trader statistics"),
        ("wallet_scores", "Trader scores"),
    ]

    print("DATABASE STATUS:")
    for table, desc in tables:
        try:
            count = db.supabase.table(table).select("*", count="exact").limit(1).execute()
            print(f"  {table:30} {count.count:>10,} records  ({desc})")
        except Exception as e:
            print(f"  {table:30} Error: {str(e)[:30]}")

    # Show eligible traders
    print("\nELIGIBLE TRADERS SUMMARY:")
    eligible = db.supabase.table("wallet_stats")\
        .select("proxy_wallet, total_volume, roi, win_rate, n_positions")\
        .eq("is_eligible", True)\
        .order("roi", desc=True)\
        .limit(5)\
        .execute()

    if eligible.data:
        print(f"  Found {len(eligible.data)} eligible traders (showing top 5):")
        for i, trader in enumerate(eligible.data, 1):
            print(f"    {i}. {trader['proxy_wallet'][:12]}... "
                  f"Vol=${trader['total_volume']:,.0f} "
                  f"ROI={trader['roi']*100:.2f}% "
                  f"WR={trader['win_rate']*100:.1f}% "
                  f"Trades={trader['n_positions']}")
    else:
        print("  No eligible traders found yet.")

    print("\n" + "=" * 80)
    print("FLOW VALIDATION COMPLETE!")
    print("=" * 80)
    print(f"Completed at: {datetime.now(timezone.utc).isoformat()}")
    print("\nYour trader analytics system is now populated with:")
    print("  - Traders discovered from your events")
    print("  - Their historical P&L from Polymarket")
    print("  - Statistics (ROI, Win Rate, Volume)")
    print("  - Composite scores for ranking")
    print("  - Historical events they've traded")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Step-by-step trader flow validation")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Start from this step (default: 1)"
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip step 2 (wallet discovery) if wallets already exist"
    )
    args = parser.parse_args()
    asyncio.run(run_flow_step_by_step(start_step=args.step, skip_discovery=args.skip_discovery))


if __name__ == "__main__":
    main()
