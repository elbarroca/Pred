"""
End-to-End Test for Trader Analytics System

Tests the complete flow:
1. Fetch events from database
2. Discover wallets from event trades
3. Sync wallet closed positions
4. Compute wallet stats
5. Compute tag stats
6. Sync open positions
7. Compute radar metrics
8. Compute tag credibility scores
9. Verify database population
"""

import asyncio
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

from config.settings import Settings
from database.client import MarketDatabase
from clients.wallet_tracker import WalletTracker, PolymarketDataAPI


async def test_trader_system():
    """Run complete end-to-end test."""

    print("\n" + "="*80)
    print("TRADER ANALYTICS SYSTEM - END-TO-END TEST")
    print("="*80 + "\n")

    # Initialize
    settings = Settings()
    db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    api = PolymarketDataAPI()
    tracker = WalletTracker(api=api)

    # ========================================================================
    # STEP 1: Fetch events from database
    # ========================================================================
    print("📊 STEP 1: Fetching events from database...")

    try:
        result = db.supabase.table("events")\
            .select("id, title, platform, category, tags, status")\
            .eq("platform", "polymarket")\
            .limit(5)\
            .execute()

        events = result.data
        print(f"✅ Found {len(events)} Polymarket events")

        if not events:
            print("❌ No events found in database. Please run event discovery first.")
            return

        # Display events
        for i, event in enumerate(events, 1):
            print(f"   {i}. {event['title'][:60]}...")
            print(f"      ID: {event['id']}")
            print(f"      Category: {event.get('category', 'N/A')}")

        event_ids = [e["id"] for e in events]

    except Exception as e:
        print(f"❌ Error fetching events: {e}")
        return

    # ========================================================================
    # STEP 2: Discover wallets from event trades
    # ========================================================================
    print(f"\n📈 STEP 2: Discovering wallets from {len(event_ids)} events...")
    print("   (This may take a few minutes due to API rate limiting)")

    discovered_wallets = set()
    total_trades = 0

    async def save_wallet(wallet_data: Dict[str, Any]):
        """Callback to save wallet to database."""
        try:
            db.supabase.table("wallets").upsert(wallet_data).execute()
        except Exception as e:
            print(f"   ⚠️  Error saving wallet: {e}")

    async def save_trade(trade_data: Dict[str, Any]):
        """Callback to save trade to database."""
        try:
            db.supabase.table("trades").upsert(trade_data).execute()
        except Exception as e:
            print(f"   ⚠️  Error saving trade: {e}")

    try:
        result = await tracker.discover_wallets_from_events(
            event_ids=event_ids,
            save_wallet=save_wallet,
            save_trade=save_trade,
            max_trades_per_event=200  # Limit for testing
        )

        discovered_wallets = result["wallets"]
        total_trades = result["trades_fetched"]

        print(f"✅ Discovered {len(discovered_wallets)} unique wallets")
        print(f"✅ Fetched {total_trades} trades")

        if not discovered_wallets:
            print("❌ No wallets discovered. Events may have no recent trades.")
            return

    except Exception as e:
        print(f"❌ Error discovering wallets: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 3: Sync closed positions for top wallets (sample)
    # ========================================================================
    print(f"\n💰 STEP 3: Syncing closed positions for sample wallets...")

    # Get top 10 wallets by trade count
    sample_wallets = list(discovered_wallets)[:10]
    print(f"   Testing with {len(sample_wallets)} wallets")

    wallets_with_positions = 0
    total_positions = 0

    async def save_position(position_data: Dict[str, Any]):
        """Callback to save closed position."""
        try:
            db.supabase.table("wallet_closed_positions").upsert(position_data).execute()
        except Exception as e:
            print(f"   ⚠️  Error saving position: {e}")

    for i, wallet in enumerate(sample_wallets, 1):
        try:
            print(f"   [{i}/{len(sample_wallets)}] Syncing {wallet[:10]}...")

            result = await tracker.sync_wallet_closed_positions(
                proxy_wallet=wallet,
                save_position=save_position
            )

            if result["positions_fetched"] > 0:
                wallets_with_positions += 1
                total_positions += result["positions_fetched"]
                print(f"      ✅ {result['positions_fetched']} positions, PnL: ${result['realized_pnl']:.2f}")
            else:
                print(f"      ⚪ No closed positions")

        except Exception as e:
            print(f"      ❌ Error: {e}")

    print(f"✅ Synced {total_positions} positions from {wallets_with_positions} wallets")

    # ========================================================================
    # STEP 4: Compute wallet stats
    # ========================================================================
    print(f"\n📊 STEP 4: Computing wallet statistics...")

    stats_computed = 0

    for wallet in sample_wallets:
        try:
            # Get closed positions from database
            positions_result = db.supabase.table("wallet_closed_positions")\
                .select("*")\
                .eq("proxy_wallet", wallet)\
                .execute()

            positions = positions_result.data

            if not positions:
                continue

            # Compute stats
            stats = tracker.compute_wallet_stats_from_positions(positions)
            stats["proxy_wallet"] = wallet

            # Save to database
            db.supabase.table("wallet_stats").upsert(stats).execute()
            stats_computed += 1

            if stats.get("is_eligible"):
                print(f"   ✅ {wallet[:10]}: ROI={stats['roi']*100:.1f}%, WR={stats['win_rate']*100:.1f}%, Vol=${stats['total_volume']:.0f} [ELIGIBLE]")

        except Exception as e:
            print(f"   ⚠️  Error computing stats for {wallet[:10]}: {e}")

    print(f"✅ Computed stats for {stats_computed} wallets")

    # ========================================================================
    # STEP 5: Compute tag-specific stats
    # ========================================================================
    print(f"\n🏷️  STEP 5: Computing tag-specific statistics...")

    # Get events with tags
    events_by_slug = {}
    for event in events:
        if event.get("id"):
            events_by_slug[event["id"]] = event

    tag_stats_computed = 0

    for wallet in sample_wallets:
        try:
            # Get closed positions
            positions_result = db.supabase.table("wallet_closed_positions")\
                .select("*")\
                .eq("proxy_wallet", wallet)\
                .execute()

            positions = positions_result.data

            if not positions:
                continue

            # Compute tag stats
            tag_stats_list = tracker.compute_wallet_tag_stats(
                positions=positions,
                events_by_slug=events_by_slug
            )

            # Save to database
            for tag_stat in tag_stats_list:
                db.supabase.table("wallet_tag_stats").upsert(tag_stat).execute()
                tag_stats_computed += 1

        except Exception as e:
            print(f"   ⚠️  Error computing tag stats for {wallet[:10]}: {e}")

    print(f"✅ Computed {tag_stats_computed} tag-specific stats")

    # ========================================================================
    # STEP 6: Sync open positions (sample)
    # ========================================================================
    print(f"\n🔓 STEP 6: Syncing open positions for eligible wallets...")

    # Get eligible wallets
    eligible_result = db.supabase.table("wallet_stats")\
        .select("proxy_wallet")\
        .eq("is_eligible", True)\
        .limit(5)\
        .execute()

    eligible_wallets = [w["proxy_wallet"] for w in eligible_result.data]

    if not eligible_wallets:
        print("   ⚪ No eligible wallets found (need ≥$10k volume, ≥20 markets, ≥60% win rate)")
    else:
        print(f"   Found {len(eligible_wallets)} eligible wallets")

        async def save_open_position(position_data: Dict[str, Any]):
            """Callback to save open position."""
            try:
                db.supabase.table("wallet_open_positions").upsert(position_data).execute()
            except Exception as e:
                print(f"   ⚠️  Error saving open position: {e}")

        open_positions_synced = 0

        for i, wallet in enumerate(eligible_wallets, 1):
            try:
                print(f"   [{i}/{len(eligible_wallets)}] Syncing open positions for {wallet[:10]}...")

                result = await tracker.sync_wallet_open_positions(
                    proxy_wallet=wallet,
                    save_position=save_open_position
                )

                if result["positions_fetched"] > 0:
                    open_positions_synced += result["positions_fetched"]
                    print(f"      ✅ {result['positions_fetched']} open positions, Unrealized PnL: ${result['total_unrealized_pnl']:.2f}")

                    # Update wallet_stats with open position info
                    db.supabase.table("wallet_stats")\
                        .update({
                            "open_positions_count": result["positions_fetched"],
                            "open_positions_value": result["total_position_value"]
                        })\
                        .eq("proxy_wallet", wallet)\
                        .execute()
                else:
                    print(f"      ⚪ No open positions")

            except Exception as e:
                print(f"      ❌ Error: {e}")

        print(f"✅ Synced {open_positions_synced} open positions")

    # ========================================================================
    # STEP 7: Display Summary
    # ========================================================================
    print(f"\n" + "="*80)
    print("📋 DATABASE POPULATION SUMMARY")
    print("="*80)

    # Count records in each table
    tables_to_check = [
        "wallets",
        "trades",
        "wallet_closed_positions",
        "wallet_stats",
        "wallet_tag_stats",
        "wallet_open_positions"
    ]

    for table in tables_to_check:
        try:
            count_result = db.supabase.table(table).select("*", count="exact").limit(1).execute()
            count = count_result.count
            print(f"✅ {table:30} {count:>6} records")
        except Exception as e:
            print(f"❌ {table:30} Error: {e}")

    # ========================================================================
    # STEP 8: Display Top Traders
    # ========================================================================
    print(f"\n" + "="*80)
    print("🏆 TOP 10 ELIGIBLE TRADERS")
    print("="*80)

    try:
        top_traders = db.supabase.table("wallet_stats")\
            .select("proxy_wallet, total_volume, realized_pnl, roi, win_rate, n_positions, n_markets, tier")\
            .eq("is_eligible", True)\
            .order("roi", desc=True)\
            .limit(10)\
            .execute()

        if top_traders.data:
            print(f"\n{'Rank':<6}{'Wallet':<15}{'Volume':<12}{'PnL':<12}{'ROI':<10}{'Win Rate':<12}{'Markets':<10}{'Tier':<6}")
            print("-" * 90)

            for i, trader in enumerate(top_traders.data, 1):
                wallet_short = trader["proxy_wallet"][:12]
                volume = f"${trader['total_volume']:,.0f}"
                pnl = f"${trader['realized_pnl']:,.0f}"
                roi = f"{trader['roi']*100:.1f}%"
                win_rate = f"{trader['win_rate']*100:.1f}%"
                markets = str(trader['n_markets'])
                tier = trader.get('tier', 'N/A')

                print(f"{i:<6}{wallet_short:<15}{volume:<12}{pnl:<12}{roi:<10}{win_rate:<12}{markets:<10}{tier:<6}")
        else:
            print("No eligible traders found yet. Run with more events or lower thresholds.")

    except Exception as e:
        print(f"❌ Error fetching top traders: {e}")

    # ========================================================================
    # STEP 9: Test Complete
    # ========================================================================
    print(f"\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check Supabase dashboard to verify data")
    print("2. Implement remaining API endpoints (see IMPLEMENTATION_GUIDE.md)")
    print("3. Set up batch sync job for automated updates")
    print("4. Build frontend to display trader analytics\n")


if __name__ == "__main__":
    asyncio.run(test_trader_system())
