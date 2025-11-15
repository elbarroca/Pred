#!/usr/bin/env python3
"""
Reset Trader Tables Script

Clears all trader-related tables EXCEPT events and markets.
This allows testing the complete trader discovery pipeline with fresh data.

Usage:
    python scripts/reset_trader_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from database.client import MarketDatabase


def reset_trader_tables():
    """Reset all trader tables while preserving events and markets."""

    print("🗑️  Resetting Trader Tables")
    print("=" * 50)

    settings = Settings()
    db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    # Tables to reset (clear all data but keep structure)
    tables_to_reset = [
        "wallets",
        "trades",
        "wallet_closed_positions",
        "wallet_stats",
        # Add any other trader-related tables here
    ]

    print(f"📋 Tables to reset: {', '.join(tables_to_reset)}")
    print("💾 Preserving: events, markets")

    # Confirm before proceeding
    confirm = input("\n⚠️  This will DELETE ALL TRADER DATA. Continue? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("❌ Operation cancelled")
        return

    total_deleted = 0

    for table_name in tables_to_reset:
        try:
            print(f"\n🗑️  Clearing {table_name}...")

            # Get count before deletion
            count_result = db.supabase.table(table_name).select("*", count="exact").limit(1).execute()
            initial_count = count_result.count or 0
            print(f"   Found {initial_count:,} records")

            # Delete all records (use appropriate primary key column)
            if table_name == "wallets":
                delete_result = db.supabase.table(table_name).delete().neq("proxy_wallet", "nonexistent").execute()
            elif table_name == "wallet_stats":
                delete_result = db.supabase.table(table_name).delete().neq("proxy_wallet", "nonexistent").execute()
            else:
                delete_result = db.supabase.table(table_name).delete().neq("id", "nonexistent").execute()

            # Verify deletion
            verify_result = db.supabase.table(table_name).select("*", count="exact").limit(1).execute()
            final_count = verify_result.count or 0

            deleted_count = initial_count - final_count
            total_deleted += deleted_count

            if deleted_count > 0:
                print(f"   ✅ Deleted {deleted_count:,} records")
            else:
                print(f"   ℹ️  Table was already empty")

        except Exception as e:
            print(f"   ❌ Error clearing {table_name}: {e}")
            continue

    print("\n🎉 Reset Complete!")
    print(f"   Total records deleted: {total_deleted:,}")
    print("\n📝 Next steps:")
    print("   1. Run: python test_trader_flow_step_by_step.py")
    print("   2. Or run: python test_trader_flow_step_by_step.py --step 2")
    print("   3. Check that events table has proper categories/market counts")
    print("   4. Verify wallet eligibility filtering works")


if __name__ == "__main__":
    reset_trader_tables()
