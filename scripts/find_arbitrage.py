#!/usr/bin/env python3
"""
Quick Arbitrage Finder
Scan active markets for arbitrage opportunities (both mispricing and cross-platform)
"""
import asyncio
import sys
from typing import List, Dict, Any
from arbee.agents.arbitrage import ArbitrageDetector
from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient
from config.settings import settings


async def scan_for_arbitrage(
    limit: int = 20,
    bankroll: float = settings.DEFAULT_BANKROLL,
    min_profit_threshold: float = 0.005,  # 0.5% minimum profit
    category_filter: str = None
) -> List[Dict[str, Any]]:
    """
    Scan active markets for arbitrage opportunities

    Args:
        limit: Max markets to check from each platform
        bankroll: Available capital
        min_profit_threshold: Minimum profit (0.01 = 1%)
        category_filter: Optional keyword filter

    Returns:
        List of arbitrage opportunities found
    """
    print("="*80)
    print("POLYSEER ARBITRAGE SCANNER")
    print("="*80)
    print(f"\nScanning up to {limit} markets per platform for arbitrage...")
    print(f"Bankroll: ${bankroll:,.2f}")
    print(f"Minimum profit threshold: {min_profit_threshold:.1%}\n")

    # Initialize clients
    pm_client = PolymarketClient()
    kalshi_client = KalshiClient()
    arb_detector = ArbitrageDetector()

    # Fetch markets from both platforms
    print("[1/3] Fetching markets from Polymarket...")
    pm_markets = await pm_client.gamma.get_markets(active=True, limit=limit)

    print(f"[2/3] Fetching markets from Kalshi...")
    kalshi_markets = await kalshi_client.get_markets(status="open", limit=limit)

    if category_filter:
        filter_lower = category_filter.lower()
        pm_markets = [
            m for m in pm_markets
            if filter_lower in m.get('question', '').lower()
        ]
        kalshi_markets = [
            m for m in kalshi_markets
            if filter_lower in m.get('title', '').lower()
        ]

    print(f"✓ Found {len(pm_markets)} Polymarket markets, {len(kalshi_markets)} Kalshi markets")

    # Check each Polymarket market for cross-platform arbitrage
    print(f"\n[3/3] Checking for arbitrage opportunities...\n")

    all_opportunities = []

    for i, pm_market in enumerate(pm_markets[:10]):  # Limit to first 10 for speed
        market_slug = pm_market.get('slug', pm_market.get('id'))
        question = pm_market.get('question', '')

        print(f"  [{i+1}/{min(10, len(pm_markets))}] Checking: {question[:60]}...")

        try:
            # Check for cross-platform arbitrage only (no Bayesian estimate needed)
            opportunities = await arb_detector.detect_cross_platform_arbitrage(
                market_slug=market_slug,
                market_question=question,
                providers=["polymarket", "kalshi"],
                bankroll=bankroll,
                max_kelly=settings.MAX_KELLY_FRACTION
            )

            # Filter by minimum profit
            profitable_opps = [
                opp for opp in opportunities
                if (opp.guaranteed_profit or 0) >= min_profit_threshold
            ]

            if profitable_opps:
                print(f"      🎰 FOUND {len(profitable_opps)} OPPORTUNITY(IES)!")
                for opp in profitable_opps:
                    print(f"         Profit: {opp.guaranteed_profit*100:.2f}% ({opp.platform_pair})")
                    all_opportunities.append({
                        'market': question,
                        'slug': market_slug,
                        'opportunity': opp
                    })

        except Exception as e:
            print(f"      ⚠ Error: {e}")
            continue

    return all_opportunities


async def display_arbitrage_results(opportunities: List[Dict[str, Any]]):
    """Display found arbitrage opportunities"""
    print("\n" + "="*80)
    print("ARBITRAGE RESULTS")
    print("="*80)

    if not opportunities:
        print("\n❌ No arbitrage opportunities found above threshold.")
        print("\nReasons:")
        print("  • Markets may be efficiently priced")
        print("  • Transaction fees eat into potential profits")
        print("  • Liquidity constraints")
        print("\nTry:")
        print("  • Lower the profit threshold")
        print("  • Check more markets (increase --limit)")
        print("  • Focus on less liquid markets\n")
        return

    print(f"\n✅ Found {len(opportunities)} arbitrage opportunit{'y' if len(opportunities) == 1 else 'ies'}!\n")

    # Sort by profit
    opportunities.sort(
        key=lambda x: x['opportunity'].guaranteed_profit or 0,
        reverse=True
    )

    for i, opp_data in enumerate(opportunities, 1):
        opp = opp_data['opportunity']

        print(f"\n[{i}] {opp_data['market'][:70]}...")
        print(f"    Slug: {opp_data['slug']}")
        print(f"\n    🎰 CROSS-PLATFORM ARBITRAGE")
        print(f"    Platform Pair: {' ↔ '.join(opp.platform_pair)}")
        print(f"    Strategy:")
        print(f"      • Buy {opp.side_a.outcome} on {opp.side_a.platform} at {opp.side_a.price:.1%}")
        print(f"      • Buy {opp.side_b.outcome} on {opp.side_b.platform} at {opp.side_b.price:.1%}")
        print(f"\n    💰 Profitability:")
        print(f"      Total Cost (with fees): ${opp.total_cost:.4f}")
        print(f"      Payout (guaranteed): $1.0000")
        print(f"      Guaranteed Profit: ${opp.guaranteed_profit:.4f} ({opp.guaranteed_profit*100:.2f}%)")
        print(f"\n    📊 Position Sizing:")
        print(f"      Suggested Total Stake: ${opp.suggested_stake:.2f}")
        print(f"      Stake on {opp.side_a.platform}: ${opp.side_a.stake:.2f}")
        print(f"      Stake on {opp.side_b.platform}: ${opp.side_b.stake:.2f}")
        print(f"      Expected Profit: ${opp.suggested_stake * opp.guaranteed_profit:.2f}")
        print(f"\n    {'-'*76}")

    print("\n" + "="*80)
    print("DISCLAIMER")
    print("="*80)
    print("""
This is for RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.

Risks to consider:
• Execution risk: Prices may move before you complete both trades
• Platform risk: One platform could freeze/cancel
• Liquidity risk: May not be able to fill full size
• Regulatory risk: Terms of service may prohibit arbitrage
• Settlement risk: Markets could resolve differently on each platform

Always conduct your own due diligence and risk assessment.
""")


async def main():
    """Main arbitrage scanning workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="POLYSEER Arbitrage Scanner")
    parser.add_argument("--limit", type=int, default=20, help="Max markets per platform")
    parser.add_argument("--bankroll", type=float, default=settings.DEFAULT_BANKROLL, help="Available capital")
    parser.add_argument("--min-profit", type=float, default=0.01, help="Minimum profit (0.01 = 1%%)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category keyword")

    args = parser.parse_args()

    # Scan for opportunities
    opportunities = await scan_for_arbitrage(
        limit=args.limit,
        bankroll=args.bankroll,
        min_profit_threshold=args.min_profit,
        category_filter=args.category
    )

    # Display results
    await display_arbitrage_results(opportunities)


if __name__ == "__main__":
    asyncio.run(main())
