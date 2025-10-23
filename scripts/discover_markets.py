#!/usr/bin/env python3
"""
Market Discovery Script
Find markets that exist on BOTH Polymarket and Kalshi for arbitrage analysis
"""
import asyncio
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient


def similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity between two strings (0-1)"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_matching_markets(
    polymarket_markets: List[Dict[str, Any]],
    kalshi_markets: List[Dict[str, Any]],
    threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Find markets that likely represent the same event across platforms

    Args:
        polymarket_markets: List of Polymarket market dicts
        kalshi_markets: List of Kalshi market dicts
        threshold: Minimum similarity score (0-1) to consider a match

    Returns:
        List of matched market pairs
    """
    matches = []

    for pm_market in polymarket_markets:
        pm_question = pm_market.get('question', '')

        best_match = None
        best_score = 0.0

        for kalshi_market in kalshi_markets:
            kalshi_title = kalshi_market.get('title', '')

            score = similarity_score(pm_question, kalshi_title)

            if score > best_score and score >= threshold:
                best_score = score
                best_match = kalshi_market

        if best_match:
            matches.append({
                'similarity': best_score,
                'polymarket': {
                    'question': pm_market.get('question'),
                    'slug': pm_market.get('slug', pm_market.get('id')),
                    'id': pm_market.get('id'),
                    'end_date': pm_market.get('end_date_iso'),
                    'volume': pm_market.get('volume', 0)
                },
                'kalshi': {
                    'title': best_match.get('title'),
                    'ticker': best_match.get('ticker'),
                    'event_ticker': best_match.get('event_ticker'),
                    'close_time': best_match.get('close_time'),
                    'volume': best_match.get('volume', 0)
                }
            })

    # Sort by similarity score
    matches.sort(key=lambda x: x['similarity'], reverse=True)

    return matches


async def discover_political_markets(
    limit: int = 50,
    category_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Discover active political markets across both platforms

    Args:
        limit: Max markets to fetch from each platform
        category_filter: Optional keyword to filter markets

    Returns:
        Dict with separate lists and matches
    """
    print("Fetching markets from both platforms...\n")

    pm_client = PolymarketClient()
    kalshi_client = KalshiClient()

    # Fetch markets from both platforms
    print(f"[1/2] Fetching {limit} markets from Polymarket...")
    pm_markets = await pm_client.gamma.get_markets(active=True, limit=limit)

    print(f"[2/2] Fetching {limit} markets from Kalshi...")
    kalshi_markets = await kalshi_client.get_markets(status="open", limit=limit)

    # Filter by category if specified
    if category_filter:
        filter_lower = category_filter.lower()
        pm_markets = [
            m for m in pm_markets
            if filter_lower in m.get('question', '').lower()
            or filter_lower in m.get('description', '').lower()
        ]
        kalshi_markets = [
            m for m in kalshi_markets
            if filter_lower in m.get('title', '').lower()
            or filter_lower in m.get('subtitle', '').lower()
        ]

    print(f"\n✓ Fetched {len(pm_markets)} Polymarket markets")
    print(f"✓ Fetched {len(kalshi_markets)} Kalshi markets")

    # Find matches
    print(f"\nFinding matching markets...")
    matches = find_matching_markets(pm_markets, kalshi_markets, threshold=0.5)

    print(f"✓ Found {len(matches)} potential matches\n")

    return {
        'polymarket_markets': pm_markets,
        'kalshi_markets': kalshi_markets,
        'matches': matches,
        'summary': {
            'polymarket_count': len(pm_markets),
            'kalshi_count': len(kalshi_markets),
            'matched_count': len(matches)
        }
    }


async def fetch_prices_for_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch current prices for a matched market pair

    Args:
        match: Match dict from discover_political_markets()

    Returns:
        Match dict enriched with price data
    """
    pm_client = PolymarketClient()
    kalshi_client = KalshiClient()

    pm_slug = match['polymarket']['slug']
    kalshi_ticker = match['kalshi']['ticker']

    # Fetch Polymarket price
    try:
        pm_price = await pm_client.get_current_price(pm_slug)
        match['polymarket']['price'] = pm_price
    except Exception as e:
        print(f"⚠ Could not fetch Polymarket price for {pm_slug}: {e}")
        match['polymarket']['price'] = None

    # Fetch Kalshi price
    try:
        kalshi_price = await kalshi_client.get_market_price(kalshi_ticker)
        match['kalshi']['price'] = kalshi_price
    except Exception as e:
        print(f"⚠ Could not fetch Kalshi price for {kalshi_ticker}: {e}")
        match['kalshi']['price'] = None

    # Calculate arbitrage potential
    if match['polymarket']['price'] is not None and match['kalshi']['price'] is not None:
        pm_p = match['polymarket']['price']
        kalshi_p = match['kalshi']['price']

        # Simple mispricing check
        match['price_difference'] = abs(pm_p - kalshi_p)
        match['higher_platform'] = 'polymarket' if pm_p > kalshi_p else 'kalshi'

        # Cross-platform arbitrage check (buy both YES and NO)
        # Cost to buy YES on both platforms
        total_cost_both_yes = pm_p + kalshi_p

        # Cost to buy opposite sides (YES on one, NO on other)
        cost_yes_pm_no_kalshi = pm_p + (1.0 - kalshi_p)
        cost_no_pm_yes_kalshi = (1.0 - pm_p) + kalshi_p

        # Account for typical fees (2% Polymarket, 5% Kalshi)
        pm_fee = 0.02
        kalshi_fee = 0.05

        # Best cross-platform opportunity
        min_cost = min(cost_yes_pm_no_kalshi, cost_no_pm_yes_kalshi)
        total_cost_with_fees = min_cost * (1 + pm_fee + kalshi_fee)

        match['cross_platform_arbitrage'] = {
            'total_cost': total_cost_with_fees,
            'profit_per_dollar': max(0, 1.0 - total_cost_with_fees),
            'is_opportunity': total_cost_with_fees < 1.0
        }

    return match


async def display_discovery_results(
    results: Dict[str, Any],
    show_prices: bool = True,
    top_n: int = 10
):
    """Display discovery results in a readable format"""
    print("\n" + "="*80)
    print("MARKET DISCOVERY RESULTS")
    print("="*80)

    summary = results['summary']
    print(f"\n📊 Summary:")
    print(f"  Polymarket: {summary['polymarket_count']} markets")
    print(f"  Kalshi: {summary['kalshi_count']} markets")
    print(f"  Matches: {summary['matched_count']} potential matches")

    if summary['matched_count'] == 0:
        print("\n⚠ No matching markets found")
        return

    print(f"\n🎯 Top {min(top_n, len(results['matches']))} Matches:\n")

    matches = results['matches'][:top_n]

    for i, match in enumerate(matches, 1):
        print(f"\n[{i}] Similarity: {match['similarity']:.1%}")
        print(f"  📈 Polymarket: {match['polymarket']['question'][:70]}...")
        print(f"     Slug: {match['polymarket']['slug']}")

        print(f"  📊 Kalshi: {match['kalshi']['title'][:70]}...")
        print(f"     Ticker: {match['kalshi']['ticker']}")

        if show_prices:
            print(f"\n  ⏳ Fetching prices...")
            enriched_match = await fetch_prices_for_match(match)

            pm_price = enriched_match['polymarket'].get('price')
            kalshi_price = enriched_match['kalshi'].get('price')

            if pm_price is not None and kalshi_price is not None:
                print(f"  💰 Prices:")
                print(f"     Polymarket YES: {pm_price:.1%}")
                print(f"     Kalshi YES: {kalshi_price:.1%}")
                print(f"     Difference: {abs(pm_price - kalshi_price):.1%}")

                arb = enriched_match.get('cross_platform_arbitrage')
                if arb and arb['is_opportunity']:
                    print(f"  🎰 ARBITRAGE OPPORTUNITY!")
                    print(f"     Total cost (both sides): ${arb['total_cost']:.2f}")
                    print(f"     Profit per $1: ${arb['profit_per_dollar']:.4f}")
                    print(f"     Guaranteed return: {arb['profit_per_dollar']*100:.2f}%")

        print(f"  " + "-"*76)

    print("\n" + "="*80 + "\n")


async def main():
    """Main discovery workflow"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         POLYSEER MARKET DISCOVERY                            ║
║                   Find arbitrage opportunities across platforms              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Discover political markets
    results = await discover_political_markets(
        limit=100,
        category_filter="president"  # Focus on presidential markets
    )

    # Display results with prices
    await display_discovery_results(
        results,
        show_prices=True,
        top_n=10
    )

    print("✓ Discovery complete!")
    print("\nNext steps:")
    print("  1. Review matches above for arbitrage opportunities")
    print("  2. Use market slugs/tickers to run full POLYSEER analysis")
    print("  3. Check find_arbitrage.py for automated opportunity scanning\n")


if __name__ == "__main__":
    asyncio.run(main())
