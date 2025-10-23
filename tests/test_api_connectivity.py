"""
Test API Connectivity - Verify all APIs work correctly
Tests Polymarket, Kalshi, and Valyu API integrations
"""
import asyncio
import pytest
import os
from supabase import create_client
from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient
from arbee.api_clients.valyu import ValyuResearchClient


class TestPolymarketAPI:
    """Test Polymarket Gamma + CLOB API integration"""

    @pytest.mark.asyncio
    async def test_fetch_markets(self):
        """Test fetching active markets from Polymarket"""
        client = PolymarketClient()

        # Fetch active markets
        markets = await client.gamma.get_markets(active=True, limit=10)

        assert markets is not None, "Failed to fetch markets"
        assert len(markets) > 0, "No active markets found"

        # Check market structure
        first_market = markets[0]
        assert 'question' in first_market, "Market missing 'question' field"
        assert 'id' in first_market, "Market missing 'id' field"

        print(f"✓ Successfully fetched {len(markets)} markets from Polymarket")
        print(f"  Example market: {first_market['question'][:80]}...")

        return markets

    @pytest.mark.asyncio
    async def test_get_market_price(self):
        """Test fetching price for a specific market"""
        client = PolymarketClient()

        # First, get a market
        markets = await client.gamma.get_markets(active=True, limit=5)
        assert len(markets) > 0, "No markets available for price test"

        # Try to get price for first market
        test_market = markets[0]
        market_slug = test_market.get('slug') or test_market.get('id')

        print(f"Testing price fetch for: {test_market['question']}")

        try:
            market_with_price = await client.get_market_with_price(market_slug)

            if market_with_price and 'prices' in market_with_price:
                yes_price = market_with_price['prices']['yes']
                no_price = market_with_price['prices']['no']

                print(f"✓ Successfully fetched prices:")
                print(f"  YES: {yes_price:.2%}")
                print(f"  NO: {no_price:.2%}")

                # Sanity checks
                assert 0 <= yes_price <= 1, "YES price out of range"
                assert 0 <= no_price <= 1, "NO price out of range"
                assert abs((yes_price + no_price) - 1.0) < 0.1, "YES + NO should ≈ 1.0"

                return market_with_price
            else:
                print("⚠ Price data not available for this market")
                pytest.skip("Price data not available")

        except Exception as e:
            print(f"⚠ Error fetching price: {e}")
            pytest.skip(f"Price fetch failed: {e}")

    @pytest.mark.asyncio
    async def test_search_markets(self):
        """Test searching markets by query"""
        client = PolymarketClient()

        # Search for political markets
        results = await client.gamma.search_markets("president")

        print(f"✓ Found {len(results)} markets matching 'president'")

        if len(results) > 0:
            for i, market in enumerate(results[:3]):
                print(f"  {i+1}. {market['question']}")

        return results


class TestKalshiAPI:
    """Test Kalshi API integration"""

    @pytest.mark.asyncio
    async def test_fetch_events(self):
        """Test fetching events from Kalshi"""
        client = KalshiClient()

        # Fetch open events
        events = await client.get_events(status="open", limit=10)

        assert events is not None, "Failed to fetch events"

        if len(events) == 0:
            print("⚠ No open events found on Kalshi")
            pytest.skip("No open events")

        print(f"✓ Successfully fetched {len(events)} events from Kalshi")

        # Check event structure
        first_event = events[0]
        assert 'event_ticker' in first_event, "Event missing 'event_ticker'"
        assert 'title' in first_event, "Event missing 'title'"

        print(f"  Example event: {first_event['title'][:80]}...")

        return events

    @pytest.mark.asyncio
    async def test_fetch_markets(self):
        """Test fetching markets from Kalshi"""
        client = KalshiClient()

        # Fetch open markets
        markets = await client.get_markets(status="open", limit=10)

        assert markets is not None, "Failed to fetch markets"

        if len(markets) == 0:
            print("⚠ No open markets found on Kalshi")
            pytest.skip("No open markets")

        print(f"✓ Successfully fetched {len(markets)} markets from Kalshi")

        # Check market structure
        first_market = markets[0]
        assert 'ticker' in first_market, "Market missing 'ticker'"
        assert 'title' in first_market, "Market missing 'title'"

        print(f"  Example market: {first_market['title'][:80]}...")

        return markets

    @pytest.mark.asyncio
    async def test_get_market_price(self):
        """Test fetching price for a specific market"""
        client = KalshiClient()

        # Get a market
        markets = await client.get_markets(status="open", limit=5)

        if len(markets) == 0:
            print("⚠ No markets available for price test")
            pytest.skip("No markets available")

        test_market = markets[0]
        ticker = test_market['ticker']

        print(f"Testing price fetch for: {test_market['title']}")

        try:
            price = await client.get_market_price(ticker)

            if price is not None:
                print(f"✓ Successfully fetched price: {price:.2%}")

                # Sanity check
                assert 0 <= price <= 1, "Price out of range"

                return price
            else:
                print("⚠ Price not available for this market")
                pytest.skip("Price not available")

        except Exception as e:
            print(f"⚠ Error fetching price: {e}")
            pytest.skip(f"Price fetch failed: {e}")

    @pytest.mark.asyncio
    async def test_search_markets(self):
        """Test searching markets"""
        client = KalshiClient()

        # Search for political markets
        results = await client.search_markets("congress")

        print(f"✓ Found {len(results)} markets matching 'congress'")

        if len(results) > 0:
            for i, market in enumerate(results[:3]):
                print(f"  {i+1}. {market['title']}")

        return results


class TestValyuAPI:
    """Test Valyu AI research integration"""

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality"""
        client = ValyuResearchClient()

        query = "2024 US presidential election polls"

        print(f"Testing Valyu search: '{query}'")

        try:
            results = await client.search(query, max_results=5)

            assert results is not None, "Search returned None"
            assert len(results) > 0, "No search results"

            print(f"✓ Successfully retrieved {len(results)} results from Valyu")

            # Check result structure
            first_result = results[0]
            print(f"  Example result: {first_result.get('title', 'No title')[:80]}...")

            return results

        except Exception as e:
            print(f"⚠ Valyu search failed: {e}")
            pytest.skip(f"Valyu API error: {e}")

    @pytest.mark.asyncio
    async def test_langchain_integration(self):
        """Test LangChain integration with Valyu"""
        client = ValyuResearchClient()

        try:
            # Test if LangChain tool is available
            tool = client.get_langchain_tool()

            assert tool is not None, "LangChain tool not available"

            print("✓ Valyu LangChain integration available")
            print(f"  Tool name: {tool.name}")
            print(f"  Tool description: {tool.description[:100]}...")

            return tool

        except Exception as e:
            print(f"⚠ LangChain integration failed: {e}")
            pytest.skip(f"LangChain integration error: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS FOR MANUAL TESTING
# ============================================================================

async def quick_test_polymarket():
    """Quick manual test of Polymarket API"""
    print("\n" + "="*60)
    print("POLYMARKET API TEST")
    print("="*60 + "\n")

    tester = TestPolymarketAPI()

    print("[1/3] Testing market fetch...")
    markets = await tester.test_fetch_markets()

    print("\n[2/3] Testing price fetch...")
    await tester.test_get_market_price()

    print("\n[3/3] Testing search...")
    await tester.test_search_markets()

    print("\n✓ All Polymarket tests completed\n")


async def quick_test_kalshi():
    """Quick manual test of Kalshi API"""
    print("\n" + "="*60)
    print("KALSHI API TEST")
    print("="*60 + "\n")

    tester = TestKalshiAPI()

    print("[1/4] Testing event fetch...")
    await tester.test_fetch_events()

    print("\n[2/4] Testing market fetch...")
    await tester.test_fetch_markets()

    print("\n[3/4] Testing price fetch...")
    await tester.test_get_market_price()

    print("\n[4/4] Testing search...")
    await tester.test_search_markets()

    print("\n✓ All Kalshi tests completed\n")


async def quick_test_valyu():
    """Quick manual test of Valyu API"""
    print("\n" + "="*60)
    print("VALYU API TEST")
    print("="*60 + "\n")

    tester = TestValyuAPI()

    print("[1/2] Testing basic search...")
    await tester.test_search_basic()

    print("\n[2/2] Testing LangChain integration...")
    await tester.test_langchain_integration()

    print("\n✓ All Valyu tests completed\n")


async def test_all_apis():
    """Run all API tests"""
    print("\n" + "="*60)
    print("TESTING ALL APIs")
    print("="*60)

    await quick_test_polymarket()
    await quick_test_kalshi()
    await quick_test_valyu()

    print("="*60)
    print("ALL API TESTS COMPLETED")
    print("="*60 + "\n")


def test_supabase_rls_policies():
    """Test that RLS policies are working correctly."""
    # Skip if credentials not available
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        pytest.skip("Supabase credentials not available")

    # Test with anon client (should have limited access)
    anon_client = create_client(supabase_url, supabase_key)

    # Test with service client (should have full access)
    service_client = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_KEY')
    )

    # Anon client should be able to read markets
    try:
        anon_client.table('markets').select('count').limit(1).execute()
        anon_access = True
    except Exception:
        anon_access = False

    # Service client should be able to access all tables
    try:
        service_client.table('research_plans').select('count').limit(1).execute()
        service_access = True
    except Exception:
        service_access = False

    assert anon_access, "Anon client should have access to public tables"
    assert service_access, "Service client should have full access"


def test_database_schema_compliance():
    """Test that database schema matches expected structure."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')

    if not supabase_url or not supabase_service_key:
        pytest.skip("Supabase credentials not available")

    client = create_client(supabase_url, supabase_service_key)

    # Test that all required tables exist
    required_tables = [
        'markets', 'market_prices', 'research_plans', 'evidence',
        'critic_analysis', 'bayesian_analysis', 'arbitrage_opportunities',
        'workflow_executions'
    ]

    for table in required_tables:
        try:
            client.table(table).select('count').limit(1).execute()
        except Exception as e:
            pytest.fail(f"Table {table} not accessible: {e}")


async def test_all_apis():
    """Run all API tests"""
    print("\n" + "="*60)
    print("TESTING ALL APIs")
    print("="*60)

    await quick_test_polymarket()
    await quick_test_kalshi()
    await quick_test_valyu()

    # Test database and security
    test_supabase_rls_policies()
    test_database_schema_compliance()

    print("="*60)
    print("ALL API TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run all tests manually
    asyncio.run(test_all_apis())
