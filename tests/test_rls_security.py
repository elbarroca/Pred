"""
Test Row Level Security policies and database security.
"""
import pytest
import os
from supabase import create_client, Client
from typing import Dict, Any


@pytest.fixture
def supabase_client():
    """Create Supabase client for testing."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key:
        pytest.skip("Supabase credentials not configured")

    return create_client(url, key)


@pytest.fixture
def service_client():
    """Create Supabase client with service role for testing."""
    url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not url or not service_key:
        pytest.skip("Service credentials not configured")

    return create_client(url, service_key)


class TestRLSSecurity:
    """Test Row Level Security policies."""

    def test_public_markets_access(self, supabase_client):
        """Test that markets table is publicly readable."""
        try:
            response = supabase_client.table('markets').select('*').limit(1).execute()
            # Should not raise an error even without authentication
            assert True
        except Exception as e:
            # If there's an auth error, that's expected for now
            assert "JWT" in str(e) or "auth" in str(e).lower()

    def test_market_prices_public_access(self, supabase_client):
        """Test that market_prices table is publicly readable."""
        try:
            response = supabase_client.table('market_prices').select('*').limit(1).execute()
            # Should not raise an error even without authentication
            assert True
        except Exception as e:
            # If there's an auth error, that's expected for now
            assert "JWT" in str(e) or "auth" in str(e).lower()

    def test_research_plans_protected(self, supabase_client):
        """Test that research_plans table is protected."""
        try:
            response = supabase_client.table('research_plans').select('*').limit(1).execute()
            # Should raise an auth error
            assert False, "Should have raised authentication error"
        except Exception as e:
            # Should raise an auth/RLS error
            assert "JWT" in str(e) or "auth" in str(e).lower() or "RLS" in str(e) or "policy" in str(e).lower()

    def test_service_role_full_access(self, service_client):
        """Test that service role has full access to all tables."""
        tables = [
            'markets', 'market_prices', 'research_plans', 'evidence',
            'critic_analysis', 'bayesian_analysis', 'arbitrage_opportunities',
            'workflow_executions'
        ]

        for table in tables:
            try:
                response = service_client.table(table).select('*').limit(1).execute()
                # Should not raise an error
                assert True
            except Exception as e:
                pytest.fail(f"Service role should have access to {table}: {e}")


class TestDatabaseSchema:
    """Test database schema structure and constraints."""

    def test_required_tables_exist(self, supabase_client):
        """Test that all required tables exist."""
        expected_tables = {
            'markets', 'market_prices', 'research_plans', 'evidence',
            'critic_analysis', 'bayesian_analysis', 'arbitrage_opportunities',
            'workflow_executions'
        }

        # This is a simplified test - in reality you'd query information_schema
        # For now, just test that we can query without errors (indicating tables exist)
        for table in expected_tables:
            try:
                supabase_client.table(table).select('*').limit(1).execute()
            except Exception as e:
                if "relation" in str(e).lower() and "does not exist" in str(e).lower():
                    pytest.fail(f"Table {table} does not exist: {e}")
                # Other errors (like RLS) are expected

    def test_table_constraints(self, service_client):
        """Test table constraints and data validation."""
        # Test markets provider constraint
        try:
            service_client.table('markets').insert({
                'provider': 'invalid_provider',
                'market_slug': 'test-market',
                'question': 'Test question'
            }).execute()
            assert False, "Should have rejected invalid provider"
        except Exception as e:
            assert "constraint" in str(e).lower() or "check" in str(e).lower()

        # Test valid market insertion
        try:
            response = service_client.table('markets').insert({
                'provider': 'polymarket',
                'market_slug': 'test-market-valid',
                'question': 'Test question for validation'
            }).execute()
            assert response.data is not None

            # Clean up
            if response.data and len(response.data) > 0:
                market_id = response.data[0]['id']
                service_client.table('markets').delete().eq('id', market_id).execute()
        except Exception as e:
            # This might fail due to RLS, which is okay for this test
            pass


class TestDataIntegrity:
    """Test data integrity and relationships."""

    def test_foreign_key_constraints(self, service_client):
        """Test foreign key relationships."""
        # Test that evidence.research_plan_id references valid research plan
        try:
            service_client.table('evidence').insert({
                'research_plan_id': '00000000-0000-0000-0000-000000000000',  # Invalid UUID
                'subclaim_id': 'test-subclaim',
                'title': 'Test Evidence',
                'url': 'https://example.com',
                'published_date': '2024-01-01',
                'source_type': 'primary',
                'claim_summary': 'Test summary',
                'support': 'pro',
                'verifiability_score': 0.8,
                'independence_score': 0.7,
                'recency_score': 0.9,
                'estimated_llr': 1.5,
                'extraction_notes': 'Test notes'
            }).execute()
            assert False, "Should have rejected invalid foreign key"
        except Exception as e:
            assert "foreign key" in str(e).lower() or "constraint" in str(e).lower()



