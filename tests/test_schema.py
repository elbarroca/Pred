"""
Test schema validation for the prediction market database.
"""
import pytest
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from supabase import create_client


@pytest.fixture
def db_engine():
    """Create database engine for testing."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

    if not supabase_url or not supabase_key:
        pytest.skip("Supabase credentials not available")

    # Create direct PostgreSQL connection
    # This assumes SUPABASE_URL is in format: https://project.supabase.co
    # We need to convert it to postgres:// format
    db_url = supabase_url.replace('https://', 'postgresql://')
    db_url = db_url.replace('.supabase.co', '.supabase.co:5432/postgres')
    db_url = db_url + f"?sslmode=require"

    engine = create_engine(db_url, connect_args={"sslmode": "require"})
    return engine


class TestSchema:
    """Test database schema structure and constraints."""

    def test_schema_tables_exist(self, db_engine):
        """Test that all required tables exist in the schema."""
        expected_tables = [
            'markets',
            'market_prices',
            'research_plans',
            'evidence',
            'critic_analysis',
            'bayesian_analysis',
            'arbitrage_opportunities',
            'workflow_executions'
        ]

        with db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """))
            actual_tables = [row[0] for row in result]

        for table in expected_tables:
            assert table in actual_tables, f"Table {table} not found in schema"

    def test_markets_table_constraints(self, db_engine):
        """Test markets table constraints and structure."""
        with db_engine.connect() as conn:
            # Test provider constraint
            with pytest.raises(ProgrammingError):
                conn.execute(text("""
                    INSERT INTO markets (provider, market_slug, question)
                    VALUES ('invalid_provider', 'test', 'Test question')
                """))
                conn.commit()

            # Test unique constraint on provider + market_slug
            conn.execute(text("""
                INSERT INTO markets (provider, market_slug, question)
                VALUES ('polymarket', 'test-market', 'Test question')
            """))
            conn.commit()

            with pytest.raises(ProgrammingError):
                conn.execute(text("""
                    INSERT INTO markets (provider, market_slug, question)
                    VALUES ('polymarket', 'test-market', 'Another test question')
                """))
                conn.commit()

    def test_market_prices_time_series_structure(self, db_engine):
        """Test market_prices table is properly structured for time-series data."""
        with db_engine.connect() as conn:
            # Test price constraints (0-1 range)
            with pytest.raises(ProgrammingError):
                conn.execute(text("""
                    INSERT INTO markets (provider, market_slug, question)
                    VALUES ('polymarket', 'test-price', 'Test question')
                """))
                market_id = conn.execute(text("SELECT id FROM markets WHERE market_slug = 'test-price'")).fetchone()[0]

                conn.execute(text("""
                    INSERT INTO market_prices (market_id, price, implied_prob, timestamp)
                    VALUES (:market_id, 1.5, 1.5, NOW())
                """), {'market_id': market_id})
                conn.commit()

            # Test that timestamp defaults work
            conn.execute(text("""
                INSERT INTO markets (provider, market_slug, question)
                VALUES ('polymarket', 'test-timestamp', 'Test question')
            """))
            market_id = conn.execute(text("SELECT id FROM markets WHERE market_slug = 'test-timestamp'")).fetchone()[0]

            conn.execute(text("""
                INSERT INTO market_prices (market_id, price, implied_prob)
                VALUES (:market_id, 0.5, 0.5)
            """), {'market_id': market_id})
            conn.commit()

            result = conn.execute(text("""
                SELECT COUNT(*) FROM market_prices
                WHERE market_id = :market_id AND timestamp IS NOT NULL
            """), {'market_id': market_id}).fetchone()

            assert result[0] > 0, "Timestamp should be auto-populated"
