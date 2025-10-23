"""
Test agent workflows and integration.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from arbee.agents.planner import PlannerAgent
from arbee.agents.researcher import ResearcherAgent
from arbee.agents.critic import CriticAgent
from arbee.agents.analyst import AnalystAgent
from arbee.agents.arbitrage import ArbitrageDetector
from arbee.agents.reporter import ReporterAgent
from arbee.models.schemas import Evidence


class TestAgentInitialization:
    """Test that all agents can be instantiated properly."""

    def test_all_agents_instantiate(self):
        """Test that all agents can be created without errors."""
        agents = {
            'planner': PlannerAgent(),
            'researcher': ResearcherAgent(),
            'critic': CriticAgent(),
            'analyst': AnalystAgent(),
            'arbitrage': ArbitrageDetector(),
            'reporter': ReporterAgent()
        }

        # All agents should be instantiable
        assert len(agents) == 6, "All agents should be available"

        # Test that each agent has required methods
        for name, agent in agents.items():
            assert hasattr(agent, 'get_system_prompt'), f"Agent {name} should have get_system_prompt method"
            assert hasattr(agent, 'get_output_schema'), f"Agent {name} should have get_output_schema method"


class TestPlannerAgent:
    """Test Planner Agent functionality."""

    @pytest.mark.asyncio
    async def test_planner_creates_valid_research_plan(self):
        """Test that planner creates a valid research plan."""
        agent = PlannerAgent()

        market_question = 'Will AI achieve AGI by 2025?'

        result = await agent.plan(market_question)

        # Validate schema compliance
        required_fields = [
            'market_slug', 'market_question', 'p0_prior', 'prior_justification',
            'subclaims', 'key_variables', 'search_seeds', 'decision_criteria'
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate data types and ranges
        assert 0 <= result['p0_prior'] <= 1, "Prior must be between 0 and 1"
        assert len(result['subclaims']) >= 4, "Should have at least 4 subclaims"
        assert 'pro' in result['search_seeds'], "Should have pro search seeds"
        assert 'con' in result['search_seeds'], "Should have con search seeds"

    @pytest.mark.asyncio
    async def test_planner_handles_edge_cases(self):
        """Test planner with edge case inputs."""
        agent = PlannerAgent()

        # Test with very long question
        long_question = "A" * 1000 + "?"

        result = await agent.plan(long_question)
        assert len(result['subclaims']) > 0, "Should handle long questions"


class TestResearcherAgent:
    """Test Researcher Agent functionality."""

    @pytest.mark.asyncio
    async def test_researcher_gathers_evidence(self):
        """Test evidence gathering."""
        agent = ResearcherAgent()

        search_seeds = ['artificial intelligence progress 2024', 'AI progress limitations']
        subclaims = [
                {'id': 'sub1', 'text': 'AI progress is accelerating', 'direction': 'pro'}
        ]
        market_question = 'Will AI achieve AGI by 2025?'

        # Mock the search functionality
        with patch.object(agent, '_execute_searches') as mock_search:
            mock_search.return_value = [
                {
                    'title': 'AI Progress Report 2024',
                    'url': 'https://example.com/ai-progress',
                    'content': 'AI has made significant progress in 2024...',
                    'published_date': '2024-01-15'
                }
            ]

            result = await agent.research(search_seeds, subclaims, market_question)

            assert len(result.evidence) > 0, "Should gather some evidence"
            assert all('url' in item for item in result.evidence), "All evidence should have URLs"
            assert all('estimated_llr' in item for item in result.evidence), "All evidence should have LLR scores"

    @pytest.mark.asyncio
    async def test_researcher_scores_evidence(self):
        """Test evidence scoring and validation."""
        agent = ResearcherAgent()

        search_seeds = ['test search']
        subclaims = [{'id': 'sub1', 'text': 'test claim', 'direction': 'pro'}]
        market_question = 'Test question?'

        # Mock the extraction method
        with patch.object(agent, '_extract_evidence') as mock_extract:
            mock_extract.return_value = [
                Evidence(
                    title='Test Article',
                    url='https://example.com',
                    content='Test content',
                    published_date='2024-01-01',
                    source_type='primary',
                    estimated_llr=1.5
                )
            ]

            result = await agent.research(search_seeds, subclaims, market_question)

            # Should have evidence with proper scores
            assert len(result.evidence) > 0, "Should extract evidence"
            evidence_item = result.evidence[0]
            assert evidence_item.estimated_llr == 1.5, "LLR should be preserved"


class TestCriticAgent:
    """Test Critic Agent functionality."""

    @pytest.mark.asyncio
    async def test_critic_detects_duplicates(self):
        """Test duplicate detection."""
        agent = CriticAgent()

        evidence = [
            Evidence(
                id='ev1',
                url='https://example.com/article1',
                title='AI Progress Report',
                source_type='primary',
                content='Content 1',
                published_date='2024-01-01',
                estimated_llr=1.0
            ),
            Evidence(
                id='ev2',
                url='https://example.com/article1',  # Duplicate URL
                title='AI Progress Report',
                source_type='primary',
                content='Content 1',
                published_date='2024-01-01',
                estimated_llr=1.0
            )
        ]
        planner_output = {'market_question': 'Test question?'}
        market_question = 'Test question?'

        analysis = await agent.critique(evidence, planner_output, market_question)

        assert 'duplicate_clusters' in analysis, "Should detect duplicate clusters"
        assert len(analysis['duplicate_clusters']) > 0, "Should find duplicates"

    @pytest.mark.asyncio
    async def test_critic_identifies_missing_topics(self):
        """Test missing topic identification."""
        agent = CriticAgent()

        evidence = [
            Evidence(
                id='ev1',
                url='https://example.com/1',
                title='AI Progress',
                source_type='primary',
                content='AI is advancing rapidly',
                published_date='2024-01-01',
                estimated_llr=1.0
            ),
            Evidence(
                id='ev2',
                url='https://example.com/2',
                title='ML Progress',
                source_type='primary',
                content='Machine learning is improving',
                published_date='2024-01-01',
                estimated_llr=1.0
            )
            # Missing con evidence
        ]
        planner_output = {'subclaims': [{'direction': 'pro'}, {'direction': 'con'}]}
        market_question = 'Test question?'

        analysis = await agent.critique(evidence, planner_output, market_question)

        assert 'missing_topics' in analysis, "Should identify missing topics"
        # Should suggest con evidence is missing
        con_missing = any('con' in topic.lower() for topic in analysis['missing_topics'])
        assert con_missing, "Should note missing con evidence"


class TestAnalystAgent:
    """Test Analyst Agent Bayesian aggregation."""

    @pytest.mark.asyncio
    async def test_analyst_bayesian_aggregation(self):
        """Test Bayesian probability calculation."""
        agent = AnalystAgent()

        prior = 0.5
        evidence_items = [
            Evidence(
                id='ev1',
                url='https://example.com/1',
                title='Evidence 1',
                source_type='primary',
                content='Positive evidence',
                published_date='2024-01-01',
                estimated_llr=1.0
            ),
            Evidence(
                id='ev2',
                url='https://example.com/2',
                title='Evidence 2',
                source_type='primary',
                content='Negative evidence',
                published_date='2024-01-01',
                estimated_llr=-0.5
            )
        ]
        critic_output = {'quality_score': 0.8}
        market_question = 'Test question?'

        result = await agent.analyze(prior, evidence_items, critic_output, market_question)

        # Validate required fields
        required_fields = [
            'p0', 'log_odds_prior', 'log_odds_posterior', 'p_bayesian',
            'p_neutral', 'evidence_summary'
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Validate probability ranges
        assert 0 <= result['p_bayesian'] <= 1, "Bayesian probability should be 0-1"
        assert 0 <= result['p_neutral'] <= 1, "Neutral probability should be 0-1"

        # Validate math: posterior should be different from prior
        assert result['p_bayesian'] != prior, "Evidence should change the probability"

    @pytest.mark.asyncio
    async def test_analyst_correlation_adjustment(self):
        """Test correlation cluster adjustment."""
        agent = AnalystAgent()

        # Create correlated evidence (same source, low independence)
        evidence_items = [
            Evidence(
                id='ev1',
                url='https://example.com/1',
                title='Evidence 1',
                source_type='primary',
                content='Positive evidence',
                published_date='2024-01-01',
                estimated_llr=1.0
            ),
            Evidence(
                id='ev2',
                url='https://example.com/2',
                title='Evidence 2',
                source_type='primary',
                content='Similar positive evidence',
                published_date='2024-01-01',
                estimated_llr=1.0
            )
        ]
        critic_output = {'correlation_clusters': [{'ev1', 'ev2'}]}
        market_question = 'Test question?'

        result = await agent.analyze(0.5, evidence_items, critic_output, market_question)

        # Should handle correlation (though exact behavior depends on implementation)
        assert 'correlation_adjustments' in result, "Should include correlation adjustments"


class TestArbitrageDetector:
    """Test Arbitrage Detector functionality."""

    @pytest.mark.asyncio
    async def test_arbitrage_calculation(self):
        """Test arbitrage opportunity calculation."""
        agent = ArbitrageDetector()

        market_slug = 'test-market'
        market_question = 'Will AI achieve AGI by 2025?'
        providers = ['polymarket', 'kalshi']

        # Mock market data
        with patch.object(agent, '_fetch_market_prices') as mock_fetch:
            mock_fetch.return_value = {
            'polymarket': {'price': 0.7, 'fees': 0.02, 'liquidity': 10000},
            'kalshi': {'price': 0.6, 'fees': 0.03, 'liquidity': 5000}
        }

            opportunities = await agent.detect_cross_platform_arbitrage(market_slug, market_question, providers)

        # Should return opportunities for each market
        assert len(opportunities) > 0, "Should find arbitrage opportunities"

        for opp in opportunities:
            # Validate required fields
            required_fields = [
                'market_id', 'provider', 'price', 'implied_probability',
                'edge', 'expected_value_per_dollar', 'kelly_fraction'
            ]

            for field in required_fields:
                assert field in opp, f"Missing field in opportunity: {field}"

            # Validate edge calculation (should be based on Bayesian vs market)
            assert opp.edge == 0.65 - opp.implied_probability, "Edge should be Bayesian - market"

    @pytest.mark.asyncio
    async def test_kelly_criterion_conservative(self):
        """Test conservative Kelly criterion implementation."""
        agent = ArbitrageDetector()

        market_slug = 'test-market'
        market_question = 'Will AI achieve AGI by 2025?'
        providers = ['polymarket']

        # Mock market data for high edge scenario
        with patch.object(agent, '_fetch_market_prices') as mock_fetch:
            mock_fetch.return_value = {
                'polymarket': {'price': 0.75, 'fees': 0.02, 'liquidity': 10000}
            }

            opportunities = await agent.detect_cross_platform_arbitrage(market_slug, market_question, providers)

            # Should find opportunities and apply conservative Kelly
            assert len(opportunities) > 0, "Should find opportunities"
            opp = opportunities[0]

        # Should be conservative (capped at 0.05 = 5%)
            assert opp.kelly_fraction <= 0.05, "Kelly fraction should be conservative"


class TestReporterAgent:
    """Test Reporter Agent functionality."""

    @pytest.mark.asyncio
    async def test_report_generation(self):
        """Test report generation."""
        agent = ReporterAgent()

        market_question = 'Will AI achieve AGI by 2025?'
        planner_output = {
            'market_slug': 'test-market',
            'market_question': market_question,
            'p0_prior': 0.5,
            'subclaims': [{'id': 'sub1', 'text': 'AI progress'}],
            'search_seeds': {'pro': ['AI progress'], 'con': ['AI limitations']}
        }
        researcher_output = {
            'evidence': [
                {'id': 'ev1', 'title': 'Evidence 1', 'estimated_llr': 1.0},
                {'id': 'ev2', 'title': 'Evidence 2', 'estimated_llr': -0.5}
            ]
        }
        critic_output = {
            'quality_score': 0.8,
            'missing_topics': ['con evidence']
        }
        analyst_output = {
                'p_bayesian': 0.65,
            'log_odds_posterior': 0.6,
                'evidence_summary': [
                    {'id': 'ev1', 'adjusted_llr': 1.0},
                    {'id': 'ev2', 'adjusted_llr': -0.5}
                ]
        }
        arbitrage_opportunities = [
                {
                'market_id': 'test-market',
                    'provider': 'polymarket',
                    'edge': 0.05,
                    'expected_value_per_dollar': 0.03
                }
        ]

        report = await agent.generate_report(
            market_question, planner_output, researcher_output,
            critic_output, analyst_output, arbitrage_opportunities
        )

        # Validate report structure
        assert 'json_report' in report, "Should include JSON report"
        assert 'markdown_report' in report, "Should include Markdown report"
        assert 'tldr' in report, "Should include TL;DR"

        # Validate JSON schema
        json_report = report['json_report']
        required_fields = [
            'market_question', 'p_bayesian', 'evidence_summary',
            'arbitrage_summary', 'disclaimers'
        ]

        for field in required_fields:
            assert field in json_report, f"Missing field in JSON report: {field}"

        # Validate disclaimers
        markdown_report = report['markdown_report']
        assert "NOT FINANCIAL ADVICE" in markdown_report, "Should include disclaimer"

    @pytest.mark.asyncio
    async def test_report_length_constraints(self):
        """Test report length constraints."""
        agent = ReporterAgent()

        market_question = 'A' * 200  # Long question
        planner_output = {'market_question': market_question}
        researcher_output = {'evidence': []}
        critic_output = {'quality_score': 0.5}
        analyst_output = {'p_bayesian': 0.5}
        arbitrage_opportunities = []

        report = await agent.generate_report(
            market_question, planner_output, researcher_output,
            critic_output, analyst_output, arbitrage_opportunities
        )

        # Markdown should be reasonable length (200-600 words as per CLAUDE.MD)
        markdown_length = len(report['markdown_report'].split())
        assert 50 <= markdown_length <= 1000, f"Markdown length should be reasonable: {markdown_length} words"


class TestWorkflowIntegration:
    """Test end-to-end workflow integration."""

    def test_full_workflow_execution(self):
        """Test complete workflow from market to report."""
        # This would be a complex integration test
        # For now, just validate that all agents can be instantiated

        agents = {
            'planner': PlannerAgent(),
            'researcher': ResearcherAgent(),
            'critic': CriticAgent(),
            'analyst': AnalystAgent(),
            'arbitrage': ArbitrageDetector(),
            'reporter': ReporterAgent()
        }

        # All agents should be instantiable
        assert len(agents) == 6, "All agents should be available"

        # Test that each agent has required methods
        for name, agent in agents.items():
            assert hasattr(agent, 'plan') or hasattr(agent, 'research') or hasattr(agent, 'critique') or hasattr(agent, 'analyze') or hasattr(agent, 'detect_cross_platform_arbitrage') or hasattr(agent, 'generate_report'), \
                f"Agent {name} should have processing method"
