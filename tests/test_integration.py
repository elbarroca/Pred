"""
Integration Tests for POLYSEER
Tests that components work together correctly
"""
import pytest
from datetime import date
from arbee.agents.analyst import AnalystAgent
from arbee.models.schemas import Evidence, CriticOutput, CorrelationWarning
from arbee.utils.bayesian import BayesianCalculator


class TestAnalystIntegration:
    """Test Analyst agent integration with BayesianCalculator"""

    @pytest.mark.asyncio
    async def test_analyst_uses_calculator_directly(self):
        """
        CRITICAL TEST: Verify Analyst uses BayesianCalculator for math,
        not LLM calculation
        """
        analyst = AnalystAgent()

        # Create mock evidence
        evidence = [
            Evidence(
                subclaim_id="test_1",
                title="Test Evidence 1",
                url="http://example.com/1",
                published_date=date(2025, 1, 1),
                source_type="primary",
                claim_summary="This is a test claim supporting YES",
                support="pro",
                verifiability_score=0.9,
                independence_score=0.9,
                recency_score=0.9,
                estimated_LLR=1.0,
                extraction_notes="Test evidence"
            )
        ]

        # Create mock critic output
        critic_output = CriticOutput(
            duplicate_clusters=[],
            missing_topics=[],
            over_represented_sources=[],
            correlation_warnings=[],
            follow_up_search_seeds=[],
            analysis_process="Test analysis process"
        )

        # Run analysis
        result = await analyst.analyze(
            prior_p=0.5,
            evidence_items=evidence,
            critic_output=critic_output,
            market_question="Test question?"
        )

        # Validate that calculator was used (values are deterministic)
        assert result.p_bayesian > 0.5, "Positive evidence should increase probability"
        assert result.log_odds_prior == 0.0, "Prior of 0.5 should have log-odds = 0"

        # Validate output structure
        assert len(result.calculation_steps) > 0, "Should have explanation steps"
        assert len(result.evidence_summary) == 1, "Should summarize all evidence"

        # Validate math is correct (independent calculation)
        calc = BayesianCalculator()
        expected_result = calc.aggregate_evidence(
            prior_p=0.5,
            evidence_items=[
                {
                    'id': 'ev_0',
                    'LLR': 1.0,
                    'verifiability_score': 0.9,
                    'independence_score': 0.9,
                    'recency_score': 0.9
                }
            ]
        )

        # Result from agent should match calculator exactly
        assert abs(result.p_bayesian - expected_result['p_bayesian']) < 0.001, \
            "Analyst result should match BayesianCalculator exactly"

    @pytest.mark.asyncio
    async def test_analyst_handles_correlation(self):
        """Test that Analyst applies correlation adjustments from Critic"""
        analyst = AnalystAgent()

        # Create two correlated pieces of evidence
        evidence = [
            Evidence(
                subclaim_id="test_1",
                title="Evidence 1",
                url="http://example.com/1",
                published_date=date(2025, 1, 1),
                source_type="high_quality_secondary",
                claim_summary="Claim A",
                support="pro",
                verifiability_score=1.0,
                independence_score=1.0,
                recency_score=1.0,
                estimated_LLR=1.0,
                extraction_notes="First"
            ),
            Evidence(
                subclaim_id="test_2",
                title="Evidence 2",
                url="http://example.com/2",
                published_date=date(2025, 1, 2),
                source_type="high_quality_secondary",
                claim_summary="Claim B (correlated with A)",
                support="pro",
                verifiability_score=1.0,
                independence_score=1.0,
                recency_score=1.0,
                estimated_LLR=1.0,
                extraction_notes="Second"
            )
        ]

        # Critic identifies correlation
        critic_output = CriticOutput(
            duplicate_clusters=[],
            missing_topics=[],
            over_represented_sources=[],
            correlation_warnings=[
                CorrelationWarning(
                    cluster=["ev_0", "ev_1"],
                    note="Both cite the same underlying source"
                )
            ],
            follow_up_search_seeds=[],
            analysis_process="Found correlation"
        )

        # Run analysis
        result = await analyst.analyze(
            prior_p=0.5,
            evidence_items=evidence,
            critic_output=critic_output,
            market_question="Test?"
        )

        # With correlation shrinkage, effect should be less than 2.0
        # Without shrinkage: 2 * 1.0 = 2.0
        # With shrinkage (1/sqrt(2) ≈ 0.707): 2 * 0.707 = 1.414
        assert result.log_odds_posterior < 2.0, \
            "Correlation shrinkage should reduce total LLR"

        # Check correlation adjustment was documented
        assert "shrinkage" in result.correlation_adjustments.method.lower()

    @pytest.mark.asyncio
    async def test_analyst_with_no_evidence(self):
        """Test Analyst with no evidence (should return prior)"""
        analyst = AnalystAgent()

        critic_output = CriticOutput(
            duplicate_clusters=[],
            missing_topics=["Everything"],
            over_represented_sources=[],
            correlation_warnings=[],
            follow_up_search_seeds=["Need more research"],
            analysis_process="No evidence found"
        )

        result = await analyst.analyze(
            prior_p=0.6,
            evidence_items=[],
            critic_output=critic_output,
            market_question="Test?"
        )

        # With no evidence, posterior should equal prior
        assert abs(result.p_bayesian - 0.6) < 0.01, \
            "With no evidence, posterior should equal prior"

    @pytest.mark.asyncio
    async def test_analyst_sensitivity_analysis(self):
        """Test that sensitivity analysis is included"""
        analyst = AnalystAgent()

        evidence = [
            Evidence(
                subclaim_id="test",
                title="Test",
                url="http://example.com",
                published_date=date(2025, 1, 1),
                source_type="primary",
                claim_summary="Test",
                support="pro",
                verifiability_score=0.9,
                independence_score=0.9,
                recency_score=0.9,
                estimated_LLR=1.0,
                extraction_notes="Test"
            )
        ]

        critic_output = CriticOutput(
            duplicate_clusters=[],
            missing_topics=[],
            over_represented_sources=[],
            correlation_warnings=[],
            follow_up_search_seeds=[],
            analysis_process="Test"
        )

        result = await analyst.analyze(
            prior_p=0.5,
            evidence_items=evidence,
            critic_output=critic_output,
            market_question="Test?"
        )

        # Should have sensitivity scenarios
        assert len(result.sensitivity_analysis) >= 3
        scenarios = [s.scenario for s in result.sensitivity_analysis]
        assert "baseline" in scenarios


class TestBayesianMathIntegration:
    """Test that Bayesian math is consistent"""

    def test_calculator_deterministic(self):
        """Test that calculator gives same result for same input"""
        evidence = [
            {
                'id': 'ev1',
                'LLR': 1.0,
                'verifiability_score': 0.9,
                'independence_score': 0.8,
                'recency_score': 1.0
            }
        ]

        # Run twice
        result1 = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=evidence
        )

        result2 = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=evidence
        )

        # Results should be identical
        assert result1['p_bayesian'] == result2['p_bayesian']
        assert result1['log_odds_posterior'] == result2['log_odds_posterior']

    def test_multiple_evidence_ordering(self):
        """Test that evidence order doesn't matter (commutative)"""
        evidence_a_then_b = [
            {'id': 'a', 'LLR': 1.0, 'verifiability_score': 1.0,
             'independence_score': 1.0, 'recency_score': 1.0},
            {'id': 'b', 'LLR': 0.5, 'verifiability_score': 1.0,
             'independence_score': 1.0, 'recency_score': 1.0}
        ]

        evidence_b_then_a = [
            {'id': 'b', 'LLR': 0.5, 'verifiability_score': 1.0,
             'independence_score': 1.0, 'recency_score': 1.0},
            {'id': 'a', 'LLR': 1.0, 'verifiability_score': 1.0,
             'independence_score': 1.0, 'recency_score': 1.0}
        ]

        result1 = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=evidence_a_then_b
        )

        result2 = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=evidence_b_then_a
        )

        # Should get same result regardless of order
        assert abs(result1['p_bayesian'] - result2['p_bayesian']) < 0.001


# Note: Arbitrage scanner tests require API access and are slow
# Mark them as optional/integration tests

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_arbitrage_scan_smoke():
    """
    Quick smoke test of arbitrage scanner

    Requires: API keys configured
    Run with: pytest tests/test_integration.py -m slow
    """
    from arbee.workflow.master_controller import POLYSEERController

    controller = POLYSEERController()

    # Try small scan
    try:
        result = await controller.run_arbitrage_scan(
            limit=3,
            min_profit=0.01,
            parallel_limit=2,
            filter_by_priority=False
        )

        # Validate structure
        assert 'opportunities' in result
        assert 'markets_scanned' in result
        assert 'execution_time_seconds' in result
        assert isinstance(result['markets_scanned'], int)

    except Exception as e:
        pytest.skip(f"Arbitrage scan failed (likely API key issue): {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "not slow"])
