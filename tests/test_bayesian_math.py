"""
Unit Tests for BayesianCalculator
Tests mathematical correctness of Bayesian aggregation
"""
import pytest
import math
from arbee.utils.bayesian import BayesianCalculator, KellyCalculator


class TestBayesianCalculator:
    """Test suite for BayesianCalculator"""

    def test_prob_to_log_odds_neutral(self):
        """Test conversion of 50% probability to log-odds"""
        calc = BayesianCalculator()
        result = calc.prob_to_log_odds(0.5)
        assert result == 0.0, "50% probability should convert to 0.0 log-odds"

    def test_prob_to_log_odds_positive(self):
        """Test conversion of high probabilities"""
        calc = BayesianCalculator()
        result = calc.prob_to_log_odds(0.9)
        assert result > 0, "High probability should have positive log-odds"
        # ln(0.9/0.1) = ln(9) ≈ 2.197
        assert abs(result - 2.197) < 0.01

    def test_prob_to_log_odds_negative(self):
        """Test conversion of low probabilities"""
        calc = BayesianCalculator()
        result = calc.prob_to_log_odds(0.1)
        assert result < 0, "Low probability should have negative log-odds"
        # ln(0.1/0.9) = ln(1/9) ≈ -2.197
        assert abs(result - (-2.197)) < 0.01

    def test_log_odds_to_prob_neutral(self):
        """Test conversion of 0 log-odds to probability"""
        calc = BayesianCalculator()
        result = calc.log_odds_to_prob(0.0)
        assert abs(result - 0.5) < 0.001, "0 log-odds should convert to 50%"

    def test_log_odds_to_prob_positive(self):
        """Test conversion of positive log-odds"""
        calc = BayesianCalculator()
        result = calc.log_odds_to_prob(2.0)
        assert result > 0.5, "Positive log-odds should be > 50%"
        # exp(2)/(1+exp(2)) ≈ 0.881
        assert abs(result - 0.881) < 0.01

    def test_log_odds_to_prob_negative(self):
        """Test conversion of negative log-odds"""
        calc = BayesianCalculator()
        result = calc.log_odds_to_prob(-2.0)
        assert result < 0.5, "Negative log-odds should be < 50%"
        # exp(-2)/(1+exp(-2)) ≈ 0.119
        assert abs(result - 0.119) < 0.01

    def test_roundtrip_conversion(self):
        """Test that prob->log_odds->prob is identity"""
        calc = BayesianCalculator()
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            log_odds = calc.prob_to_log_odds(p)
            p_recovered = calc.log_odds_to_prob(log_odds)
            assert abs(p - p_recovered) < 0.001, f"Roundtrip failed for p={p}"

    def test_adjust_llr(self):
        """Test LLR adjustment by quality scores"""
        calc = BayesianCalculator()

        # Full quality scores
        adjusted = calc.adjust_llr(
            llr=1.0,
            verifiability=1.0,
            independence=1.0,
            recency=1.0
        )
        assert adjusted == 1.0, "Full quality should not change LLR"

        # Reduced quality scores
        adjusted = calc.adjust_llr(
            llr=1.0,
            verifiability=0.9,
            independence=0.8,
            recency=1.0
        )
        expected = 1.0 * 0.9 * 0.8 * 1.0
        assert abs(adjusted - expected) < 0.001

    def test_adjust_llr_zero_quality(self):
        """Test that zero quality score zeros out LLR"""
        calc = BayesianCalculator()
        adjusted = calc.adjust_llr(
            llr=1.0,
            verifiability=0.0,  # Zero quality
            independence=1.0,
            recency=1.0
        )
        assert adjusted == 0.0, "Zero quality should zero out LLR"

    def test_correlation_shrinkage_single(self):
        """Test that single item cluster has no shrinkage"""
        calc = BayesianCalculator()
        llrs = [1.0]
        shrunk = calc.apply_correlation_shrinkage(llrs, cluster_size=1)
        assert shrunk == [1.0], "Single item should not be shrunk"

    def test_correlation_shrinkage_pair(self):
        """Test shrinkage for pair of correlated items"""
        calc = BayesianCalculator()
        llrs = [1.0, 1.0]
        shrunk = calc.apply_correlation_shrinkage(llrs, cluster_size=2)

        # Should be shrunk by 1/sqrt(2) ≈ 0.707
        expected = 1.0 / math.sqrt(2)
        assert abs(shrunk[0] - expected) < 0.001
        assert abs(shrunk[1] - expected) < 0.001

    def test_correlation_shrinkage_triple(self):
        """Test shrinkage for triple of correlated items"""
        calc = BayesianCalculator()
        llrs = [1.0, 1.0, 1.0]
        shrunk = calc.apply_correlation_shrinkage(llrs, cluster_size=3)

        # Should be shrunk by 1/sqrt(3) ≈ 0.577
        expected = 1.0 / math.sqrt(3)
        for val in shrunk:
            assert abs(val - expected) < 0.001

    def test_aggregate_evidence_no_evidence(self):
        """Test aggregation with no evidence (should return prior)"""
        result = BayesianCalculator.aggregate_evidence(
            prior_p=0.6,
            evidence_items=[],
            correlation_clusters=None
        )

        assert abs(result['p_bayesian'] - 0.6) < 0.001, \
            "With no evidence, posterior should equal prior"

    def test_aggregate_evidence_positive(self):
        """Test aggregation with positive evidence"""
        result = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=[
                {
                    'id': 'ev1',
                    'LLR': 1.0,
                    'verifiability_score': 1.0,
                    'independence_score': 1.0,
                    'recency_score': 1.0
                }
            ]
        )

        assert 'p_bayesian' in result
        assert result['p_bayesian'] > 0.5, \
            "Positive evidence should increase probability"

    def test_aggregate_evidence_negative(self):
        """Test aggregation with negative evidence"""
        result = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=[
                {
                    'id': 'ev1',
                    'LLR': -1.0,
                    'verifiability_score': 1.0,
                    'independence_score': 1.0,
                    'recency_score': 1.0
                }
            ]
        )

        assert result['p_bayesian'] < 0.5, \
            "Negative evidence should decrease probability"

    def test_aggregate_evidence_mixed(self):
        """Test aggregation with mixed evidence"""
        result = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=[
                {
                    'id': 'ev1',
                    'LLR': 1.0,  # Positive
                    'verifiability_score': 1.0,
                    'independence_score': 1.0,
                    'recency_score': 1.0
                },
                {
                    'id': 'ev2',
                    'LLR': -0.5,  # Negative
                    'verifiability_score': 1.0,
                    'independence_score': 1.0,
                    'recency_score': 1.0
                }
            ]
        )

        # Net effect: +1.0 - 0.5 = +0.5 (should increase)
        assert result['p_bayesian'] > 0.5

    def test_aggregate_evidence_structure(self):
        """Test that aggregation result has correct structure"""
        result = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=[
                {
                    'id': 'ev1',
                    'LLR': 1.0,
                    'verifiability_score': 1.0,
                    'independence_score': 1.0,
                    'recency_score': 1.0
                }
            ]
        )

        # Check all required fields present
        assert 'p0' in result
        assert 'log_odds_prior' in result
        assert 'evidence_summary' in result
        assert 'total_adjusted_LLR' in result
        assert 'log_odds_posterior' in result
        assert 'p_bayesian' in result

        # Check evidence summary structure
        assert len(result['evidence_summary']) == 1
        ev_summary = result['evidence_summary'][0]
        assert 'id' in ev_summary
        assert 'LLR' in ev_summary
        assert 'weight' in ev_summary
        assert 'adjusted_LLR' in ev_summary

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis scenarios"""
        evidence_items = [
            {
                'id': 'ev1',
                'LLR': 1.0,
                'verifiability_score': 1.0,
                'independence_score': 1.0,
                'recency_score': 1.0
            }
        ]

        results = BayesianCalculator.sensitivity_analysis(
            prior_p=0.5,
            evidence_items=evidence_items
        )

        # Should return list of scenarios
        assert isinstance(results, list)
        assert len(results) >= 4  # baseline, +25%, -25%, remove weakest

        # Check structure
        for scenario in results:
            assert 'scenario' in scenario
            assert 'p' in scenario
            assert 0 <= scenario['p'] <= 1

    def test_clamping_extremes(self):
        """Test that extreme probabilities are clamped"""
        result = BayesianCalculator.aggregate_evidence(
            prior_p=0.5,
            evidence_items=[
                {
                    'id': 'ev1',
                    'LLR': 10.0,  # Very high LLR
                    'verifiability_score': 1.0,
                    'independence_score': 1.0,
                    'recency_score': 1.0
                }
            ]
        )

        # Should be clamped to 0.99
        assert result['p_bayesian'] <= 0.99


class TestKellyCalculator:
    """Test suite for KellyCalculator"""

    def test_kelly_fraction_positive_edge(self):
        """Test Kelly calculation with positive edge"""
        kelly = KellyCalculator.kelly_fraction(edge=0.1, odds=1.0, max_fraction=1.0)
        assert kelly > 0, "Positive edge should give positive Kelly"
        assert kelly <= 1.0, "Kelly should be capped"

    def test_kelly_fraction_zero_edge(self):
        """Test Kelly calculation with zero edge"""
        kelly = KellyCalculator.kelly_fraction(edge=0.0, odds=1.0)
        assert kelly == 0.0, "Zero edge should give zero Kelly"

    def test_kelly_fraction_negative_edge(self):
        """Test Kelly calculation with negative edge"""
        kelly = KellyCalculator.kelly_fraction(edge=-0.1, odds=1.0)
        assert kelly == 0.0, "Negative edge should give zero Kelly"

    def test_kelly_fraction_max_cap(self):
        """Test that Kelly is capped at max_fraction"""
        kelly = KellyCalculator.kelly_fraction(
            edge=0.5,  # Large edge
            odds=1.0,
            max_fraction=0.05  # 5% cap
        )
        assert kelly <= 0.05, "Kelly should be capped at max_fraction"

    def test_expected_value_positive(self):
        """Test EV calculation with edge"""
        ev = KellyCalculator.expected_value(
            p_true=0.6,
            p_market=0.5,
            transaction_costs=0.02,
            slippage=0.01
        )
        # Edge = 0.1, costs = 0.03, EV = 0.07
        assert abs(ev - 0.07) < 0.001

    def test_expected_value_negative(self):
        """Test EV calculation with high costs"""
        ev = KellyCalculator.expected_value(
            p_true=0.52,
            p_market=0.5,
            transaction_costs=0.05,  # High costs
            slippage=0.02
        )
        # Edge = 0.02, costs = 0.07, EV = -0.05
        assert ev < 0, "High costs should give negative EV"

    def test_calculate_stake(self):
        """Test full stake calculation"""
        result = KellyCalculator.calculate_stake(
            bankroll=10000,
            p_true=0.6,
            p_market=0.5,
            transaction_costs=0.02,
            slippage=0.01,
            max_kelly=0.05
        )

        # Check structure
        assert 'edge' in result
        assert 'expected_value_per_dollar' in result
        assert 'kelly_fraction' in result
        assert 'suggested_stake' in result

        # Check values
        assert abs(result['edge'] - 0.1) < 0.001  # Use approximate comparison
        assert result['suggested_stake'] > 0
        assert result['suggested_stake'] <= 10000 * 0.05  # Capped at 5%

    def test_calculate_stake_negative_ev(self):
        """Test that negative EV gives zero stake"""
        result = KellyCalculator.calculate_stake(
            bankroll=10000,
            p_true=0.5,
            p_market=0.5,
            transaction_costs=0.1,  # High costs, no edge
            slippage=0.0
        )

        assert result['expected_value_per_dollar'] <= 0
        assert result['kelly_fraction'] == 0.0
        assert result['suggested_stake'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
