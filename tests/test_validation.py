"""
Unit Tests for Validation Utilities
Tests CoT validation and evidence quality checks
"""
import pytest
from datetime import date
from arbee.utils.validation import (
    validate_cot_output,
    validate_evidence_quality,
    validate_probability,
    validate_llr
)
from arbee.models.schemas import (
    PlannerOutput,
    AnalystOutput,
    CriticOutput,
    Evidence,
    Subclaim,
    CorrelationWarning
)


class TestCoTValidation:
    """Test Chain-of-Thought validation"""

    def test_valid_planner_output(self):
        """Test that valid PlannerOutput passes validation"""
        output = PlannerOutput(
            market_slug="test-market",
            market_question="Will this pass?",
            p0_prior=0.5,
            prior_justification="Base rate reasoning",
            subclaims=[
                Subclaim(id="sc1", text="Subclaim 1", direction="pro"),
                Subclaim(id="sc2", text="Subclaim 2", direction="con"),
                Subclaim(id="sc3", text="Subclaim 3", direction="pro"),
                Subclaim(id="sc4", text="Subclaim 4", direction="con")
            ],
            key_variables=["var1"],
            search_seeds={"pro": ["query1"], "con": ["query2"], "general": ["query3"]},
            decision_criteria=["criteria1"],
            reasoning_trace=(
                "Step 1: Initial assessment of the question.\n"
                "Step 2: Analyzed base rates and reference classes.\n"
                "Step 3: Identified key variables affecting outcome.\n"
                "Step 4: Generated balanced subclaims.\n"
                "Step 5: Designed search strategy.\n"
                "Step 6: Estimated prior probability based on analysis."
            )
        )

        # Should not raise
        validate_cot_output(output)

    def test_short_reasoning_trace_fails(self):
        """Test that short reasoning_trace fails validation"""
        output = PlannerOutput(
            market_slug="test",
            market_question="Will this pass?",
            p0_prior=0.5,
            prior_justification="Base rate",
            subclaims=[
                Subclaim(id="sc1", text="Test", direction="pro"),
                Subclaim(id="sc2", text="Test", direction="con"),
                Subclaim(id="sc3", text="Test", direction="pro"),
                Subclaim(id="sc4", text="Test", direction="con")
            ],
            key_variables=["var1"],
            search_seeds={"pro": ["q1"], "con": ["q2"], "general": ["q3"]},
            decision_criteria=["c1"],
            reasoning_trace="Too short"  # Only 9 chars
        )

        with pytest.raises(ValueError) as exc_info:
            validate_cot_output(output, min_length=100)

        assert "reasoning_trace too short" in str(exc_info.value)

    def test_missing_step_structure_fails(self):
        """Test that reasoning without step structure fails"""
        output = PlannerOutput(
            market_slug="test",
            market_question="Will this pass?",
            p0_prior=0.5,
            prior_justification="Base rate",
            subclaims=[
                Subclaim(id="sc1", text="Test", direction="pro"),
                Subclaim(id="sc2", text="Test", direction="con"),
                Subclaim(id="sc3", text="Test", direction="pro"),
                Subclaim(id="sc4", text="Test", direction="con")
            ],
            key_variables=["var1"],
            search_seeds={"pro": ["q1"], "con": ["q2"], "general": ["q3"]},
            decision_criteria=["c1"],
            reasoning_trace=(
                "This reasoning trace has enough characters to pass the length check "
                "and goes through various aspects of the question and analysis process. "
                "However, it lacks proper structure with explicit markers that would "
                "indicate clear progressive reasoning through discrete stages."
            )
        )

        with pytest.raises(ValueError) as exc_info:
            validate_cot_output(output)

        assert "does not appear to contain step-by-step reasoning" in str(exc_info.value)

    def test_valid_analyst_output(self):
        """Test that valid AnalystOutput passes validation"""
        output = AnalystOutput(
            p0=0.5,
            log_odds_prior=0.0,
            evidence_summary=[{"id": "ev1", "LLR": 1.0, "weight": 0.9, "adjusted_LLR": 0.9}],
            correlation_adjustments={"method": "shrinkage", "details": "1/sqrt(n)"},
            log_odds_posterior=0.9,
            p_bayesian=0.71,
            p_neutral=0.5,
            sensitivity_analysis=[
                {"scenario": "baseline", "p": 0.71},
                {"scenario": "+25%", "p": 0.78},
                {"scenario": "-25%", "p": 0.64}
            ],
            calculation_steps=[
                "Step 1: Started with prior p0 = 0.50 (neutral), log-odds = 0.0",
                "Step 2: Processed 10 evidence items with quality adjustments",
                "Step 3: Applied correlation shrinkage to 2 clusters",
                "Step 4: Summed adjusted LLRs: total = +0.9",
                "Step 5: Updated log-odds: 0.0 + 0.9 = 0.9",
                "Step 6: Converted to probability: p = 71%"
            ]
        )

        # Should not raise
        validate_cot_output(output)

    def test_few_calculation_steps_fails(self):
        """Test that too few calculation steps fails"""
        output = AnalystOutput(
            p0=0.5,
            log_odds_prior=0.0,
            evidence_summary=[],
            correlation_adjustments={"method": "none", "details": ""},
            log_odds_posterior=0.0,
            p_bayesian=0.5,
            p_neutral=0.5,
            sensitivity_analysis=[],
            calculation_steps=[
                "Step 1: Started with prior",
                "Step 2: No evidence"
            ]  # Only 2 steps
        )

        with pytest.raises(ValueError) as exc_info:
            validate_cot_output(output, min_steps=3)

        assert "calculation_steps has only 2 steps" in str(exc_info.value)

    def test_short_calculation_steps_fails(self):
        """Test that very short step explanations fail"""
        output = AnalystOutput(
            p0=0.5,
            log_odds_prior=0.0,
            evidence_summary=[],
            correlation_adjustments={"method": "none", "details": ""},
            log_odds_posterior=0.0,
            p_bayesian=0.5,
            p_neutral=0.5,
            sensitivity_analysis=[],
            calculation_steps=[
                "Step 1: This is detailed",
                "Step 2: Short",  # Too short
                "Step 3: Also detailed enough"
            ]
        )

        with pytest.raises(ValueError) as exc_info:
            validate_cot_output(output)

        assert "steps that are too short" in str(exc_info.value)

    def test_non_strict_mode_logs_warning(self):
        """Test that non-strict mode logs warning instead of raising"""
        output = PlannerOutput(
            market_slug="test",
            market_question="Test?",
            p0_prior=0.5,
            prior_justification="Test",
            subclaims=[
                Subclaim(id="sc1", text="Test", direction="pro"),
                Subclaim(id="sc2", text="Test", direction="con"),
                Subclaim(id="sc3", text="Test", direction="pro"),
                Subclaim(id="sc4", text="Test", direction="con")
            ],
            key_variables=[],
            search_seeds={"pro": [], "con": [], "general": []},
            decision_criteria=[],
            reasoning_trace="Too short"
        )

        # Should not raise, but log warning
        validate_cot_output(output, strict=False)


class TestEvidenceQuality:
    """Test evidence quality validation"""

    def test_valid_evidence_passes(self):
        """Test that high-quality evidence passes validation"""
        evidence = Evidence(
            subclaim_id="sc1",
            title="High Quality Evidence",
            url="https://example.com/article",
            published_date=date(2025, 1, 15),
            source_type="primary",
            claim_summary=(
                "This is a detailed claim summary that is at least 50 characters long "
                "and provides specific, falsifiable information about the topic."
            ),
            support="pro",
            verifiability_score=0.9,
            independence_score=0.85,
            recency_score=1.0,
            estimated_LLR=1.5,
            extraction_notes="Strong evidence from primary source"
        )

        # Should not raise
        validate_evidence_quality(evidence)

    def test_low_verifiability_fails(self):
        """Test that low verifiability score fails"""
        evidence = Evidence(
            subclaim_id="sc1",
            title="Weak Evidence",
            url="https://example.com",
            published_date=date(2025, 1, 15),
            source_type="weak",
            claim_summary="This is a claim summary with enough characters to pass length check.",
            support="pro",
            verifiability_score=0.2,  # Too low
            independence_score=0.8,
            recency_score=1.0,
            estimated_LLR=0.1,
            extraction_notes="Weak source"
        )

        with pytest.raises(ValueError) as exc_info:
            validate_evidence_quality(evidence, min_verifiability=0.5)

        assert "verifiability_score too low" in str(exc_info.value)

    def test_missing_url_fails(self):
        """Test that missing URL fails when required"""
        evidence = Evidence(
            subclaim_id="sc1",
            title="Evidence without URL",
            url="unknown",  # Missing URL
            published_date=date(2025, 1, 15),
            source_type="secondary",
            claim_summary="This is a claim summary with enough characters to pass length check.",
            support="pro",
            verifiability_score=0.8,
            independence_score=0.8,
            recency_score=1.0,
            estimated_LLR=0.5,
            extraction_notes="Test"
        )

        with pytest.raises(ValueError) as exc_info:
            validate_evidence_quality(evidence, require_url=True)

        assert "URL is missing" in str(exc_info.value)

    def test_short_claim_summary_fails(self):
        """Test that short claim summary fails"""
        evidence = Evidence(
            subclaim_id="sc1",
            title="Evidence",
            url="https://example.com",
            published_date=date(2025, 1, 15),
            source_type="secondary",
            claim_summary="Too short",  # Only 9 chars
            support="pro",
            verifiability_score=0.8,
            independence_score=0.8,
            recency_score=1.0,
            estimated_LLR=0.5,
            extraction_notes="Test"
        )

        with pytest.raises(ValueError) as exc_info:
            validate_evidence_quality(evidence)

        assert "claim_summary too short" in str(exc_info.value)


class TestProbabilityValidation:
    """Test probability validation"""

    def test_valid_probabilities_pass(self):
        """Test that valid probabilities pass"""
        validate_probability(0.0, "p_min")
        validate_probability(0.5, "p_neutral")
        validate_probability(1.0, "p_max")
        validate_probability(0.75, "p_test")

    def test_negative_probability_fails(self):
        """Test that negative probability fails"""
        with pytest.raises(ValueError) as exc_info:
            validate_probability(-0.1, "prior_p")

        assert "must be in range [0.0, 1.0]" in str(exc_info.value)

    def test_probability_above_one_fails(self):
        """Test that probability > 1 fails"""
        with pytest.raises(ValueError) as exc_info:
            validate_probability(1.5, "prior_p")

        assert "must be in range [0.0, 1.0]" in str(exc_info.value)

    def test_non_numeric_probability_fails(self):
        """Test that non-numeric probability fails"""
        with pytest.raises(ValueError) as exc_info:
            validate_probability("0.5", "prior_p")

        assert "must be numeric" in str(exc_info.value)


class TestLLRValidation:
    """Test LLR validation"""

    def test_valid_llrs_pass(self):
        """Test that reasonable LLRs pass"""
        validate_llr(0.0)
        validate_llr(1.0)
        validate_llr(-1.0)
        validate_llr(3.0)
        validate_llr(-3.0)

    def test_extreme_positive_llr_fails(self):
        """Test that extreme positive LLR fails"""
        with pytest.raises(ValueError) as exc_info:
            validate_llr(10.0, max_magnitude=5.0)

        assert "LLR magnitude too large" in str(exc_info.value)

    def test_extreme_negative_llr_fails(self):
        """Test that extreme negative LLR fails"""
        with pytest.raises(ValueError) as exc_info:
            validate_llr(-10.0, max_magnitude=5.0)

        assert "LLR magnitude too large" in str(exc_info.value)

    def test_non_numeric_llr_fails(self):
        """Test that non-numeric LLR fails"""
        with pytest.raises(ValueError) as exc_info:
            validate_llr("1.0")

        assert "LLR must be numeric" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
