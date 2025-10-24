"""
Analyst Agent - Bayesian aggregation and probability estimation
Integrates BayesianCalculator for rigorous mathematical analysis
"""
from typing import Type, List, Dict, Any, Optional
from arbee.agents.base import BaseAgent
from arbee.agents.schemas import AnalystOutput, CriticOutput, Evidence
from arbee.utils.bayesian import BayesianCalculator
from pydantic import BaseModel


class AnalystAgent(BaseAgent):
    """
    Analyst Agent - Performs Bayesian aggregation of evidence

    Responsibilities:
    1. Aggregate evidence using log-likelihood ratios (LLRs)
    2. Apply correlation adjustments based on Critic warnings
    3. Calculate posterior probability (p_bayesian)
    4. Run sensitivity analysis on key assumptions
    5. Generate evidence summary with contributions

    Mathematical Process:
    1. Convert prior to log-odds: log_odds_prior = ln(p0 / (1-p0))
    2. Adjust each LLR: adjusted_LLR = LLR × verifiability × independence × recency
    3. Apply correlation shrinkage: shrinkage = 1/sqrt(cluster_size)
    4. Sum adjusted LLRs: log_odds_posterior = log_odds_prior + Σ(adjusted_LLR)
    5. Convert back: p_bayesian = exp(log_odds) / (1 + exp(log_odds))

    This agent "thinks deeply" by:
    - Following rigorous Bayesian mathematics
    - Properly handling correlated evidence
    - Testing robustness with sensitivity analysis
    - Explaining each adjustment transparently
    - Providing numeric trace for auditability
    """

    def __init__(self, **kwargs):
        """Initialize with Bayesian calculator"""
        super().__init__(**kwargs)
        self.calculator = BayesianCalculator()

    def get_system_prompt(self) -> str:
        """System prompt for EXPLAINING pre-calculated Bayesian results"""
        return """You are the Analyst Agent in POLYSEER.

IMPORTANT: You DO NOT perform calculations. The BayesianCalculator has already done all math.

Your ONLY role: EXPLAIN the pre-calculated Bayesian results in clear language.

## What You Receive

You will be given ALREADY CALCULATED values:
- Prior probability (p0)
- Log-odds prior
- Evidence summary with adjusted LLRs
- Total adjusted LLR
- Posterior log-odds
- Final p_bayesian
- Sensitivity analysis results

## Your Job

Generate a "calculation_steps" list that explains what these numbers mean.

Each step should:
1. Reference the actual calculated values
2. Explain what the calculation represents
3. Be written in plain language

## Example Output Format

Your output should contain a calculation_steps array with explanations like:
Step 1 - Started with prior p0 = 0.50 (neutral position), which converts to log-odds = 0.0
Step 2 - Processed 10 evidence items, adjusting each LLR by quality scores (verifiability × independence × recency)
Step 3 - Applied correlation shrinkage (1/sqrt(n)) to 2 clusters to avoid double-counting related evidence
Step 4 - Summed all adjusted LLRs: total = +1.85 (net evidence points toward YES)
Step 5 - Updated log-odds: 0.0 + 1.85 = 1.85
Step 6 - Converted to probability: p_bayesian = exp(1.85)/(1+exp(1.85)) = 86.4%
Step 7 - Sensitivity check shows result is robust (ranges from 79% to 91% across scenarios)

## Critical Rules

❌ DO NOT recalculate anything
❌ DO NOT perform mathematical operations
✅ DO use the exact values provided
✅ DO explain what each step means
✅ DO make it understandable to non-technical users

Your output will be combined with the ACTUAL calculated values from BayesianCalculator.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return AnalystOutput schema"""
        return AnalystOutput

    def get_human_prompt(self) -> str:
        """Human prompt for Bayesian analysis explanation"""
        return """You are explaining ALREADY CALCULATED Bayesian results in clear language.

Market Question: {market_question}

Prior Probability (p0): {prior_p}

Evidence Items Analyzed: {evidence_items}

Bayesian Calculation Results: {bayesian_result}

Sensitivity Analysis: {sensitivity_results}

Correlation Adjustments: {correlation_adjustments}

CRITICAL: The math is already done. Your job is to EXPLAIN these results.

Generate calculation_steps that explain what these numbers mean in plain language.
Each step should reference the actual calculated values and explain what they represent.

Example steps:
Step 1 - Started with prior p0 = X.XX (neutral position)
Step 2 - Processed Y evidence items, adjusting each by quality scores
Step 3 - Applied correlation shrinkage to Z clusters
Step 4 - Total adjusted LLR = +/-N.NN
Step 5 - Final p_bayesian = XX.X%

Make it understandable to non-technical users."""

    async def analyze(
        self,
        prior_p: float,
        evidence_items: List[Evidence],
        critic_output: CriticOutput,
        market_question: str
    ) -> AnalystOutput:
        """
        Perform Bayesian aggregation of evidence

        Args:
            prior_p: Prior probability from Planner
            evidence_items: All evidence from Researchers
            critic_output: Correlation warnings from Critic
            market_question: Main question (for context)

        Returns:
            AnalystOutput with p_bayesian and full analysis

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if prior_p is None or not isinstance(prior_p, (int, float)):
            raise ValueError(f"prior_p must be numeric, got {type(prior_p)}")

        if not (0.0 <= prior_p <= 1.0):
            raise ValueError(
                f"prior_p must be in range [0.0, 1.0], got {prior_p}"
            )

        if evidence_items is None or not isinstance(evidence_items, list):
            raise ValueError(f"evidence_items must be list, got {type(evidence_items)}")

        if market_question is None or not isinstance(market_question, str):
            raise ValueError(f"market_question must be string, got {type(market_question)}")

        if not market_question.strip():
            raise ValueError("market_question cannot be empty")

        if critic_output is None or not isinstance(critic_output, CriticOutput):
            raise ValueError(f"critic_output must be CriticOutput, got {type(critic_output)}")

        self.logger.info(
            f"Analyzing {len(evidence_items)} evidence items with prior={prior_p:.2%}"
        )

        # Convert evidence to format for calculator
        evidence_dicts = [
            {
                'id': f"ev_{i}",
                'LLR': ev.estimated_LLR,
                'verifiability_score': ev.verifiability_score,
                'independence_score': ev.independence_score,
                'recency_score': ev.recency_score
            }
            for i, ev in enumerate(evidence_items)
        ]

        # Extract correlation clusters from Critic
        correlation_clusters = [
            warning.cluster for warning in critic_output.correlation_warnings
        ]

        # Perform Bayesian calculation
        bayesian_result = self.calculator.aggregate_evidence(
            prior_p=prior_p,
            evidence_items=evidence_dicts,
            correlation_clusters=correlation_clusters
        )

        # Run sensitivity analysis
        sensitivity_results = self.calculator.sensitivity_analysis(
            prior_p=prior_p,
            evidence_items=evidence_dicts
        )

        # Prepare input for LLM (to generate explanatory output)
        input_data = {
            "market_question": market_question,
            "prior_p": prior_p,
            "evidence_items": evidence_dicts,
            "bayesian_result": bayesian_result,
            "sensitivity_results": sensitivity_results,
            "correlation_adjustments": {
                "method": "1/sqrt(n) shrinkage",
                "clusters": correlation_clusters,
                "num_clusters": len(correlation_clusters)
            }
        }

        # Get LLM to format output with explanations
        result = await self.invoke(input_data)

        self.logger.info(
            f"Analysis complete: p_bayesian={result.p_bayesian:.2%} "
            f"(prior={prior_p:.2%})"
        )

        return result

    def validate_output(self, output: BaseModel) -> tuple[bool, Optional[str]]:
        """
        Validate AnalystOutput with Bayesian calculation checks

        Checks:
        1. p_bayesian is in valid range [0.01, 0.99]
        2. calculation_steps array is present and non-empty
        3. Sensitivity analysis shows reasonable robustness
        4. Evidence summary is present
        """
        # Base validation
        is_valid, feedback = super().validate_output(output)
        if not is_valid:
            return is_valid, feedback

        # Type check
        if not isinstance(output, AnalystOutput):
            return False, f"Expected AnalystOutput, got {type(output)}"

        issues = []

        # Check p_bayesian range
        if not (0.01 <= output.p_bayesian <= 0.99):
            issues.append(f"p_bayesian ({output.p_bayesian:.2%}) is outside valid range [1%, 99%]")

        # Check calculation_steps
        if not output.calculation_steps or len(output.calculation_steps) == 0:
            issues.append("calculation_steps is empty - must provide step-by-step reasoning")
        elif len(output.calculation_steps) < 3:
            issues.append(f"calculation_steps has only {len(output.calculation_steps)} steps - need at least 3 steps explaining the Bayesian process")

        # Check evidence summary
        if not output.evidence_summary or len(output.evidence_summary) == 0:
            issues.append("evidence_summary is empty - must summarize evidence contributions")

        # Sensitivity analysis robustness check (optional, warn only)
        if output.sensitivity_analysis and len(output.sensitivity_analysis) >= 2:
            probs = [s.p for s in output.sensitivity_analysis]
            sensitivity_range = max(probs) - min(probs)
            if sensitivity_range > 0.4:  # More than 40% variation
                # This is just a warning, not a failure
                self.logger.warning(
                    f"⚠️  High sensitivity detected: probability ranges from "
                    f"{min(probs):.1%} to {max(probs):.1%} (range={sensitivity_range:.1%})"
                )

        if issues:
            feedback_msg = "Bayesian analysis validation issues:\n" + "\n".join(issues)
            return False, feedback_msg

        # All validation passed
        return True, None
