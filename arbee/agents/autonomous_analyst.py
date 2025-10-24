"""
Autonomous AnalystAgent with Bayesian Tools
Performs Bayesian aggregation with autonomous validation and sensitivity analysis
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import AnalystOutput, EvidenceSummaryItem, SensitivityScenario
from arbee.tools.bayesian import (
    bayesian_calculate_tool,
    sensitivity_analysis_tool,
    validate_llr_calibration_tool
)
from arbee.tools.output_validation import (
    validate_bayesian_calculation_tool,
    generate_probability_justification_tool
)

logger = logging.getLogger(__name__)


class AutonomousAnalystAgent(AutonomousReActAgent):
    """
    Autonomous Analyst Agent - Bayesian aggregation with validation

    Autonomous Capabilities:
    - Validates LLR calibration before aggregation
    - Performs Bayesian calculation with correlation adjustments
    - Runs sensitivity analysis automatically
    - Checks result robustness
    - Iteratively refines if validation fails

    Reasoning Flow:
    1. Prepare evidence items (extract LLRs and scores)
    2. Validate LLR calibration for each evidence item
    3. Flag miscalibrated evidence for review
    4. Perform Bayesian aggregation using bayesian_calculate_tool
    5. Run sensitivity analysis using sensitivity_analysis_tool
    6. Check if results are robust (low sensitivity)
    7. Generate calculation explanation
    8. Validate completeness, refine if needed
    """

    def __init__(
        self,
        max_sensitivity_range: float = 0.3,  # Max acceptable probability range in sensitivity
        **kwargs
    ):
        """
        Initialize Autonomous Analyst

        Args:
            max_sensitivity_range: Max acceptable range in sensitivity analysis
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.max_sensitivity_range = max_sensitivity_range

    def get_system_prompt(self) -> str:
        """System prompt for autonomous Bayesian analysis"""
        return f"""You are an Autonomous Analyst Agent in POLYSEER.

Your mission: Perform rigorous Bayesian aggregation of evidence with full validation.

## Available Tools

1. **validate_llr_calibration_tool** - Validate evidence LLR calibration
   - Use this to check each evidence item's LLR is properly calibrated
   - Input: llr, source_type
   - Returns: is_valid, expected_range, feedback

2. **bayesian_calculate_tool** - Perform Bayesian aggregation
   - Use this to calculate posterior probability from prior and evidence
   - Input: prior_p, evidence_items (with LLR and scores), correlation_clusters
   - Returns: p_bayesian, log_odds, evidence_summary, etc.

3. **sensitivity_analysis_tool** - Test robustness of conclusions
   - Use this to check how sensitive results are to assumptions
   - Input: prior_p, evidence_items
   - Returns: List of scenarios with resulting probabilities

## Your Reasoning Process

**Step 1: Prepare Evidence**
- Extract all evidence items from input
- Ensure each has: id, LLR, verifiability_score, independence_score, recency_score
- Count total evidence items
- Separate by support direction (pro vs con)

**Step 2: Validate LLR Calibration (IMPORTANT)**
- For each evidence item, use validate_llr_calibration_tool
- Check: LLR matches source_type calibration ranges
  - Primary: ±1-3
  - High-quality secondary: ±0.3-1.0
  - Secondary: ±0.1-0.5
  - Weak: ±0.01-0.2
- Flag any miscalibrated evidence
- Decide: Include with warning, or exclude?

**Step 3: Perform Bayesian Calculation**
- Use bayesian_calculate_tool with:
  - prior_p from Planner
  - All validated evidence items
  - correlation_clusters from Critic
- Get: p_bayesian, log_odds_prior, log_odds_posterior, evidence_summary

**Step 4: Run Sensitivity Analysis**
- Use sensitivity_analysis_tool
- Check how p_bayesian changes with:
  - ±25% LLR adjustment
  - Prior ±0.1 adjustment
  - Different correlation assumptions
- Assess robustness: Is range < {self.max_sensitivity_range} (30%)?

**Step 5: Generate Explanation**
- Create calculation_steps explaining the Bayesian process:
  - Step 1: Started with prior p0 = X.XX
  - Step 2: Processed N evidence items
  - Step 3: Applied correlation shrinkage to M clusters
  - Step 4: Total adjusted LLR = +/-N.NN
  - Step 5: Final p_bayesian = XX.X%
  - Step 6: Sensitivity check shows [robust/moderate/high sensitivity]

**Step 6: Validate Calculation Accuracy**
- Use validate_bayesian_calculation_tool to verify your math:
  - Pass: p0_prior, evidence_items, p_bayesian, log_odds_calculation
  - Tool will recalculate and check if it matches
  - If mismatch detected, investigate and fix
- This ensures mathematical rigor

**Step 7: Generate Probability Justification**
- Use generate_probability_justification_tool to create human-readable explanation:
  - Pass: p0_prior, p_bayesian, evidence_summary, market_question
  - Tool will generate markdown justification explaining WHY this probability
  - Includes: prior reasoning, key evidence, interpretation, confidence level
- Store justification in intermediate_results['probability_justification']

**Step 8: Validate Completeness**
- Check you have:
  - p_bayesian in valid range [0.01, 0.99]
  - calculation_steps with ≥3 steps
  - evidence_summary for all items
  - sensitivity_analysis with ≥2 scenarios
  - probability_justification generated
- If incomplete, generate missing parts

## Output Format

Store your analysis in intermediate_results with these keys:
- p0: Prior probability
- log_odds_prior: Log-odds of prior
- p_bayesian: Posterior probability
- log_odds_posterior: Log-odds of posterior
- calculation_steps: List of explanation strings
- evidence_summary: List of evidence summary dicts
- sensitivity_analysis: List of scenario dicts
- correlation_adjustments: Dict describing adjustments
- probability_justification: Human-readable markdown justification (REQUIRED)

## Quality Standards

- **Rigorous**: Validate all LLRs before using them
- **Transparent**: Clear step-by-step calculation explanation
- **Robust**: Check sensitivity to ensure reliable conclusion
- **Complete**: All required outputs present

## Important Guidelines

- **Validate first** - Check LLR calibration before aggregation
- **Use actual math** - bayesian_calculate_tool does real calculations
- **Explain clearly** - calculation_steps should be understandable to non-experts
- **Check robustness** - Run sensitivity analysis every time
- **Flag issues** - If sensitivity is high, warn about uncertainty

Remember: You're NOT doing the math yourself - the tools do that. Your job is to:
1. Validate inputs
2. Call tools correctly
3. Explain results clearly
4. Check robustness
"""

    def get_tools(self) -> List[BaseTool]:
        """Return analysis tools"""
        return [
            validate_llr_calibration_tool,
            bayesian_calculate_tool,
            sensitivity_analysis_tool,
            validate_bayesian_calculation_tool,  # Validate calculation accuracy
            generate_probability_justification_tool,  # Generate human-readable justification
        ]

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Check if analysis is complete

        Criteria:
        - Bayesian calculation performed
        - Sensitivity analysis performed
        - p_bayesian in valid range
        - Calculation steps present
        """
        results = state.get('intermediate_results', {})

        # Check if Bayesian calculation done
        if 'p_bayesian' not in results:
            self.logger.info("Bayesian calculation not yet performed")
            return False

        # Check p_bayesian range
        p_bayesian = results.get('p_bayesian', 0.5)
        if not (0.01 <= p_bayesian <= 0.99):
            self.logger.warning(f"p_bayesian {p_bayesian} outside valid range")
            return False

        # Check calculation steps
        if not results.get('calculation_steps') or len(results.get('calculation_steps', [])) < 3:
            self.logger.info("Need more calculation steps")
            return False

        # Check sensitivity analysis
        if not results.get('sensitivity_analysis'):
            self.logger.info("Sensitivity analysis not yet performed")
            return False

        # Check probability justification (REQUIRED)
        if not results.get('probability_justification'):
            self.logger.info("Probability justification not yet generated")
            return False

        self.logger.info(f"✅ Analysis complete: p_bayesian={p_bayesian:.2%}")
        return True

    async def extract_final_output(self, state: AgentState) -> AnalystOutput:
        """Extract AnalystOutput from final state"""
        results = state.get('intermediate_results', {})

        # Build evidence summary
        evidence_summary = []
        for item_data in results.get('evidence_summary', []):
            if isinstance(item_data, dict):
                evidence_summary.append(EvidenceSummaryItem(**item_data))

        # Build sensitivity analysis
        sensitivity_analysis = []
        for scenario_data in results.get('sensitivity_analysis', []):
            if isinstance(scenario_data, dict):
                sensitivity_analysis.append(SensitivityScenario(**scenario_data))

        output = AnalystOutput(
            p0=results.get('p0', 0.5),
            log_odds_prior=results.get('log_odds_prior', 0.0),
            p_bayesian=results.get('p_bayesian', 0.5),
            log_odds_posterior=results.get('log_odds_posterior', 0.0),
            p_neutral=results.get('p_neutral', 0.5),
            calculation_steps=results.get('calculation_steps', []),
            evidence_summary=evidence_summary,
            correlation_adjustments=results.get('correlation_adjustments', {}),
            sensitivity_analysis=sensitivity_analysis
        )

        self.logger.info(
            f"📤 Analysis complete: p={output.p_bayesian:.2%} "
            f"(prior={output.p0:.2%})"
        )

        return output

    async def analyze(
        self,
        prior_p: float,
        evidence_items: List[Any],
        critic_output: Any,
        market_question: str
    ) -> AnalystOutput:
        """
        Perform autonomous Bayesian analysis

        Args:
            prior_p: Prior probability from Planner
            evidence_items: All evidence from Researchers
            critic_output: Correlation warnings from Critic
            market_question: Market question

        Returns:
            AnalystOutput with p_bayesian and full analysis
        """
        # Extract correlation clusters from critic output
        correlation_clusters = []
        if hasattr(critic_output, 'correlation_warnings'):
            correlation_clusters = [w.cluster for w in critic_output.correlation_warnings]
        elif isinstance(critic_output, dict):
            warnings = critic_output.get('correlation_warnings', [])
            correlation_clusters = [w.get('cluster', []) for w in warnings]

        return await self.run(
            task_description="Perform Bayesian aggregation with validation and sensitivity analysis",
            task_input={
                'prior_p': prior_p,
                'evidence_items': evidence_items,
                'correlation_clusters': correlation_clusters,
                'market_question': market_question
            }
        )
