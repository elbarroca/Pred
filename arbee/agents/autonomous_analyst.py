"""
Autonomous AnalystAgent with Bayesian Tools
Performs Bayesian aggregation with autonomous validation and sensitivity analysis.
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.tools import BaseTool

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import (
    AnalystOutput,
    EvidenceSummaryItem,
    SensitivityScenario,
    CorrelationAdjustment,
)
from arbee.tools.bayesian import (
    bayesian_calculate_tool,
    sensitivity_analysis_tool,
    validate_llr_calibration_tool,
)


class AutonomousAnalystAgent(AutonomousReActAgent):
    """
    Bayesian aggregation with validation, correlation handling, and sensitivity analysis.

    Flow:
      1) Prepare evidence (LLRs + scores)
      2) Validate LLR calibration
      3) Aggregate via bayesian_calculate_tool
      4) Run sensitivity_analysis_tool
      5) Verify completeness & ranges
    """

    def __init__(self, max_sensitivity_range: float = 0.3, **kwargs):
        """
        Args:
            max_sensitivity_range: Max acceptable probability spread in sensitivity analysis.
        """
        assert 0.0 < max_sensitivity_range <= 1.0, "max_sensitivity_range must be in (0,1]"
        super().__init__(**kwargs)
        self.max_sensitivity_range = max_sensitivity_range

    def get_system_prompt(self) -> str:
        """System prompt for autonomous Bayesian analysis."""
        return f"""You are an Autonomous Analyst Agent in POLYSEER.

Your mission: Perform rigorous Bayesian aggregation of evidence.

## Available Tools

1. **validate_llr_calibration_tool** - Validate evidence LLR calibration
   - Use this to check each evidence item's LLR is properly calibrated
   - Input: llr, source_type
   - Returns: is_valid, expected_range, feedback

2. **bayesian_calculate_tool** - Perform Bayesian aggregation
   - Use this to calculate posterior probability from prior and evidence
   - Input: prior_p, evidence_items (with LLR and scores), correlation_clusters
   - Returns: p_bayesian, log_odds, evidence_summary, correlation_adjustments
   - **Results are AUTOMATICALLY stored in intermediate_results**

3. **sensitivity_analysis_tool** - Test robustness of conclusions
   - Use this to check how sensitive results are to assumptions
   - Input: prior_p, evidence_items
   - Returns: List of scenarios with resulting probabilities
   - **Results are AUTOMATICALLY stored in intermediate_results['sensitivity_analysis']**

## Your Reasoning Process

**Step 1: Prepare Evidence**
- Extract all evidence items from input
- If evidence_items is empty (0 items):
  - Use prior p0 as p_bayesian (no update)
  - Still run sensitivity analysis with p0
- Ensure each has: id, LLR, verifiability_score, independence_score, recency_score
- Count total evidence items
- Separate by support direction (pro vs con)

**Step 2: Validate LLR Calibration**
- For each evidence item, use validate_llr_calibration_tool
- Check LLR matches source_type calibration ranges:
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
- Results automatically stored in intermediate_results:
  - p_bayesian, log_odds_prior, log_odds_posterior
  - evidence_summary, correlation_adjustments

**Step 4: Run Sensitivity Analysis**
- Use sensitivity_analysis_tool
- Check how p_bayesian changes with:
  - ±25% LLR adjustment
  - Prior ±0.1 adjustment
  - Different correlation assumptions
- Assess robustness: Is range < {self.max_sensitivity_range} (30%)?
- Results automatically stored

**Step 5: Verify Completion**
- Check you have:
  - p_bayesian in valid range [0.01, 0.99]
  - evidence_summary for all items (empty list OK if 0 evidence)
  - sensitivity_analysis with ≥2 scenarios
  - correlation_adjustments present
- If any missing, investigate and ensure tool calls succeeded

## Output Format

Automatically stored in intermediate_results:
- p0: Prior probability
- log_odds_prior: Log-odds of prior
- p_bayesian: Posterior probability
- log_odds_posterior: Log-odds of posterior
- evidence_summary: List of evidence summary dicts
- sensitivity_analysis: List of scenario dicts
- correlation_adjustments: Dict describing adjustments

## Quality Standards

- **Rigorous**: Validate all LLRs before using them
- **Transparent**: Evidence summary shows all calculations
- **Robust**: Sensitivity analysis tests assumptions
- **Complete**: All core outputs present

Remember: You're NOT doing the math yourself - the tools do that. Your job is to:
1. Validate inputs
2. Call tools correctly
3. Check robustness
4. Verify completeness
"""

    def get_tools(self) -> List[BaseTool]:
        """Return analysis tools."""
        return [validate_llr_calibration_tool, bayesian_calculate_tool, sensitivity_analysis_tool]

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Complete when:
          - p_bayesian present and within [0.01, 0.99]
          - evidence_summary present (list, possibly empty)
          - sensitivity_analysis present (≥2 scenarios, or ≥1 if 0 evidence)
          - correlation_adjustments present
        """
        results = state.get("intermediate_results", {})
        task_input = state.get("task_input", {})
        evidence_items = task_input.get("evidence_items", [])
        is_zero_evidence = len(evidence_items) == 0

        # Attempt recovery if tool wrote output but auto-store failed.
        if "p_bayesian" not in results:
            tool_calls = state.get("tool_calls", [])
            bayes_calls = [t for t in tool_calls if getattr(t, "tool_name", "") == "bayesian_calculate_tool"]
            if bayes_calls:
                last = bayes_calls[-1]
                try:
                    import json
                    payload = last.tool_output
                    data = json.loads(payload) if isinstance(payload, str) else payload
                    if isinstance(data, dict) and "p_bayesian" in data:
                        results["p_bayesian"] = data.get("p_bayesian")
                        results["p0"] = data.get("p0", task_input.get("prior_p", 0.5))
                        results["log_odds_prior"] = data.get("log_odds_prior", 0.0)
                        results["log_odds_posterior"] = data.get("log_odds_posterior", 0.0)
                except Exception:
                    pass

            if "p_bayesian" not in results:
                return False

        p_bayesian = results.get("p_bayesian", 0.5)
        if not (0.01 <= p_bayesian <= 0.99):
            return False

        if "evidence_summary" not in results:
            return False

        min_scenarios = 1 if is_zero_evidence else 2
        if len(results.get("sensitivity_analysis", [])) < min_scenarios:
            return False

        if "correlation_adjustments" not in results:
            return False

        return True

    async def extract_final_output(self, state: AgentState) -> AnalystOutput:
        """Convert intermediate_results into AnalystOutput."""
        results = state.get("intermediate_results", {})
        evidence_summary = self._as_evidence_summary(results.get("evidence_summary", []))
        sensitivity_analysis = self._as_sensitivity_list(results.get("sensitivity_analysis", []))
        p_bayesian = results.get("p_bayesian", 0.5)

        p_low, p_high, conf = self._compute_ci_from_sensitivity(p_bayesian, sensitivity_analysis)

        corr_adj = results.get("correlation_adjustments", {})
        correlation_adjustments = (
            corr_adj
            if isinstance(corr_adj, CorrelationAdjustment)
            else CorrelationAdjustment(
                method=(corr_adj.get("method") if isinstance(corr_adj, dict) else "none") or "none",
                details=(corr_adj.get("details") if isinstance(corr_adj, dict) else "No correlation detected") or "No correlation detected",
            )
        )

        return AnalystOutput(
            p0=results.get("p0", 0.5),
            log_odds_prior=results.get("log_odds_prior", 0.0),
            p_bayesian=p_bayesian,
            p_bayesian_low=p_low,
            p_bayesian_high=p_high,
            confidence_level=conf,
            log_odds_posterior=results.get("log_odds_posterior", 0.0),
            p_neutral=results.get("p_neutral", 0.5),
            calculation_steps=results.get("calculation_steps", []),
            evidence_summary=evidence_summary,
            correlation_adjustments=correlation_adjustments,
            sensitivity_analysis=sensitivity_analysis,
        )

    async def analyze(
        self,
        prior_p: float,
        evidence_items: List[Any],
        critic_output: Any,
        market_question: str,
    ) -> AnalystOutput:
        """
        Run autonomous Bayesian analysis and return AnalystOutput.
        """
        assert isinstance(prior_p, (int, float)) and 0.0 <= prior_p <= 1.0, "prior_p must be in [0,1]"
        assert isinstance(evidence_items, list), "evidence_items must be a list"
        assert isinstance(market_question, str) and market_question.strip(), "market_question is required"

        # Extract correlation clusters from critic output.
        if hasattr(critic_output, "correlation_warnings"):
            correlation_clusters = [w.cluster for w in critic_output.correlation_warnings]
        elif isinstance(critic_output, dict):
            correlation_clusters = [w.get("cluster", []) for w in critic_output.get("correlation_warnings", [])]
        else:
            correlation_clusters = []

        return await self.run(
            task_description="Perform Bayesian aggregation with validation and sensitivity analysis",
            task_input={
                "prior_p": prior_p,
                "evidence_items": evidence_items,
                "correlation_clusters": correlation_clusters,
                "market_question": market_question,
            },
        )

    # -------------------------
    # Private helpers
    # -------------------------
    @staticmethod
    def _as_evidence_summary(items: List[Any]) -> List[EvidenceSummaryItem]:
        out: List[EvidenceSummaryItem] = []
        for it in items:
            if isinstance(it, dict):
                out.append(EvidenceSummaryItem(**it))
        return out

    @staticmethod
    def _as_sensitivity_list(items: List[Any]) -> List[SensitivityScenario]:
        out: List[SensitivityScenario] = []
        for it in items:
            if isinstance(it, dict):
                out.append(SensitivityScenario(**it))
        return out

    def _compute_ci_from_sensitivity(
        self, p_bayesian: float, scenarios: List[SensitivityScenario]
    ) -> tuple[float, float, float]:
        if scenarios and len(scenarios) > 1:
            probs = [s.p for s in scenarios if isinstance(s.scenario, str) and s.scenario.lower() != "baseline"]
            if probs:
                return min(probs), max(probs), 0.80
            return max(0.01, p_bayesian - 0.10), min(0.99, p_bayesian + 0.10), 0.50
        return max(0.01, p_bayesian - 0.15), min(0.99, p_bayesian + 0.15), 0.50