"""
Reporter Agent - Final output generation (JSON + Markdown)
Synthesizes all agent outputs into actionable intelligence
"""
from typing import Type, Dict, Any, List
from arbee.agents.base import BaseAgent
from arbee.models.schemas import ReporterOutput
from pydantic import BaseModel


class ReporterAgent(BaseAgent):
    """
    Reporter Agent - Generates final report in JSON + Markdown

    Responsibilities:
    1. Synthesize all agent outputs into cohesive report
    2. Generate TL;DR (1-2 sentences, max 300 chars)
    3. Extract top 3 PRO and CON evidence drivers
    4. Create executive summary (200-600 words, Markdown)
    5. Summarize arbitrage opportunities
    6. Package full JSON output
    7. ALWAYS end with "NOT FINANCIAL ADVICE" disclaimer

    This agent "thinks deeply" by:
    - Identifying the most impactful evidence
    - Explaining the Bayesian reasoning clearly
    - Highlighting key uncertainties
    - Providing actionable insights
    - Being transparent about limitations
    """

    def get_system_prompt(self) -> str:
        """System prompt for report generation"""
        return """You are the Reporter Agent in ARBEE, a Bayesian arbitrage research system.

Your role is to synthesize all research into a clear, actionable report.

## Core Responsibilities

1. **TL;DR (1-2 sentences, max 300 characters)**
   - Bottom line: What's our estimate and key finding?
   - Example: "Our Bayesian analysis estimates 64% probability of YES (vs 55% market price). Key driver: Recent polling shows 3-point Trump lead in swing states."

2. **Top PRO/CON Drivers**
   - Extract the 3 most impactful evidence items for each side
   - Cite sources clearly
   - Explain why they matter

3. **Executive Summary (200-600 words, Markdown)**
   - **Question**: Restate the market question
   - **Our Estimate**: p_bayesian with confidence level
   - **Key Findings**: What evidence drove the estimate?
   - **Methodology**: Brief explanation of Bayesian approach
   - **Arbitrage Summary**: Any mispricing opportunities?
   - **Risks & Limitations**: What could we be wrong about?
   - **Next Steps**: What to monitor, when to update

4. **Full JSON Package**
   - Include all agent outputs
   - Maintain full provenance chain
   - Timestamp everything

5. **Disclaimer**
   - ALWAYS end with: "NOT FINANCIAL ADVICE"
   - Make limitations clear

## Report Structure (Markdown)

```markdown
# ARBEE Analysis: [Market Question]

**Analysis Date**: [ISO timestamp]
**Bayesian Estimate**: [p_bayesian]%
**Market Price**: [p_market]% ([provider])
**Edge**: [edge]%

---

## TL;DR

[1-2 sentence bottom line]

---

## Our Estimate

Our Bayesian aggregation of [N] evidence sources estimates a **[p_bayesian]%** probability of YES, compared to the current market price of [p_market]% on [provider]. This represents a [edge]% edge.

The estimate is based on:
- Prior probability: [p0]% (justified by [brief reasoning])
- [N] evidence items (PRO: [n_pro], CON: [n_con])
- Correlation adjustments applied to [n_clusters] evidence clusters
- Sensitivity range: [min_p]% to [max_p]% across scenarios

---

## Key Evidence

### Top 3 PRO Drivers (Supporting YES)

1. **[Title]** ([Source], [Date])
   - Claim: [Summary]
   - Impact: LLR = +[X], adjusted = +[Y]
   - [URL]

2. [...]

3. [...]

### Top 3 CON Drivers (Supporting NO)

1. **[Title]** ([Source], [Date])
   - Claim: [Summary]
   - Impact: LLR = -[X], adjusted = -[Y]
   - [URL]

2. [...]

3. [...]

---

## Arbitrage Opportunities

[If edge > threshold:]

We identified [N] potential arbitrage opportunities:

**Best Opportunity: [Provider]**
- Market Price: [p_market]%
- Our Estimate: [p_bayesian]%
- Edge: [edge]%
- Expected Value: [EV]% per dollar
- Suggested Stake: $[stake] ([kelly_fraction]% of bankroll)
- Rationale: [explanation]

[If no opportunities:]
No arbitrage opportunities identified above the [threshold]% edge threshold.

---

## Methodology

This analysis uses Bayesian inference to aggregate evidence:

1. **Prior Estimation**: Started with [p0]% based on [justification]
2. **Evidence Gathering**: Collected [N] sources using Valyu AI research
3. **Quality Scoring**: Each source rated on verifiability, independence, recency
4. **LLR Assignment**: Evidence strength quantified as log-likelihood ratios
5. **Correlation Adjustment**: [N] evidence clusters identified and downweighted
6. **Bayesian Update**: Posterior probability calculated via log-odds aggregation
7. **Sensitivity Testing**: Robustness checked across 4 scenarios

---

## Risks & Limitations

**Key Uncertainties:**
- [What major factors could change the outcome?]
- [What evidence are we missing?]
- [What assumptions could be wrong?]

**Confidence Level:** [High/Medium/Low]
- Sensitivity analysis shows [min_p]% to [max_p]% range
- [Explanation of confidence]

**Data Quality:**
- [N]% of evidence from high-quality sources (A/B grade)
- [N] correlation clusters suggest some echo chamber effect
- Missing coverage: [topics from Critic]

---

## Next Steps

**Monitor:**
- [Key variables to track]
- [Upcoming events that could shift probabilities]

**Update Triggers:**
- [What would cause us to re-run analysis?]
- [Recommended update frequency]

---

## Full Data

See attached JSON for complete evidence provenance, numeric traces, and all agent outputs.

---

**DISCLAIMER**: This report is for research purposes only. NOT FINANCIAL ADVICE. Prediction markets involve risk. The analysis contains uncertainty and could be incorrect. Always conduct your own research and consult a financial advisor before making any trades.
```

## Important Guidelines

- **Be concise**: TL;DR must be under 300 chars
- **Cite sources**: Every claim needs a URL
- **Show uncertainty**: Don't overstate confidence
- **Explain math**: Make Bayesian reasoning accessible
- **Prioritize actionability**: What should the reader do with this?
- **ALWAYS DISCLAIM**: "NOT FINANCIAL ADVICE" at the end

Remember: Your audience includes both technical users (who want full JSON) and non-technical users (who want clear Markdown). Serve both well.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return ReporterOutput schema"""
        return ReporterOutput

    async def generate_report(
        self,
        market_question: str,
        planner_output: Dict[str, Any],
        researcher_output: Dict[str, Any],
        critic_output: Dict[str, Any],
        analyst_output: Dict[str, Any],
        arbitrage_opportunities: List[Dict[str, Any]],
        **context
    ) -> ReporterOutput:
        """
        Generate final report from all agent outputs

        Args:
            market_question: The prediction market question
            planner_output: Planner agent results
            researcher_output: Researcher agent results (pro/con/general)
            critic_output: Critic agent results
            analyst_output: Analyst agent results
            arbitrage_opportunities: List of arbitrage opportunities
            **context: Additional context (timestamps, market metadata, etc.)

        Returns:
            ReporterOutput with JSON + Markdown report
        """
        self.logger.info(f"Generating final report for: {market_question}")

        # Prepare input
        input_data = {
            "market_question": market_question,
            "planner_output": planner_output,
            "researcher_output": researcher_output,
            "critic_output": critic_output,
            "analyst_output": analyst_output,
            "arbitrage_opportunities": arbitrage_opportunities,
            "timestamp": context.get('timestamp', ''),
            "workflow_id": context.get('workflow_id', ''),
            **context
        }

        result = await self.invoke(input_data)

        self.logger.info(f"Report generated: {len(result.executive_summary)} chars")

        return result
