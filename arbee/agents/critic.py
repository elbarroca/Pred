"""
Critic Agent - Quality control and correlation detection
Reviews evidence for dependencies, biases, and coverage gaps
"""
from typing import Type, List, Dict, Any, Optional
from arbee.agents.base import BaseAgent
from arbee.agents.schemas import CriticOutput, Evidence
from pydantic import BaseModel


class CriticAgent(BaseAgent):
    """
    Critic Agent - Reviews research quality and detects correlations

    Responsibilities:
    1. Identify duplicate or highly correlated evidence
    2. Detect over-represented sources (echo chambers)
    3. Find missing topics or coverage gaps
    4. Generate correlation warnings for Analyst
    5. Suggest follow-up search seeds

    This agent "thinks deeply" by:
    - Analyzing source diversity and independence
    - Detecting subtle correlations (same underlying data sources)
    - Identifying systematic biases in evidence collection
    - Recognizing what's NOT covered (negative space)
    - Proposing targeted searches to fill gaps
    """

    def get_system_prompt(self) -> str:
        """System prompt for critical analysis"""
        return """You are the Critic Agent in ARBEE, an autonomous Bayesian research system.

Your role is to review gathered evidence for quality, balance, and independence.

## Core Responsibilities

1. **Detect Duplicate/Correlated Evidence**
   - Same source cited multiple times
   - Different articles reporting the same underlying data
   - Echo chamber effect (multiple outlets repeating same claim)
   - Shared authorship or institutional bias

2. **Identify Over-Represented Sources**
   - Are we relying too heavily on one news outlet?
   - Are multiple evidence items from the same author/organization?
   - Is there geographic or ideological bias in sources?

3. **Find Missing Topics**
   - What aspects of the question aren't covered?
   - What evidence types are underrepresented?
   - What time periods or geographic regions are missing?

4. **Assess Evidence Quality**
   - How reliable are the sources?
   - What is the recency of the information?
   - Are there any obvious biases or conflicts of interest?

5. **Propose Follow-up Research**
   - What additional searches would improve coverage?
   - What specific questions need answering?
   - What types of evidence are most needed?

## Guidelines

- **Be thorough**: Check every evidence item for potential issues
- **Be specific**: Point out exact problems and solutions
- **Consider context**: Understand how evidence relates to the market question
- **Balance criticism**: Don't just find problems, suggest improvements
- **Prioritize issues**: Focus on the most important gaps and biases

Remember: Your analysis helps ensure the research is comprehensive, unbiased, and reliable.

## CHAIN-OF-THOUGHT CRITICAL ANALYSIS REQUIRED

You MUST analyze step-by-step:

**Step 1: Evidence Inventory**
- How many evidence items total?
- How many PRO vs CON vs NEUTRAL?
- What's the distribution of source types?

**Step 2: Source Independence Check**
- Group evidence by source domain/author
- Which sources appear multiple times?
- Do different evidence items cite the same underlying data?

**Step 3: Correlation Detection**
- Which evidence items are similar or overlapping?
- Which ones likely depend on the same underlying information?
- What clusters of correlated evidence exist?

**Step 4: Coverage Gap Analysis**
- Review the original subclaims from Planner
- Which subclaims have strong evidence? Which are weak?
- What topics are completely missing?
- What perspectives are underrepresented?

**Step 5: Quality Assessment**
- Are sources credible and verifiable?
- Is information recent enough?
- Are there obvious biases or conflicts of interest?

**Step 6: Improvement Recommendations**
- What specific searches would fill the biggest gaps?
- What additional evidence would strengthen weak areas?
- What would make this analysis more balanced?

Include your complete analysis process in the "analysis_process" field.

You must respond with valid JSON in exactly this format:
{{
  "duplicate_clusters": [["evidence_id1", "evidence_id2"]],
  "missing_topics": ["topic1", "topic2"],
  "over_represented_sources": ["source1", "source2"],
  "correlation_warnings": [{{"cluster": ["id1", "id2"], "note": "reason"}}],
  "follow_up_search_seeds": ["query1", "query2"],
  "analysis_process": "Step 1: Evidence Inventory - Found X evidence items...[detailed analysis]"
}}"""
    def get_output_schema(self) -> Type[BaseModel]:
        """Return CriticOutput schema"""
        return CriticOutput

    def get_human_prompt(self) -> str:
        """Human prompt for evidence critique"""
        return """Analyze the following evidence for quality, balance, and completeness.

Evidence Items ({evidence_count} total):
{evidence_items}

Planner Output (Original Research Plan):
{planner_output}

Market Question: {market_question}

CRITICAL: You MUST provide your complete chain-of-thought analysis in the "analysis_process" field FIRST,
then fill in all other required output fields based on your analysis.

Your analysis_process should include:
Step 1: Evidence Inventory - Count and categorize all evidence
Step 2: Source Independence Check - Group by source/domain
Step 3: Correlation Detection - Identify similar/overlapping items
Step 4: Coverage Gap Analysis - Compare to original subclaims
Step 5: Quality Assessment - Evaluate credibility and bias
Step 6: Improvement Recommendations - Suggest follow-up searches

Think step-by-step before making judgments."""

    async def critique(
        self,
        evidence_items: List[Evidence],
        planner_output: Dict[str, Any],
        market_question: str
    ) -> CriticOutput:
        """
        Analyze evidence for quality, balance, and completeness

        Args:
            evidence_items: List of evidence items to analyze
            planner_output: Original research plan
            market_question: The market question being analyzed

        Returns:
            CriticOutput with analysis and recommendations

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if evidence_items is None or not isinstance(evidence_items, list):
            raise ValueError(f"evidence_items must be list, got {type(evidence_items)}")

        if planner_output is None or not isinstance(planner_output, dict):
            raise ValueError(f"planner_output must be dict, got {type(planner_output)}")

        if market_question is None or not isinstance(market_question, str):
            raise ValueError(f"market_question must be string, got {type(market_question)}")

        if not market_question.strip():
            raise ValueError("market_question cannot be empty")

        input_data = {
            "evidence_items": evidence_items,
            "evidence_count": len(evidence_items),
            "planner_output": planner_output,
            "market_question": market_question
        }

        self.logger.info(f"Critiquing {len(evidence_items)} evidence items")

        result = await self.invoke(input_data)

        self.logger.info(
            f"Critique complete: {len(result.duplicate_clusters)} clusters, "
            f"{len(result.missing_topics)} missing topics"
        )

        return result

    def validate_output(self, output: BaseModel) -> tuple[bool, Optional[str]]:
        """
        Validate CriticOutput for systematic analysis process

        Checks:
        1. Correlation clusters have at least 2 evidence items
        2. Analysis process is present and detailed
        3. Output fields are properly structured
        """
        # Base validation
        is_valid, feedback = super().validate_output(output)
        if not is_valid:
            return is_valid, feedback

        # Type check
        if not isinstance(output, CriticOutput):
            return False, f"Expected CriticOutput, got {type(output)}"

        issues = []

        # Check correlation warning clusters
        for i, warning in enumerate(output.correlation_warnings):
            if not warning.cluster or len(warning.cluster) < 2:
                issues.append(
                    f"Correlation warning {i+1} has cluster with <2 items: {warning.cluster}. "
                    f"Clusters should contain at least 2 correlated evidence IDs."
                )

        # Check duplicate clusters
        for i, cluster in enumerate(output.duplicate_clusters):
            if len(cluster) < 2:
                issues.append(
                    f"Duplicate cluster {i+1} has <2 items: {cluster}. "
                    f"Clusters should contain at least 2 duplicate evidence IDs."
                )

        # Check analysis process (if field exists)
        if hasattr(output, 'analysis_process'):
            if not output.analysis_process or len(output.analysis_process) < 100:
                issues.append(
                    "analysis_process is too short or empty - must provide detailed step-by-step critical analysis"
                )

        if issues:
            feedback_msg = "Critique validation issues:\n" + "\n".join(issues)
            return False, feedback_msg

        # All validation passed
        return True, None
