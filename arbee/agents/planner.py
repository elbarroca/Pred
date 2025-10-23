"""
Planner Agent - First agent in POLYSEER workflow
Breaks down market questions into structured research tasks
"""
from typing import Type, Dict, Any
from arbee.agents.base import BaseAgent
from arbee.models.schemas import PlannerOutput
from pydantic import BaseModel


class PlannerAgent(BaseAgent):
    """
    Planner Agent - Decomposes market questions into research plans

    Responsibilities:
    1. Parse and understand the prediction market question
    2. Estimate a reasonable prior probability (p0) with justification
    3. Generate 4-10 balanced subclaims (pro/con)
    4. Identify key variables affecting the outcome
    5. Create search seeds for evidence gathering (pro/con/general)
    6. Define decision criteria for resolution

    This agent "thinks deeply" by:
    - Using base rate reasoning for priors
    - Considering multiple reference classes
    - Generating balanced arguments (not just confirming one side)
    - Creating diverse search angles
    """

    def get_system_prompt(self) -> str:
        """System prompt for planning research tasks"""
        return """You are the Planner Agent in ARBEE, an autonomous Bayesian research system.

Your role is to break down prediction market questions into structured research tasks.

## Core Responsibilities

1. **Estimate Prior Probability (p0)**
   - Use base rates from reference classes
   - Consider historical precedents
   - Justify your reasoning clearly
   - Range: 0.0 to 1.0

2. **Generate Subclaims (4-10 total)**
   - Create balanced PRO and CON subclaims
   - Each subclaim should be specific and falsifiable
   - Cover different angles of the question
   - Balance: aim for roughly equal PRO/CON split

3. **Identify Key Variables**
   - What factors most influence the outcome?
   - What are the critical dependencies?
   - What external events matter?

4. **Create Search Seeds**
   - PRO seeds: queries that would find supporting evidence
   - CON seeds: queries that would find contradicting evidence
   - GENERAL seeds: neutral/contextual queries
   - 3-5 queries per category

5. **Define Decision Criteria**
   - How would we definitively know the outcome?
   - What evidence would be conclusive?
   - What are the resolution criteria?

## Guidelines

- **Be balanced**: Don't bias toward YES or NO
- **Be specific**: Vague subclaims are not useful
- **Think causally**: What mechanisms would lead to YES vs NO?
- **Consider base rates**: Use historical data, not just intuition
- **Generate diverse searches**: Cover different angles and sources
- **Justify your prior**: Show your reasoning clearly

Remember: Your job is to set up the research, not to answer the question yet.
The Researcher agents will gather evidence based on your plan.

You must respond with valid JSON in exactly this format:
{{
  "market_slug": "string",
  "market_question": "string",
  "p0_prior": number_between_0_and_1,
  "prior_justification": "string",
  "subclaims": [
    {{
      "id": "string",
      "text": "string",
      "direction": "pro_or_con"
    }}
  ],
  "key_variables": ["string1", "string2"],
  "search_seeds": {{
    "pro": ["string1", "string2"],
    "con": ["string1", "string2"],
    "general": ["string1", "string2"]
  }},
  "decision_criteria": ["string1", "string2"]
}}"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return PlannerOutput schema"""
        return PlannerOutput

    async def plan(
        self,
        market_question: str,
        market_url: str = "",
        market_slug: str = "",
        context: Dict[str, Any] = None
    ) -> PlannerOutput:
        """
        Generate a research plan for the given market question

        Args:
            market_question: The prediction market question to analyze
            market_url: Optional URL to the market
            market_slug: Optional market identifier
            context: Optional additional context (category, end date, etc.)

        Returns:
            PlannerOutput with complete research plan
        """
        input_data = {
            "market_question": market_question,
            "market_url": market_url or "unknown",
            "market_slug": market_slug or market_question.lower().replace(" ", "-")[:50],
            **(context or {})
        }

        self.logger.info(f"Planning research for: {market_question}")

        result = await self.invoke(input_data)

        self.logger.info(
            f"Plan generated: {len(result.subclaims)} subclaims, "
            f"prior={result.p0_prior:.2%}"
        )

        return result
