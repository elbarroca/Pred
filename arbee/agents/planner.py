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

## CHAIN-OF-THOUGHT REASONING REQUIRED

Before generating your final output, you MUST think step-by-step:

**Step 1: Initial Assessment**
- What is this question really asking?
- What is my immediate intuition about the probability?

**Step 2: Reference Class Analysis**
- What historical precedents or base rates apply?
- What similar events can I learn from?
- What does the outside view suggest?

**Step 3: Key Factors Identification**
- What are the 3-5 most critical variables?
- What could make YES more likely? What could make NO more likely?
- What dependencies exist between factors?

**Step 4: Subclaim Generation**
- What specific claims would support YES? (pro)
- What specific claims would support NO? (con)
- Am I being balanced? Do I have roughly equal pro/con?

**Step 5: Search Strategy**
- What queries would find the strongest PRO evidence?
- What queries would find the strongest CON evidence?
- What neutral/contextual queries give us background?

**Step 6: Prior Estimation**
- Based on base rates and key factors, what's a reasonable p0?
- Why this number and not higher/lower?
- What's my confidence in this prior?

Include your complete reasoning trace in the "reasoning_trace" field.

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
  "decision_criteria": ["string1", "string2"],
  "reasoning_trace": "Step 1: Initial Assessment...[detailed step-by-step reasoning]"
}}"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return PlannerOutput schema"""
        return PlannerOutput

    def get_human_prompt(self) -> str:
        """Human prompt for market analysis planning"""
        return """Analyze this prediction market question and create a comprehensive research plan.

Market Question: {market_question}
Market URL: {market_url}
Market Slug: {market_slug}

Your task is to:
1. Set a reasonable prior probability (p0) between 0.01 and 0.99
2. Break down the question into 4-8 specific subclaims that can be researched independently
3. Generate balanced search seeds for both PRO and CON positions
4. Identify key variables that will determine the outcome

Return a JSON response in exactly this format:
{{
    "market_slug": "clean-slug-identifier",
    "market_question": "the original question",
    "p0_prior": 0.5,
    "prior_justification": "brief explanation of why you chose this prior",
    "subclaims": [
        {{
            "id": "subclaim_1",
            "text": "specific claim that can be verified",
            "direction": "pro"
        }},
        {{
            "id": "subclaim_2",
            "text": "specific claim that can be verified",
            "direction": "con"
        }}
    ],
    "key_variables": ["variable1", "variable2"],
    "search_seeds": {{
        "pro": ["search term 1", "search term 2"],
        "con": ["search term 1", "search term 2"],
        "general": ["context search 1", "context search 2"]
    }},
    "decision_criteria": ["criterion 1", "criterion 2"]
}}

Guidelines:
- Prior should reflect genuine uncertainty (avoid 0.0, 0.5, 1.0 unless truly appropriate)
- Subclaims should be specific, falsifiable, and researchable
- Search seeds should be targeted queries that will find relevant evidence
- Balance PRO and CON research equally"""

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

        Raises:
            ValueError: If market_question is invalid
        """
        # Validate inputs
        if market_question is None or not isinstance(market_question, str):
            raise ValueError(f"market_question must be string, got {type(market_question)}")

        if not market_question.strip():
            raise ValueError("market_question cannot be empty")

        if len(market_question.strip()) < 10:
            raise ValueError(
                f"market_question too short: '{market_question}' "
                f"({len(market_question)} chars, minimum 10)"
            )

        if len(market_question) > 500:
            raise ValueError(
                f"market_question too long: {len(market_question)} chars (maximum 500)"
            )

        # Prepare input data in the format expected by the template
        input_data = {
            "market_question": market_question,
            "market_url": market_url or "unknown",
            "market_slug": market_slug or market_question.lower().replace(" ", "-")[:50],
            **(context or {})
        }

        # Also provide as a single input variable for backward compatibility
        input_data["input"] = f"""Market Question: {market_question}
Market URL: {market_url or 'unknown'}
Market Slug: {market_slug or market_question.lower().replace(' ', '-')[:50]}"""

        self.logger.info(f"Planning research for: {market_question}")

        result = await self.invoke(input_data)

        self.logger.info(
            f"Plan generated: {len(result.subclaims)} subclaims, "
            f"prior={result.p0_prior:.2%}"
        )

        return result
