"""
Planner Agent - First agent in POLYSEER workflow
Breaks down market questions into structured research tasks
"""
from typing import Type, Dict, Any, Optional
from arbee.agents.base import BaseAgent
from arbee.agents.schemas import PlannerOutput
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

## CHAIN-OF-THOUGHT REASONING & MEMORY MGMT REQUIRED

Before generating your final output, you MUST think step-by-step and track your reasoning:

**Step 1: Initial Assessment**
- What is this question really asking?
- What is my immediate intuition about the probability?
- How does this relate to previous market questions I've analyzed?

**Step 2: Reference Class Analysis**
- What historical precedents or base rates apply?
- What similar events can I learn from?
- What does the outside view suggest?
- Consult my memory of similar cases for priors

**Step 3: Key Factors Identification**
- What are the 3-5 most critical variables?
- What could make YES more likely? What could make NO more likely?
- What dependencies exist between factors?

**Step 4: Subclaim Generation**
- What specific claims would support YES? (pro)
- What specific claims would support NO? (con)
- Am I being balanced? Do I have roughly equal pro/con?
- How do these connect to the key factors identified?

**Step 5: Search Strategy**
- What queries would find the strongest PRO evidence?
- What queries would find the strongest CON evidence?
- What neutral/contextual queries give us background?
- What search strategies have worked for similar topics in the past?

**Step 6: Memory Integration & Validation**
- How does my plan build on previous agent memory?
- What should be stored in working memory for future agents?
- What might be sensitive and need to be flagged for erasure?

Include your complete reasoning trace in the "reasoning_trace" field.

## CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT

1. **FIRST**: Write your complete step-by-step reasoning in the "reasoning_trace" field
2. **THEN**: Fill in all other required fields based on your reasoning
3. Your reasoning_trace MUST follow this structure:
   - Step 1: Initial Assessment - What is this question asking? What's my intuition?
   - Step 2: Reference Class Analysis - What base rates apply?
   - Step 3: Key Factors Identification - What are the critical variables?
   - Step 4: Subclaim Generation - What specific claims support YES/NO?
   - Step 5: Search Strategy - What queries will find the best evidence?
   - Step 6: Memory Integration - How does this connect to prior knowledge and what should be remembered?

DO NOT skip the reasoning_trace. It is REQUIRED for auditability and chain-of-thought reasoning.

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

    def validate_output(self, output: BaseModel) -> tuple[bool, Optional[str]]:
        """
        Validate PlannerOutput for balanced subclaims and complete search seeds

        Checks:
        1. Prior is in reasonable range [0.01, 0.99]
        2. Subclaims are balanced (roughly equal pro/con)
        3. Search seeds are non-empty for all directions
        4. Reasoning trace is present
        """
        # Base validation
        is_valid, feedback = super().validate_output(output)
        if not is_valid:
            return is_valid, feedback

        # Type check
        if not isinstance(output, PlannerOutput):
            return False, f"Expected PlannerOutput, got {type(output)}"

        issues = []

        # Check prior range
        if not (0.01 <= output.p0_prior <= 0.99):
            issues.append(f"p0_prior ({output.p0_prior:.2%}) should be in range [1%, 99%]")

        # Check subclaims balance
        pro_count = sum(1 for sc in output.subclaims if sc.direction == "pro")
        con_count = sum(1 for sc in output.subclaims if sc.direction == "con")

        if pro_count == 0:
            issues.append("No PRO subclaims - need balanced pro/con subclaims")
        elif con_count == 0:
            issues.append("No CON subclaims - need balanced pro/con subclaims")
        else:
            ratio = pro_count / con_count if con_count > 0 else float('inf')
            # Allow imbalance up to 2:1 ratio
            if ratio > 2.0 or ratio < 0.5:
                issues.append(
                    f"Subclaims are imbalanced: {pro_count} PRO vs {con_count} CON "
                    f"(ratio {ratio:.2f}). Aim for roughly equal pro/con split."
                )

        # Check search seeds
        if not output.search_seeds.pro or len(output.search_seeds.pro) == 0:
            issues.append("PRO search seeds are empty - need queries to find supporting evidence")

        if not output.search_seeds.con or len(output.search_seeds.con) == 0:
            issues.append("CON search seeds are empty - need queries to find contradicting evidence")

        if not output.search_seeds.general or len(output.search_seeds.general) == 0:
            issues.append("GENERAL search seeds are empty - need contextual queries")

        # Check reasoning trace (if field exists)
        if hasattr(output, 'reasoning_trace'):
            if not output.reasoning_trace or len(output.reasoning_trace) < 100:
                issues.append(
                    "reasoning_trace is too short or empty - must provide detailed step-by-step reasoning"
                )

        if issues:
            feedback_msg = "Planning validation issues:\n" + "\n".join(issues)
            return False, feedback_msg

        # All validation passed
        return True, None
