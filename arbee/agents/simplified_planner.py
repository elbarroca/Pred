"""
Simplified Planner Agent - No tools, just reasoning
Generates research QUESTIONS (not search seeds)
"""
import json
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from arbee.models.schemas import SimplifiedPlannerOutput, SubjectProfile
from config.settings import Settings

logger = logging.getLogger(__name__)


class SimplifiedPlannerAgent:
    """
    Simplified Planner - focuses on identifying:
    1. Market type (sports/politics/finance/etc)
    2. Subject that needs profiling (WHO/WHAT are we researching?)
    3. Core research questions to answer
    4. Simple baseline prior

    NO tools needed - just LLM reasoning.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        self.settings = settings or Settings()
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.settings.OPENAI_API_KEY
        )
        logger.info(f"SimplifiedPlannerAgent initialized with {model_name}")

    def get_system_prompt(self) -> str:
        return """You are the Planner agent in POLYSEER. Your job is to:

1. **Classify the market type** (sports/politics/finance/entertainment/other)
2. **Identify the subject** (person/team/event that needs profiling)
3. **Generate core research questions** (not search queries - questions to answer)
4. **Set a baseline prior** (0.5 if no info, adjust if obvious)

# Guidelines

**Market Types:**
- **sports**: Athletic performance, race times, game outcomes
- **politics**: Elections, policy outcomes, political events
- **finance**: Market movements, company performance, economic indicators
- **entertainment**: Awards, box office, celebrity events
- **other**: Everything else

**Subject Identification:**
- Look for the KEY entity that determines the outcome
- Could be: person (athlete, politician), team, company, event
- If multiple subjects, identify the PRIMARY one to profile first

**Research Questions:**
- Should be SPECIFIC to the subject, not generic
- Focus on answering: Can this subject achieve the market outcome?
- Typical flow:
  1. Who/what is the subject and what's their baseline capability?
  2. What's the benchmark/average for this type of outcome?
  3. What recent evidence exists about the subject's performance?
  4. Are there specific factors that would make this more/less likely?

**Baseline Prior:**
- Default to 0.5 (50%) if you have no initial information
- Only deviate if the question makes the outcome obviously likely/unlikely
  - Example: "Will the sun rise tomorrow?" → 0.99
  - Example: "Will a randomly selected person run sub-15 minute 5k?" → 0.01
- For most markets: start at 0.5

# Output Format

Return JSON matching this EXACT schema:

```json
{
  "market_type": "sports|politics|finance|entertainment|other",
  "subject_to_profile": {
    "entity_name": "string (exact name to research)",
    "entity_type": "person|team|event|organization"
  },
  "core_research_questions": [
    "Question 1: Who is X and what is their baseline capability?",
    "Question 2: What is the benchmark for this type of outcome?",
    "Question 3: What recent evidence exists?",
    "Question 4 (optional): Other relevant factors?"
  ],
  "baseline_prior": 0.5,
  "prior_reasoning": "Why this prior makes sense given the question"
}
```

# Examples

## Example 1: Sports Performance

**Market:** "Will Diplo run a 5k in under 23 minutes by end of 2024?"

**Output:**
```json
{
  "market_type": "sports",
  "subject_to_profile": {
    "entity_name": "Diplo",
    "entity_type": "person"
  },
  "core_research_questions": [
    "Who is Diplo and what is their fitness/running background?",
    "What is a competitive 5k time for someone of Diplo's age and profile?",
    "Has Diplo run any races recently? What were the times?",
    "Is Diplo currently training for this specific goal?"
  ],
  "baseline_prior": 0.5,
  "prior_reasoning": "No initial information about Diplo's running ability, so start neutral at 50%"
}
```

## Example 2: Politics

**Market:** "Will Candidate X win the Senate race in State Y?"

**Output:**
```json
{
  "market_type": "politics",
  "subject_to_profile": {
    "entity_name": "Candidate X",
    "entity_type": "person"
  },
  "core_research_questions": [
    "Who is Candidate X and what is their political background?",
    "What are the historical voting patterns in State Y?",
    "What do recent polls show for this race?",
    "What are the key issues favoring/opposing Candidate X?"
  ],
  "baseline_prior": 0.5,
  "prior_reasoning": "Without polling data, start at even odds for a two-candidate race"
}
```

# Important

- Output ONLY the JSON, no additional text
- Questions should be answerable through research
- Questions should be SPECIFIC to this market, not generic
- The subject should be the PRIMARY entity we need to understand
"""

    async def plan(
        self,
        market_question: str,
        market_url: str = "",
        market_slug: str = ""
    ) -> SimplifiedPlannerOutput:
        """
        Generate research plan for a market question.

        Args:
            market_question: The prediction market question
            market_url: Optional URL to the market
            market_slug: Optional market identifier

        Returns:
            SimplifiedPlannerOutput with research questions and subject to profile
        """
        logger.info(f"Planning for market: {market_question}")

        # Build prompt
        system_msg = SystemMessage(content=self.get_system_prompt())
        human_msg = HumanMessage(
            content=f"Market Question: {market_question}\n\nGenerate the research plan JSON:"
        )

        # Get LLM response
        response = await self.llm.ainvoke([system_msg, human_msg])
        response_text = response.content

        # Parse JSON from response
        try:
            # Try to extract JSON if it's wrapped in markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            plan_data = json.loads(json_text)

            # Build SubjectProfile
            subject_data = plan_data["subject_to_profile"]
            subject_profile = SubjectProfile(
                entity_name=subject_data["entity_name"],
                entity_type=subject_data["entity_type"],
                key_facts={},  # Will be filled by Researcher
                baseline_capabilities=None  # Will be filled by Researcher
            )

            # Build SimplifiedPlannerOutput
            output = SimplifiedPlannerOutput(
                market_slug=market_slug or market_question.lower().replace(" ", "-")[:50],
                market_question=market_question,
                market_type=plan_data["market_type"],
                subject_to_profile=subject_profile,
                core_research_questions=plan_data["core_research_questions"],
                baseline_prior=plan_data.get("baseline_prior", 0.5),
                prior_reasoning=plan_data["prior_reasoning"]
            )

            logger.info(
                f"✅ Plan complete: {output.market_type} market, "
                f"subject={output.subject_to_profile.entity_name}, "
                f"{len(output.core_research_questions)} questions, "
                f"prior={output.baseline_prior:.2%}"
            )

            return output

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
        except KeyError as e:
            logger.error(f"Missing required field in LLM response: {e}")
            logger.error(f"Response was: {plan_data}")
            raise ValueError(f"LLM response missing required field: {e}")
