"""
Autonomous ResearcherAgent with ReAct Pattern
Pilot implementation demonstrating autonomous reasoning + tool use
"""
import json
import re
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import ResearcherOutput, Evidence
from arbee.tools.search import web_search_tool, multi_query_search_tool
from arbee.tools.evidence import extract_evidence_tool, verify_source_tool, ExtractedEvidence
from arbee.tools.validation import validate_search_results_tool
from arbee.tools.memory_search import search_similar_markets_tool

logger = logging.getLogger(__name__)


class AutonomousResearcherAgent(AutonomousReActAgent):
    """
    Autonomous Researcher Agent - Gathers and scores evidence using iterative reasoning

    This agent demonstrates the full ReAct pattern:
    1. Observes what evidence is needed (from subclaims and search seeds)
    2. Thinks about search strategy (which queries to try first)
    3. Acts by calling web search tools
    4. Observes search results quality
    5. Validates if sufficient evidence gathered
    6. Extracts structured evidence from best results
    7. Continues or terminates based on quality/quantity

    Capabilities:
    - Autonomous search strategy (tries different queries if needed)
    - Quality validation (checks if results are sufficient)
    - Source verification (assesses credibility)
    - Evidence extraction (structured parsing with LLR estimation)
    - Learning from similar cases (memory search)
    """

    def __init__(
        self,
        direction: Literal["pro", "con", "general"] = "general",
        min_evidence_items: int = 5,
        max_search_attempts: int = 10,
        **kwargs
    ):
        """
        Initialize Autonomous Researcher Agent

        Args:
            direction: Research direction (pro/con/general)
            min_evidence_items: Minimum evidence items before completion
            max_search_attempts: Maximum search queries to try
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.direction = direction
        self.min_evidence_items = min_evidence_items
        self.max_search_attempts = max_search_attempts

        self.logger.info(
            f"AutonomousResearcherAgent initialized: direction={direction}, "
            f"min_evidence={min_evidence_items}"
        )

    def get_system_prompt(self) -> str:
        """
        System prompt for autonomous research with tool usage guidelines
        """
        direction_instruction = {
            "pro": "You are seeking evidence that SUPPORTS a YES outcome.",
            "con": "You are seeking evidence that SUPPORTS a NO outcome.",
            "general": "You are seeking neutral contextual evidence."
        }

        return f"""You are an Autonomous Researcher Agent in POLYSEER.

{direction_instruction[self.direction]}

## Your Mission

Find HIGH-QUALITY, VERIFIABLE evidence for the given subclaims related to a prediction market question.
You will use multiple tools iteratively until you have gathered sufficient evidence.

## Available Tools

You have access to these tools:

1. **web_search_tool** - Search the web for information
   - Use this to find articles, reports, polls, studies
   - Be specific in your queries for better results
   - Try different phrasings if first search doesn't work well

2. **multi_query_search_tool** - Execute multiple searches in parallel
   - Use this when you have several different search angles
   - More efficient than multiple individual searches

3. **verify_source_tool** - Check source credibility
   - Use this before extracting evidence from a source
   - Helps assess if source is trustworthy

4. **extract_evidence_tool** - Parse search result into structured evidence
   - Use this on your best search results
   - Automatically estimates LLR and quality scores

5. **validate_search_results_tool** - Check if you have enough results
   - Use this to decide if you need more searches
   - Validates diversity and quality

6. **search_similar_markets_tool** - Find similar past analyses
   - Use this at the start to learn from similar cases
   - Can inform your search strategy

## Task Input Format

Your task input will contain:
- **search_seeds**: List of specific search queries to start with
- **subclaims**: List of specific claims to find evidence for
- **market_question**: The main prediction market question

**CRITICAL**: Always use the exact search_seeds provided in your task input for initial searches.

## Your Reasoning Process

Follow these steps iteratively:

**Step 1: Understand the Task**
- Review market question, subclaims, and search seeds provided in your task input
- Check if similar markets have been analyzed before (use search_similar_markets_tool)
- Plan initial search strategy using the exact search seeds provided

**Step 2: Execute Searches**
- START WITH PROVIDED SEARCH SEEDS: Use the exact search seeds in your task input
- Use web_search_tool or multi_query_search_tool
- If results are poor, try alternative queries based on the same topics
- Always include specific names, dates, and contexts from the market question

**Step 3: Validate Results**
- Use validate_search_results_tool to check if you have enough results
- Consider both quantity and diversity
- If insufficient, formulate better queries and search again

**Step 4: Extract Evidence**
- Select best search results (relevant, credible, recent)
- Optionally verify sources using verify_source_tool
- Use extract_evidence_tool to parse results into structured evidence
- **IMPORTANT**: After getting evidence from extract_evidence_tool, store it in intermediate_results['evidence_items']
- Use: intermediate_results['evidence_items'].append(evidence_item) for each new piece of evidence
- Continue until you have {self.min_evidence_items}+ quality items

**Step 5: Decide Completion**
- Check if you have sufficient evidence ({self.min_evidence_items}+ items) in intermediate_results['evidence_items']
- Check if evidence is diverse (not all from same source)
- If yes → task complete, return evidence
- If no and haven't hit search limit → continue searching
- If search limit reached → return what you have

## Quality Standards

- **Verifiable**: Primary sources and high-quality journalism preferred
- **Recent**: Prioritize recent information (within last 90 days if possible)
- **Diverse**: Avoid over-relying on single source or echo chamber
- **Specific**: Concrete claims with numbers/dates better than vague statements
- **Relevant**: Must relate to subclaims, not tangential information

## Important Guidelines

- **Think before acting**: Explain your reasoning before each tool call
- **Validate quality**: Don't just collect evidence, ensure it's high quality
- **Try alternatives**: If a search strategy isn't working, try different queries
- **Know when to stop**: Don't endlessly search if you have sufficient evidence
- **Be efficient**: Use parallel searches when possible

## Response Format

**CRITICAL: Evidence Storage**
- **IMMEDIATELY** after calling extract_evidence_tool, store the result
- **DO NOT** summarize evidence in text - store the structured evidence objects
- **ALWAYS** use: intermediate_results['evidence_items'].append(evidence_item)
- **DO NOT** write human-readable summaries of evidence

**Example Correct Usage:**
```
# After tool call returns evidence
evidence = extract_evidence_tool(...)  # This returns an ExtractedEvidence object
if evidence:
    if 'evidence_items' not in intermediate_results:
        intermediate_results['evidence_items'] = []
    intermediate_results['evidence_items'].append(evidence)
    # DO NOT write summaries like "I found evidence that..."
    # DO NOT write human-readable text
    # ONLY store the structured evidence object
```

**CRITICAL RESPONSE FORMAT:**
- If you call extract_evidence_tool and get a result, IMMEDIATELY store it
- Your response should be: "Evidence stored successfully" or similar confirmation
- DO NOT include the evidence details in your response text
- The evidence is stored in intermediate_results['evidence_items']

**When you think you're done:**
- Your final message should only confirm completion
- The extract_final_output function will parse intermediate_results['evidence_items'] into ResearcherOutput format
- Make sure all evidence items are stored in intermediate_results['evidence_items']

Remember: Quality over quantity. {self.min_evidence_items} excellent sources beats 50 weak ones.
"""

    def get_tools(self) -> List[BaseTool]:
        """
        Return research tools available to this agent
        """
        # Core research tools
        tools = [
            web_search_tool,
            multi_query_search_tool,
            extract_evidence_tool,
            verify_source_tool,
            validate_search_results_tool,
        ]

        # Memory tools (if store is configured)
        if self.store:
            tools.append(search_similar_markets_tool)

        return tools

    def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message
    ) -> None:
        """
        Automatically track search usage and persist extracted evidence.
        """
        intermediate = state.setdefault('intermediate_results', {})

        if tool_name in {"web_search_tool", "multi_query_search_tool"}:
            increment = 1
            artifact = getattr(tool_message, "artifact", None)
            if tool_name == "multi_query_search_tool" and isinstance(artifact, dict):
                increment = max(len(artifact), 1)
            intermediate['search_count'] = intermediate.get('search_count', 0) + increment
            return

        if tool_name != "extract_evidence_tool":
            return

        evidence = self._coerce_extract_evidence(tool_message)
        if evidence is None:
            return

        evidence_items = intermediate.setdefault('evidence_items', [])

        # Skip duplicates based on URL
        new_url = getattr(evidence, 'url', '') or ''
        for existing in evidence_items:
            existing_url = ''
            if isinstance(existing, ExtractedEvidence):
                existing_url = existing.url
            elif isinstance(existing, dict):
                existing_url = existing.get('url', '')
            elif hasattr(existing, 'url'):
                existing_url = getattr(existing, 'url')

            if existing_url and new_url and existing_url == new_url:
                self.logger.info("♻️  Duplicate evidence detected, skipping auto-store")
                return

        evidence_items.append(evidence)
        self.logger.info(
            f"📥 Evidence stored automatically ({len(evidence_items)} total items)"
        )

    def _coerce_extract_evidence(self, tool_message) -> Optional[ExtractedEvidence]:
        """
        Convert a tool message payload into an ExtractedEvidence object.

        Supports LangGraph tool artifacts, dict payloads, and string fallbacks.
        """
        artifact = getattr(tool_message, "artifact", None)
        if artifact is None:
            artifact = getattr(tool_message, "additional_kwargs", {}).get("return_value")

        try:
            if isinstance(artifact, ExtractedEvidence):
                return artifact
            if isinstance(artifact, dict):
                return ExtractedEvidence(**artifact)
            if hasattr(artifact, "model_dump"):
                return ExtractedEvidence(**artifact.model_dump())
        except Exception as exc:
            self.logger.warning(f"Failed to parse artifact as ExtractedEvidence: {exc}")

        # Fallback to parsing the tool message content
        text_payload = self._message_text(tool_message)
        if not text_payload:
            return None

        # Try JSON first
        try:
            data = json.loads(text_payload)
            return ExtractedEvidence(**data)
        except Exception:
            pass

        # Parse key=value pairs (repr-style) as a last resort
        # Match repr-style key=value pairs while allowing quoted strings and trimming delimiters
        kv_pairs = re.findall(
            r'(\w+)=(".*?"|\'.*?\'|[^\s,]+)',
            text_payload
        )
        if not kv_pairs:
            return None

        parsed: Dict[str, Any] = {}
        for key, raw_value in kv_pairs:
            value = raw_value.strip().strip('"\'')
            # Drop trailing delimiters that often appear in repr strings
            if value.endswith((',', ')')):
                value = value.rstrip(',)')
            parsed[key] = value

        for float_key in (
            'verifiability_score',
            'independence_score',
            'recency_score',
            'estimated_LLR'
        ):
            if float_key in parsed:
                try:
                    if isinstance(parsed[float_key], str):
                        parsed[float_key] = parsed[float_key].rstrip(',)')
                    parsed[float_key] = float(parsed[float_key])
                except ValueError:
                    pass

        if 'support' in parsed:
            parsed['support'] = parsed['support'].lower()

        try:
            return ExtractedEvidence(**parsed)
        except Exception as exc:
            self.logger.warning(f"Failed to coerce evidence from text payload: {exc}")
            return None

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Determine if research task is complete

        Criteria:
        1. Have we extracted sufficient evidence items? (min_evidence_items)
        2. Have we hit max search attempts?
        3. Did agent explicitly signal completion?

        Args:
            state: Current agent state

        Returns:
            True if task is complete
        """
        # Check if agent has gathered evidence items
        evidence_items = state.get('intermediate_results', {}).get('evidence_items', [])
        evidence_count = len(evidence_items)

        # Check search attempts
        search_count = state.get('intermediate_results', {}).get('search_count', 0)

        # Task complete if we have enough evidence
        if evidence_count >= self.min_evidence_items:
            self.logger.info(
                f"✅ Task complete: {evidence_count} evidence items gathered "
                f"(minimum: {self.min_evidence_items})"
            )
            return True

        # Also complete if we've hit max search attempts (even if not enough evidence)
        if search_count >= self.max_search_attempts:
            self.logger.warning(
                f"⚠️  Task complete: Max search attempts ({self.max_search_attempts}) reached "
                f"with only {evidence_count} evidence items"
            )
            return True

        # Otherwise, continue
        self.logger.info(
            f"🔄 Continue: {evidence_count}/{self.min_evidence_items} evidence, "
            f"{search_count}/{self.max_search_attempts} searches"
        )
        return False

    async def extract_final_output(self, state: AgentState) -> ResearcherOutput:
        """
        Extract ResearcherOutput from final agent state

        Args:
            state: Final agent state

        Returns:
            ResearcherOutput with all gathered evidence
        """
        # Get evidence items from intermediate results
        evidence_items = state.get('intermediate_results', {}).get('evidence_items', [])

        # Convert to Evidence objects if they're dicts or ExtractedEvidence objects
        evidence_list = []
        for item in evidence_items:
            if isinstance(item, dict):
                # Parse dict into Evidence model
                evidence_list.append(Evidence(**item))
            elif isinstance(item, Evidence):
                evidence_list.append(item)
            elif isinstance(item, ExtractedEvidence):  # ExtractedEvidence object
                # Convert ExtractedEvidence to Evidence
                try:
                    # Parse date string to date object
                    date_obj = None
                    if hasattr(item, 'published_date') and item.published_date != "unknown":
                        try:
                            from datetime import datetime
                            date_obj = datetime.strptime(item.published_date, "%Y-%m-%d").date()
                        except ValueError:
                            # Use a default date if parsing fails
                            from datetime import date
                            date_obj = date.today()
                    else:
                        # Use today's date as default for unknown dates
                        from datetime import date
                        date_obj = date.today()

                    evidence_dict = {
                        'subclaim_id': item.subclaim_id,
                        'title': item.title,
                        'url': item.url,
                        'published_date': date_obj,
                        'source_type': item.source_type,
                        'claim_summary': item.claim_summary,
                        'support': item.support,
                        'verifiability_score': item.verifiability_score,
                        'independence_score': item.independence_score,
                        'recency_score': item.recency_score,
                        'estimated_LLR': item.estimated_LLR,
                        'extraction_notes': item.extraction_notes
                    }
                    evidence_list.append(Evidence(**evidence_dict))
                except Exception as e:
                    self.logger.warning(f"Failed to convert evidence item: {e}")
                    continue

        pro_count = sum(1 for e in evidence_list if e.support == 'pro')
        con_count = sum(1 for e in evidence_list if e.support == 'con')
        neutral_count = len(evidence_list) - pro_count - con_count

        total_pro_llr = sum(
            e.estimated_LLR for e in evidence_list
            if e.support == 'pro' and e.estimated_LLR > 0
        )
        total_con_llr = sum(
            abs(e.estimated_LLR) for e in evidence_list
            if e.support == 'con' and e.estimated_LLR < 0
        )
        net_llr = sum(e.estimated_LLR for e in evidence_list)

        subclaims_data = state.get('task_input', {}).get('subclaims', [])
        subclaim_direction_map = {
            sc.get('id'): sc.get('direction')
            for sc in subclaims_data
            if sc.get('id') and sc.get('direction')
        }
        directional_items = 0
        aligned_items = 0
        for ev in evidence_list:
            expected_direction = subclaim_direction_map.get(ev.subclaim_id)
            if expected_direction not in {'pro', 'con'}:
                continue
            if ev.support == 'neutral':
                continue
            directional_items += 1
            if ev.support == expected_direction:
                aligned_items += 1

        context_alignment_score = (
            aligned_items / directional_items if directional_items else 0.0
        )

        self.logger.info(
            f"📤 Final output: {len(evidence_list)} evidence items "
            f"(pro={pro_count}, con={con_count}, neutral={neutral_count}), "
            f"net LLR={net_llr:+.2f}, context_alignment={context_alignment_score:.2f}"
        )

        search_strategy = (
            f"{self.direction.upper()} search captured {len(evidence_list)} items "
            f"(pro={pro_count}, con={con_count}, neutral={neutral_count}); "
            f"net LLR {net_llr:+.2f}"
        )

        return ResearcherOutput(
            evidence_items=evidence_list,
            total_pro_count=pro_count,
            total_con_count=con_count,
            total_pro_llr=total_pro_llr,
            total_con_llr=total_con_llr,
            net_llr=net_llr,
            context_alignment_score=context_alignment_score,
            search_strategy=search_strategy
        )

    async def run_research(
        self,
        search_seeds: List[str],
        subclaims: List[Dict[str, Any]],
        market_question: str,
        **kwargs
    ) -> ResearcherOutput:
        """
        Convenience method matching old ResearcherAgent interface

        Args:
            search_seeds: Search queries to execute
            subclaims: Subclaims to find evidence for
            market_question: Main market question

        Returns:
            ResearcherOutput with gathered evidence
        """
        return await self.run(
            task_description=f"Gather {self.direction.upper()} evidence for market question",
            task_input={
                'search_seeds': search_seeds,
                'subclaims': subclaims,
                'market_question': market_question,
                **kwargs
            }
        )


# Convenience function for parallel execution
async def run_parallel_autonomous_research(
    search_seeds_pro: List[str],
    search_seeds_con: List[str],
    search_seeds_general: List[str],
    subclaims: List[Dict[str, Any]],
    market_question: str,
    **kwargs
) -> Dict[str, ResearcherOutput]:
    """
    Run PRO, CON, and GENERAL autonomous researchers in parallel

    Args:
        search_seeds_pro: PRO search queries
        search_seeds_con: CON search queries
        search_seeds_general: GENERAL search queries
        subclaims: Subclaim list
        market_question: Main question
        **kwargs: Additional args for researchers

    Returns:
        Dict with 'pro', 'con', 'general' ResearcherOutput objects
    """
    import asyncio

    # Create autonomous agents
    researcher_pro = AutonomousResearcherAgent(direction="pro", **kwargs)
    researcher_con = AutonomousResearcherAgent(direction="con", **kwargs)
    researcher_general = AutonomousResearcherAgent(direction="general", **kwargs)

    # Execute in parallel
    results = await asyncio.gather(
        researcher_pro.run_research(search_seeds_pro, subclaims, market_question),
        researcher_con.run_research(search_seeds_con, subclaims, market_question),
        researcher_general.run_research(search_seeds_general, subclaims, market_question),
        return_exceptions=True
    )

    # Handle results
    output = {}
    for direction, result in zip(["pro", "con", "general"], results):
        if isinstance(result, Exception):
            logger.error(f"{direction.upper()} autonomous research failed: {result}")
            output[direction] = ResearcherOutput(evidence_items=[])
        else:
            output[direction] = result

    return output
