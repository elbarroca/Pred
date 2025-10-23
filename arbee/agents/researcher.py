"""
Researcher Agent - Evidence gathering with Valyu AI integration
Executes parallel PRO/CON research based on Planner's search seeds
"""
from typing import Type, Dict, Any, List, Literal
import asyncio
import logging
from datetime import datetime, timedelta

from arbee.agents.base import BaseAgent
from arbee.models.schemas import ResearcherOutput, Evidence
from arbee.api_clients.valyu import ValyuResearchClient
from pydantic import BaseModel


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent - Gathers and scores evidence from web sources

    Responsibilities:
    1. Execute searches using Valyu AI based on Planner's search seeds
    2. Extract relevant evidence items with proper attribution
    3. Assign Log-Likelihood Ratios (LLRs) based on source quality
    4. Score evidence on verifiability, independence, and recency
    5. Maintain balanced PRO/CON evidence collection

    LLR Calibration Ranges (from CLAUDE.MD):
    - A-grade (primary sources): ±1-3
      Examples: Official government data, direct quotes from participants
    - B-grade (high-quality secondary): ±0.3-1.0
      Examples: NYT, WSJ, Reuters, academic papers
    - C-grade (secondary): ±0.1-0.5
      Examples: Quality blogs, industry newsletters, regional news
    - D-grade (weak): ±0.01-0.2
      Examples: Social media, rumors, heavily biased sources

    This agent "thinks deeply" by:
    - Evaluating source credibility rigorously
    - Extracting precise claims (not broad summaries)
    - Assigning calibrated LLRs based on evidence strength
    - Scoring independence to avoid echo chambers
    - Prioritizing recent, verifiable information
    """

    def __init__(self, direction: Literal["pro", "con", "general"] = "general", **kwargs):
        """
        Initialize Researcher Agent

        Args:
            direction: Research direction (pro, con, or general)
            **kwargs: Additional args passed to BaseAgent
        """
        super().__init__(**kwargs)
        self.direction = direction
        self.valyu_client = ValyuResearchClient()
        self.logger.info(f"ResearcherAgent initialized with direction={direction}")

    def get_system_prompt(self) -> str:
        """System prompt with LLR calibration guidance"""
        direction_instruction = {
            "pro": "You are seeking evidence that SUPPORTS a YES outcome.",
            "con": "You are seeking evidence that SUPPORTS a NO outcome.",
            "general": "You are seeking neutral contextual evidence."
        }

        return f"""You are a Researcher Agent in ARBEE, an autonomous Bayesian research system.

{direction_instruction[self.direction]}

## Core Responsibilities

1. **Gather Evidence**
   - Use provided search queries to find relevant sources
   - Extract specific, falsifiable claims
   - Include proper attribution (URL, date, source)

2. **Assign Log-Likelihood Ratios (LLRs)**
   - LLR represents how much this evidence shifts probability
   - Positive LLR = supports YES, Negative LLR = supports NO
   - Magnitude reflects strength and quality of evidence

### LLR Calibration Guidelines

**A-Grade Sources (±1 to ±3):**
- Primary sources: official data, government statistics, direct quotes
- Highly verifiable: independently reproducible
- Examples:
  - Official election results: LLR = ±2.5
  - Direct quote from candidate announcing policy: LLR = ±1.8
  - FDA approval announcement: LLR = ±2.0

**B-Grade Sources (±0.3 to ±1.0):**
- High-quality journalism: NYT, WSJ, Reuters, Bloomberg
- Peer-reviewed research (if directly relevant)
- Examples:
  - NYT investigative report with multiple sources: LLR = ±0.8
  - Reuters poll with proper methodology: LLR = ±0.6
  - Academic paper on relevant topic: LLR = ±0.7

**C-Grade Sources (±0.1 to ±0.5):**
- Secondary analysis: quality blogs, industry newsletters
- Regional news outlets
- Expert commentary (not primary research)
- Examples:
  - 538 analysis article: LLR = ±0.4
  - Industry analyst report: LLR = ±0.3
  - Substack by domain expert: LLR = ±0.2

**D-Grade Sources (±0.01 to ±0.2):**
- Weak sources: social media, rumors, heavily biased outlets
- Unverifiable claims
- Single-source anecdotes
- Examples:
  - Twitter thread by random user: LLR = ±0.05
  - Rumor from partisan blog: LLR = ±0.02
  - Anecdotal claim: LLR = ±0.1

3. **Score Evidence Quality**

**Verifiability Score (0-1):**
- 1.0: Independently verifiable (official data, multiple confirmations)
- 0.7: Likely verifiable (credible source, specific claims)
- 0.4: Partially verifiable (some specifics, some assertions)
- 0.2: Hard to verify (anonymous sources, vague claims)

**Independence Score (0-1):**
- 1.0: Completely independent research/reporting
- 0.7: Mostly independent with some common sources
- 0.4: Likely shares sources with other evidence
- 0.2: Clearly derivative or echo chamber content

**Recency Score (0-1):**
- 1.0: Published within last 7 days
- 0.8: Published within last 30 days
- 0.6: Published within last 90 days
- 0.4: Published within last year
- 0.2: Older than 1 year

4. **Extract Precise Claims**
   - Limit summaries to <500 characters
   - Focus on specific, falsifiable statements
   - Avoid vague generalities
   - Include direct quotes when possible (max 25 words)

## Example Evidence Extraction

**Good:**
{
  "subclaim_id": "sc1_research",
  "title": "ABC/WaPo Poll: Trump leads Harris by 3 points in Arizona",
  "url": "https://example.com/poll-oct-2024",
  "published_date": "2024-10-15",
  "source_type": "high_quality_secondary",
  "claim_summary": "ABC News/Washington Post poll (Oct 10-13, n=1,247 likely voters, MoE ±3%) shows Trump 49%, Harris 46% in Arizona. Methodology rated A- by 538.",
  "support": "pro",
  "verifiability_score": 0.9,
  "independence_score": 0.85,
  "recency_score": 1.0,
  "estimated_LLR": 0.6,
  "extraction_notes": "High-quality poll with transparent methodology. LLR=0.6 because: B-grade source, recent, verifiable, but within margin of error."
}

**Bad (too vague):**
{
  "claim_summary": "Polls show Trump doing well in swing states",
  "estimated_LLR": 1.5  // Wrong: too high for vague claim
}

## Important Guidelines

- **Be honest about uncertainty**: If a source is weak, assign low LLR
- **Don't cherry-pick**: Include contradicting evidence if found
- **Provide extraction notes**: Explain your LLR reasoning
- **Check publication dates**: Recent information scores higher
- **Verify sources**: Don't trust everything you find
- **Balance specificity**: Specific claims are more valuable than broad statements

Remember: You are gathering EVIDENCE, not making the final judgment.
The Analyst agent will aggregate all evidence using Bayesian methods.
Quality over quantity: 10 high-quality sources > 100 weak ones.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return ResearcherOutput schema"""
        return ResearcherOutput

    def get_human_prompt(self) -> str:
        """Human prompt for evidence extraction"""
        return """Analyze the provided search results and extract structured evidence items.

Search Results: {search_results}
Direction: {direction}
Max Items: {max_items}

Extract up to {max_items} high-quality evidence items that are relevant to the market question. Each evidence item should be a specific, falsifiable claim with proper attribution.

Return your response as a valid JSON object with this exact structure:
{{
    "evidence_items": [
        {{
            "subclaim_id": "string (use 'general' if no specific subclaim matches)",
            "title": "Brief descriptive title",
            "url": "Source URL",
            "published_date": "YYYY-MM-DD format",
            "source_type": "primary|high_quality_secondary|secondary|weak",
            "claim_summary": "Specific claim extracted from the source",
            "support": "pro|con|neutral",
            "verifiability_score": 0.0-1.0,
            "independence_score": 0.0-1.0,
            "recency_score": 0.0-1.0,
            "estimated_LLR": number,
            "extraction_notes": "Explanation of LLR and scoring"
        }}
    ]
}}

Focus on:
- Recent, verifiable sources
- Specific claims (not broad summaries)
- Proper LLR calibration based on source quality
- Balanced coverage if multiple perspectives exist
- Clear attribution with URLs and dates"""

    async def research(
        self,
        search_seeds: List[str],
        subclaims: List[Dict[str, Any]],
        market_question: str,
        max_evidence_per_seed: int = 5,
        date_range_days: int = 90
    ) -> ResearcherOutput:
        """
        Execute research using Valyu AI and extract evidence

        Args:
            search_seeds: List of search queries to execute
            subclaims: List of subclaim dicts from Planner (for matching evidence)
            market_question: The main market question (for context)
            max_evidence_per_seed: Maximum evidence items per search query
            date_range_days: How far back to search (default 90 days)

        Returns:
            ResearcherOutput with gathered evidence

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if search_seeds is None or not isinstance(search_seeds, list):
            raise ValueError(f"search_seeds must be list, got {type(search_seeds)}")

        if not search_seeds:
            raise ValueError("search_seeds cannot be empty")

        if subclaims is None or not isinstance(subclaims, list):
            raise ValueError(f"subclaims must be list, got {type(subclaims)}")

        if market_question is None or not isinstance(market_question, str):
            raise ValueError(f"market_question must be string, got {type(market_question)}")

        if not market_question.strip():
            raise ValueError("market_question cannot be empty")

        if not isinstance(max_evidence_per_seed, int) or max_evidence_per_seed < 1:
            raise ValueError(
                f"max_evidence_per_seed must be positive integer, got {max_evidence_per_seed}"
            )

        if not isinstance(date_range_days, int) or date_range_days < 1:
            raise ValueError(
                f"date_range_days must be positive integer, got {date_range_days}"
            )

        self.logger.info(
            f"Starting {self.direction.upper()} research with {len(search_seeds)} seeds"
        )

        # Step 1: Execute searches using Valyu
        search_results = await self._execute_searches(search_seeds, date_range_days)

        # Step 2: Use LLM to extract evidence from search results
        evidence_items = await self._extract_evidence(
            search_results, subclaims, market_question, max_evidence_per_seed
        )

        self.logger.info(
            f"{self.direction.upper()} research complete: {len(evidence_items)} evidence items"
        )

        return ResearcherOutput(evidence_items=evidence_items)

    async def _execute_searches(
        self,
        search_seeds: List[str],
        date_range_days: int
    ) -> List[Dict[str, Any]]:
        """
        Execute all search queries in parallel using Valyu

        Args:
            search_seeds: List of queries
            date_range_days: Recency filter

        Returns:
            Aggregated search results
        """
        self.logger.info(f"Executing {len(search_seeds)} Valyu searches")

        # Execute searches in parallel
        try:
            results = await self.valyu_client.multi_query_search(
                queries=search_seeds,
                max_results_per_query=10
            )

            # Filter by date if needed
            cutoff_date = datetime.now() - timedelta(days=date_range_days)
            all_results = []

            for query, items in results.items():
                for item in items:
                    # Add query context
                    item['search_query'] = query
                    all_results.append(item)

            self.logger.info(f"Retrieved {len(all_results)} total search results")
            return all_results

        except Exception as e:
            self.logger.error(f"Valyu search failed: {e}")
            return []

    async def _extract_evidence(
        self,
        search_results: List[Dict[str, Any]],
        subclaims: List[Dict[str, Any]],
        market_question: str,
        max_items: int
    ) -> List[Evidence]:
        """
        Use LLM to extract structured evidence from search results

        Args:
            search_results: Raw search results from Valyu
            subclaims: Subclaim list for matching
            market_question: Main question for context
            max_items: Maximum evidence items to extract

        Returns:
            List of Evidence objects
        """
        if not search_results:
            self.logger.warning("No search results to extract from")
            return []

        # Prepare input for LLM
        input_data = {
            "search_results": search_results[:50],  # Limit context size
            "direction": self.direction,
            "max_items": max_items
        }

        # Invoke LLM to extract evidence
        try:
            result = await self.invoke(input_data)
            self.logger.info(f"Extracted {len(result.evidence_items)} evidence items")
            return result.evidence_items

        except Exception as e:
            self.logger.error(f"Evidence extraction failed: {e}")
            return []


# Convenience functions for parallel execution
async def run_parallel_research(
    search_seeds_pro: List[str],
    search_seeds_con: List[str],
    search_seeds_general: List[str],
    subclaims: List[Dict[str, Any]],
    market_question: str,
    **kwargs
) -> Dict[str, ResearcherOutput]:
    """
    Run PRO, CON, and GENERAL researchers in parallel

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
    # Create agents
    researcher_pro = ResearcherAgent(direction="pro", **kwargs)
    researcher_con = ResearcherAgent(direction="con", **kwargs)
    researcher_general = ResearcherAgent(direction="general", **kwargs)

    # Execute in parallel
    results = await asyncio.gather(
        researcher_pro.research(search_seeds_pro, subclaims, market_question),
        researcher_con.research(search_seeds_con, subclaims, market_question),
        researcher_general.research(search_seeds_general, subclaims, market_question),
        return_exceptions=True
    )

    # Handle results
    output = {}
    for direction, result in zip(["pro", "con", "general"], results):
        if isinstance(result, Exception):
            logging.error(f"{direction.upper()} research failed: {result}")
            output[direction] = ResearcherOutput(evidence=[])
        else:
            output[direction] = result

    return output
