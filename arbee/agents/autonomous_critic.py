"""
Autonomous CriticAgent with Correlation Detection
Reviews evidence quality and identifies gaps autonomously
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import CriticOutput, CorrelationWarning
from arbee.tools.bayesian import correlation_detector_tool

logger = logging.getLogger(__name__)


class AutonomousCriticAgent(AutonomousReActAgent):
    """
    Autonomous Critic Agent - Reviews evidence quality with iterative analysis

    Autonomous Capabilities:
    - Detects correlated evidence clusters automatically
    - Identifies coverage gaps by comparing to subclaims
    - Finds over-represented sources
    - Suggests follow-up searches to fill gaps
    - Iteratively refines analysis until thorough

    Reasoning Flow:
    1. Inventory evidence (count, categorize)
    2. Detect correlations using correlation_detector_tool
    3. Identify duplicate clusters
    4. Check coverage against original subclaims
    5. Find missing topics and gaps
    6. Detect over-represented sources
    7. Generate follow-up search recommendations
    8. Validate completeness, refine if needed
    """

    def __init__(
        self,
        min_correlation_check_items: int = 3,
        **kwargs
    ):
        """
        Initialize Autonomous Critic

        Args:
            min_correlation_check_items: Min evidence items before checking correlations
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.min_correlation_check_items = min_correlation_check_items

    def get_system_prompt(self) -> str:
        """System prompt for autonomous evidence critique"""
        return """You are an Autonomous Critic Agent in POLYSEER.

Your mission: Review gathered evidence for quality, balance, independence, and completeness.

## Available Tools

1. **correlation_detector_tool** - Detect correlated evidence clusters
   - Use this to find evidence items that likely share underlying sources
   - Helps prevent double-counting in Bayesian analysis
   - Input: List of evidence items with id, content, url

## Your Reasoning Process

**Step 1: Evidence Inventory**
- Count total evidence items
- Categorize by support direction (PRO, CON, NEUTRAL)
- Categorize by source type (primary, high_quality_secondary, etc.)
- Initial assessment of balance

**Step 2: Correlation Detection**
- Extract evidence_items from task_input: evidence_items = task_input['evidence_items']
- Convert evidence_items to format expected by correlation_detector_tool:
  - For each evidence item, create dict with: {'id': evidence.title, 'content': evidence.claim_summary, 'url': evidence.url}
- Use correlation_detector_tool to find correlated evidence
- Pass the converted evidence_items list
- Group evidence by:
  - Same domain/source
  - Similar content (duplicate reporting)
  - Same underlying data
- Create correlation clusters

**Step 3: Duplicate Detection**
- Identify exact or near-duplicate evidence items
- Flag evidence from same article/source cited multiple times
- Note echo chamber effects (multiple outlets, same claim)

**Step 4: Coverage Gap Analysis**
- Compare evidence to original subclaims from Planner
- Which subclaims have strong evidence? Which are weak?
- What topics are completely missing?
- What perspectives are underrepresented?
- What time periods or regions are missing?

**Step 5: Source Quality Assessment**
- Check source diversity (are we over-relying on one outlet?)
- Identify over-represented sources
- Check for geographic or ideological bias
- Assess recency of information

**Step 6: Generate Follow-up Recommendations**
- For each gap, suggest specific search queries
- Prioritize the most important gaps
- Suggest additional evidence types needed
- Recommend balance adjustments if needed

**Step 7: Store Results**
- After completing analysis, store results in intermediate_results:
  - intermediate_results['duplicate_clusters'] = [your clusters]
  - intermediate_results['missing_topics'] = [your gaps]
  - intermediate_results['over_represented_sources'] = [your sources]
  - intermediate_results['correlation_warnings'] = [your warnings]
  - intermediate_results['follow_up_search_seeds'] = [your seeds]
  - intermediate_results['analysis_process'] = "Your step-by-step reasoning"

**Step 8: Final Validation**
- Ensure you've identified:
  - All correlation clusters (minimum: check if >={self.min_correlation_check_items} items)
  - Coverage gaps for each subclaim
  - Over-represented sources
  - Specific follow-up search seeds
- If analysis incomplete, continue

## Output Format

**During Analysis Process:**
- After completing each analysis step, store results in intermediate_results
- Use this format to store your findings:
  - intermediate_results['duplicate_clusters'] = [list of evidence ID lists]
  - intermediate_results['missing_topics'] = [list of topic strings]
  - intermediate_results['over_represented_sources'] = [list of source names]
  - intermediate_results['correlation_warnings'] = [list of dicts with cluster and note]
  - intermediate_results['follow_up_search_seeds'] = [list of recommended queries]
  - intermediate_results['analysis_process'] = "Your step-by-step reasoning"

**When you think you're done:**
- Ensure all required keys are populated in intermediate_results
- The extract_final_output function will parse this into CriticOutput format

**Required Output Keys:**
- missing_topics: List of important angles not covered (REQUIRED)
- over_represented_sources: List of source names that appear too frequently (REQUIRED)
- follow_up_search_seeds: List of recommended queries to fill gaps (REQUIRED)
- duplicate_clusters: List of evidence ID lists that are duplicates
- correlation_warnings: List of dicts with cluster and note for correlated evidence
- analysis_process: Your step-by-step reasoning

## Quality Standards

- **Thorough**: Check every evidence item for correlations
- **Specific**: Point out exact problems, not vague issues
- **Actionable**: Suggest concrete follow-up searches
- **Balanced**: Consider both what's present and what's missing

## Important Guidelines

- **Use correlation detector** - Don't rely on manual inspection alone
- **Be systematic** - Check every subclaim for coverage
- **Prioritize gaps** - Focus on most important missing information
- **Suggest solutions** - Don't just criticize, recommend improvements
- **Consider quality** - Not just quantity of evidence

Remember: Your critique ensures the research is comprehensive and unbiased!
"""

    def get_tools(self) -> List[BaseTool]:
        """Return critic tools"""
        return [
            correlation_detector_tool,
        ]

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Check if critique is complete

        Criteria:
        - Correlation detection performed (if enough evidence)
        - Missing topics identified
        - Over-represented sources checked
        - Follow-up seeds generated
        """
        results = state.get('intermediate_results', {})

        # Check if correlation detection was performed
        evidence_count = len(state.get('task_input', {}).get('evidence_items', []))

        if evidence_count >= self.min_correlation_check_items:
            if 'correlation_warnings' not in results and 'duplicate_clusters' not in results:
                self.logger.info("Correlation detection not yet performed")
                return False

        # Check if we have the required outputs
        required_keys = [
            'missing_topics',
            'over_represented_sources',
            'follow_up_search_seeds'
        ]

        for key in required_keys:
            if key not in results:
                self.logger.info(f"Missing required output: {key}")
                return False

        self.logger.info("✅ Critique complete - all analysis performed")
        return True

    async def extract_final_output(self, state: AgentState) -> CriticOutput:
        """Extract CriticOutput from final state"""
        results = state.get('intermediate_results', {})

        # Build correlation warnings
        correlation_warnings = []
        for warning_data in results.get('correlation_warnings', []):
            if isinstance(warning_data, dict):
                correlation_warnings.append(CorrelationWarning(
                    cluster=warning_data.get('cluster', []),
                    note=warning_data.get('note', '')
                ))

        output = CriticOutput(
            duplicate_clusters=results.get('duplicate_clusters', []),
            missing_topics=results.get('missing_topics', []),
            over_represented_sources=results.get('over_represented_sources', []),
            correlation_warnings=correlation_warnings,
            follow_up_search_seeds=results.get('follow_up_search_seeds', [])
        )

        # Add analysis process if present
        if hasattr(output, 'analysis_process'):
            output.analysis_process = results.get('analysis_process', '')

        self.logger.info(
            f"📤 Critique complete: {len(output.correlation_warnings)} warnings, "
            f"{len(output.missing_topics)} gaps"
        )

        return output

    async def critique(
        self,
        evidence_items: List[Any],
        planner_output: Dict[str, Any],
        market_question: str
    ) -> CriticOutput:
        """
        Analyze evidence quality autonomously

        Args:
            evidence_items: List of evidence items to critique
            planner_output: Original research plan
            market_question: Market question

        Returns:
            CriticOutput with analysis and recommendations
        """
        return await self.run(
            task_description="Analyze evidence for quality, balance, and completeness",
            task_input={
                'evidence_items': evidence_items,
                'planner_output': planner_output,
                'market_question': market_question
            }
        )
