"""
Autonomous ReporterAgent
Generates final reports with autonomous validation
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import ReporterOutput

logger = logging.getLogger(__name__)


class AutonomousReporterAgent(AutonomousReActAgent):
    """
    Autonomous Reporter Agent - Generates comprehensive final reports

    Autonomous Capabilities:
    - Validates completeness of all inputs
    - Generates executive summary
    - Creates structured JSON + Markdown outputs
    - Ensures proper disclaimers
    - Iteratively refines if validation fails
    """

    def get_system_prompt(self) -> str:
        """System prompt for report generation"""
        return """You are an Autonomous Reporter Agent in POLYSEER.

Your mission: Generate comprehensive, accurate final reports with full provenance.

## Process

1. **Gather All Outputs**
   - Planner output (prior, subclaims, search seeds)
   - Researcher outputs (evidence items)
   - Critic output (warnings, gaps)
   - Analyst output (p_bayesian, sensitivity)
   - Arbitrage opportunities

2. **Generate Executive Summary**
   - Key finding: p_bayesian with confidence interval
   - Top 3 PRO drivers (strongest evidence supporting YES)
   - Top 3 CON drivers (strongest evidence supporting NO)
   - Critical uncertainties and gaps
   - Arbitrage summary (if opportunities found)
   - 200-600 words

3. **Create TL;DR**
   - 1-2 sentence summary of key finding
   - Include p_bayesian and main driver

4. **Validate Completeness**
   - All sections present
   - Provenance (URLs, sources) included
   - Disclaimers added
   - Proper formatting

5. **Add Required Disclaimers**
   - "NOT FINANCIAL ADVICE"
   - "Research only"
   - Limitations and uncertainties

Store in intermediate_results:
- executive_summary: string
- tldr: string
- key_findings: list
- provenance: dict
- disclaimers: list

Complete when all sections validated and formatted.
"""

    def get_tools(self) -> List[BaseTool]:
        """Reporter doesn't need special tools currently"""
        return []

    async def is_task_complete(self, state: AgentState) -> bool:
        """Check if report is complete"""
        results = state.get('intermediate_results', {})
        required_keys = ['executive_summary', 'tldr', 'key_findings']
        return all(key in results for key in required_keys)

    async def extract_final_output(self, state: AgentState) -> ReporterOutput:
        """Extract ReporterOutput from state"""
        results = state.get('intermediate_results', {})
        task_input = state.get('task_input', {})

        output = ReporterOutput(
            market_question=task_input.get('market_question', ''),
            workflow_id=task_input.get('workflow_id', ''),
            timestamp=task_input.get('timestamp', ''),
            executive_summary=results.get('executive_summary', ''),
            tldr=results.get('tldr', ''),
            key_findings=results.get('key_findings', []),
            planner_summary=results.get('planner_summary', {}),
            research_summary=results.get('research_summary', {}),
            critic_summary=results.get('critic_summary', {}),
            analyst_summary=results.get('analyst_summary', {}),
            arbitrage_summary=results.get('arbitrage_summary', {}),
            provenance=results.get('provenance', {}),
            disclaimers=results.get('disclaimers', [
                "NOT FINANCIAL ADVICE",
                "Research and educational purposes only"
            ])
        )

        self.logger.info(f"📤 Report generated: {len(output.executive_summary)} chars")
        return output

    async def generate_report(
        self,
        market_question: str,
        planner_output: Dict[str, Any],
        researcher_output: Dict[str, Any],
        critic_output: Dict[str, Any],
        analyst_output: Dict[str, Any],
        arbitrage_opportunities: List[Dict[str, Any]],
        timestamp: str,
        workflow_id: str
    ) -> ReporterOutput:
        """
        Generate comprehensive report autonomously

        Args:
            market_question: Market question
            planner_output: Planner results
            researcher_output: Research results
            critic_output: Critique results
            analyst_output: Analysis results
            arbitrage_opportunities: Arbitrage opportunities
            timestamp: Workflow timestamp
            workflow_id: Workflow ID

        Returns:
            ReporterOutput with complete report
        """
        return await self.run(
            task_description="Generate comprehensive market analysis report",
            task_input={
                'market_question': market_question,
                'planner_output': planner_output,
                'researcher_output': researcher_output,
                'critic_output': critic_output,
                'analyst_output': analyst_output,
                'arbitrage_opportunities': arbitrage_opportunities,
                'timestamp': timestamp,
                'workflow_id': workflow_id
            }
        )
