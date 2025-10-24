"""
Autonomous ArbitrageDetector Agent
Finds mispricing opportunities with autonomous validation
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import ArbitrageOpportunity

logger = logging.getLogger(__name__)


class AutonomousArbitrageAgent(AutonomousReActAgent):
    """
    Autonomous Arbitrage Detector - Finds mispricing opportunities

    Note: This agent has simpler tool needs since ArbitrageDetector
    already has good logic. Main improvements:
    - Autonomous retry if API calls fail
    - Validation of opportunities before returning
    - Can query multiple exchanges iteratively
    """

    def __init__(self, min_edge_threshold: float = 0.02, **kwargs):
        """
        Initialize Autonomous Arbitrage Agent

        Args:
            min_edge_threshold: Minimum edge to report
            **kwargs: Additional args
        """
        super().__init__(**kwargs)
        self.min_edge_threshold = min_edge_threshold

    def get_system_prompt(self) -> str:
        """System prompt for arbitrage detection"""
        return f"""You are an Autonomous Arbitrage Detector in POLYSEER.

Your mission: Find mispricing opportunities between Bayesian estimate and market prices.

## Process

1. **Compare p_bayesian vs market prices**
   - Calculate edge = p_bayesian - market_implied_probability
   - Calculate EV considering transaction costs
   - Apply Kelly criterion for position sizing

2. **Validate opportunities**
   - Check edge > {self.min_edge_threshold} (minimum threshold)
   - Verify liquidity is sufficient
   - Consider transaction costs and slippage

3. **Generate recommendations**
   - For each opportunity: market, provider, edge, suggested_stake
   - Include rationale and risk warnings
   - Add disclaimer: NOT FINANCIAL ADVICE

Store results in intermediate_results['opportunities'] as list of dicts.

Complete when all markets checked and opportunities validated.
"""

    def get_tools(self) -> List[BaseTool]:
        """Arbitrage agent doesn't need special tools currently"""
        return []

    async def is_task_complete(self, state: AgentState) -> bool:
        """Check if arbitrage detection complete"""
        results = state.get('intermediate_results', {})
        return 'opportunities' in results

    async def extract_final_output(self, state: AgentState) -> List[ArbitrageOpportunity]:
        """Extract arbitrage opportunities from state"""
        results = state.get('intermediate_results', {})
        opportunities_data = results.get('opportunities', [])

        opportunities = []
        for opp_data in opportunities_data:
            if isinstance(opp_data, dict):
                opportunities.append(ArbitrageOpportunity(**opp_data))

        self.logger.info(f"📤 Found {len(opportunities)} arbitrage opportunities")
        return opportunities

    async def detect_arbitrage(
        self,
        p_bayesian: float,
        market_slug: str,
        market_question: str,
        providers: List[str],
        bankroll: float,
        max_kelly: float,
        min_edge_threshold: float
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities autonomously

        Args:
            p_bayesian: Bayesian posterior probability
            market_slug: Market identifier
            market_question: Market question
            providers: List of providers to check
            bankroll: Available capital
            max_kelly: Max Kelly fraction
            min_edge_threshold: Min edge to report

        Returns:
            List of arbitrage opportunities
        """
        return await self.run(
            task_description="Find mispricing opportunities across prediction markets",
            task_input={
                'p_bayesian': p_bayesian,
                'market_slug': market_slug,
                'market_question': market_question,
                'providers': providers,
                'bankroll': bankroll,
                'max_kelly': max_kelly,
                'min_edge_threshold': min_edge_threshold
            }
        )
