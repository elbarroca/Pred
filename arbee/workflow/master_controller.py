"""
Master Workflow Controller for POLYSEER
Orchestrates two main modes:
1. Fast Arbitrage Scan - Quick cross-platform opportunity detection
2. Deep Bayesian Analysis - Full research + reasoning workflow
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient
from arbee.agents.autonomous_arbitrage import AutonomousArbitrageAgent
from arbee.workflow.market_selector import MarketSelector, fetch_and_score_markets
from arbee.workflow.autonomous_graph import run_autonomous_workflow
from config.settings import settings

logger = logging.getLogger(__name__)


class POLYSEERController:
    """
    Master controller for POLYSEER system

    Two main execution modes:
    - Mode 1: Arbitrage Scanner (fast, no research)
    - Mode 2: Deep Analysis (research + reasoning + Bayesian)
    """

    def __init__(self):
        """Initialize controller with API clients"""
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient()
        self.arbitrage_detector = AutonomousArbitrageAgent()
        self.market_selector = MarketSelector()

    async def run_arbitrage_scan(
        self,
        limit: int = 50,
        min_profit: float = 0.01,
        parallel_limit: int = 10,
        save_to_db: bool = False,
        filter_by_priority: bool = True
    ) -> Dict[str, Any]:
        """
        MODE 1: Fast Arbitrage Scanner

        Quickly scan markets for cross-platform arbitrage opportunities
        without running full Bayesian analysis.

        Args:
            limit: Max markets to fetch from each platform
            min_profit: Minimum profit threshold (0.01 = 1%)
            parallel_limit: Max concurrent API requests
            save_to_db: Whether to save results to database
            filter_by_priority: Use market selector to prioritize high-value markets

        Returns:
            Dict with:
                - opportunities: List of arbitrage opportunities found
                - markets_scanned: Number of markets checked
                - execution_time: Time taken
                - timestamp: When scan completed
        """
        start_time = datetime.now()
        scan_id = str(uuid.uuid4())

        logger.info(f"[{scan_id}] Starting arbitrage scan (limit={limit}, min_profit={min_profit:.1%})")

        # Step 1: Fetch markets from both platforms
        logger.info("[1/4] Fetching markets from platforms...")
        pm_markets = await self.polymarket.gamma.get_markets(active=True, limit=limit)
        kalshi_markets = await self.kalshi.get_markets(status="open", limit=limit)

        logger.info(f"✓ Fetched {len(pm_markets)} Polymarket + {len(kalshi_markets)} Kalshi markets")

        # Step 2: Prioritize markets if filtering enabled
        if filter_by_priority:
            logger.info("[2/4] Prioritizing markets by liquidity/urgency...")
            scored_markets = await fetch_and_score_markets(
                pm_markets,
                kalshi_markets,
                self.market_selector,
                top_n=min(30, limit)  # Take top 30 for arbitrage check
            )

            # Use top-scored markets from Polymarket
            markets_to_scan = [m for m, score in scored_markets['polymarket']]
            logger.info(f"✓ Selected {len(markets_to_scan)} high-priority markets")
        else:
            markets_to_scan = pm_markets
            logger.info("[2/4] Skipping prioritization (using all markets)")

        # Step 3: Scan for arbitrage opportunities
        logger.info(f"[3/4] Scanning {len(markets_to_scan)} markets for arbitrage...")
        opportunities = await self.arbitrage_detector.scan_markets_for_arbitrage(
            markets=markets_to_scan,
            providers=["polymarket", "kalshi"],
            min_profit_threshold=min_profit,
            parallel_limit=parallel_limit
        )

        # Flatten opportunities list
        all_opps = []
        for result in opportunities:
            for opp in result['opportunities']:
                all_opps.append({
                    'market': result['market'],
                    'opportunity': opp
                })

        logger.info(f"✓ Found {len(all_opps)} arbitrage opportunities")

        # Step 4: Save to database if requested
        if save_to_db:
            logger.info("[4/4] Saving opportunities to database...")
            # TODO: Implement database save
            logger.warning("Database save not yet implemented")
        else:
            logger.info("[4/4] Skipping database save")

        execution_time = (datetime.now() - start_time).total_seconds()

        result = {
            'scan_id': scan_id,
            'opportunities': all_opps,
            'markets_scanned': len(markets_to_scan),
            'markets_fetched': len(pm_markets) + len(kalshi_markets),
            'opportunities_found': len(all_opps),
            'execution_time_seconds': execution_time,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'limit': limit,
                'min_profit': min_profit,
                'parallel_limit': parallel_limit,
                'priority_filtering': filter_by_priority
            }
        }

        logger.info(
            f"[{scan_id}] Scan complete: {len(all_opps)} opportunities in {execution_time:.1f}s"
        )

        return result

    async def run_deep_analysis(
        self,
        market_slug: str,
        market_question: str,
        market_url: str = "",
        providers: List[str] = None,
        bankroll: float = None,
        save_to_db: bool = False
    ) -> Dict[str, Any]:
        """
        MODE 2: Deep Bayesian Analysis

        Run full POLYSEER workflow:
        - Research with Valyu (real-world evidence)
        - Agents reason with Chain-of-Thought
        - Bayesian probability aggregation
        - Compare vs market prices for mispricing

        Args:
            market_slug: Market identifier
            market_question: Question text
            market_url: URL to market (optional)
            providers: Platforms to check for arbitrage
            bankroll: Available capital for position sizing
            save_to_db: Save evidence and analysis to database

        Returns:
            Dict with complete workflow output including reporter summary
        """
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(f"[{analysis_id}] Starting deep analysis for: {market_question}")

        # Set defaults
        providers = providers or ["polymarket", "kalshi"]
        bankroll = bankroll or settings.DEFAULT_BANKROLL

        # Run the full LangGraph workflow
        logger.info("Running complete POLYSEER workflow...")
        try:
            workflow_result = await run_autonomous_workflow(
                market_question=market_question,
                market_url=market_url,
                market_slug=market_slug,
                providers=providers,
                bankroll=bankroll
            )

            logger.info("✓ Workflow completed successfully")

            # Extract key results
            result = {
                'analysis_id': analysis_id,
                'market_question': market_question,
                'p_bayesian': workflow_result.get('p_bayesian'),
                'arbitrage_opportunities': workflow_result.get('arbitrage_output', []),
                'full_report': workflow_result.get('reporter_output'),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'workflow_outputs': {
                    'planner': workflow_result.get('planner_output'),
                    'researcher': workflow_result.get('researcher_output'),
                    'critic': workflow_result.get('critic_output'),
                    'analyst': workflow_result.get('analyst_output'),
                    'arbitrage': workflow_result.get('arbitrage_output'),
                    'reporter': workflow_result.get('reporter_output')
                }
            }

            # Save to database if requested
            if save_to_db:
                logger.info("Saving analysis results to database...")
                # TODO: Implement database save
                logger.warning("Database save not yet implemented")

            logger.info(f"[{analysis_id}] Analysis complete: p_bayesian={result['p_bayesian']:.2%}")

            return result

        except Exception as e:
            logger.error(f"[{analysis_id}] Workflow failed: {e}")
            raise

    async def auto_scan_and_analyze(
        self,
        scan_limit: int = 100,
        analyze_top_n: int = 3,
        min_arbitrage_profit: float = 0.01,
        save_to_db: bool = False
    ) -> Dict[str, Any]:
        """
        COMBINED MODE: Auto scan + analyze top opportunities

        1. Run fast arbitrage scan to find opportunities
        2. Select top N opportunities by expected value
        3. Run deep analysis on each to validate with research

        Args:
            scan_limit: Markets to scan in initial sweep
            analyze_top_n: How many top opportunities to analyze deeply
            min_arbitrage_profit: Minimum profit to consider
            save_to_db: Save all results to database

        Returns:
            Dict with both scan results and deep analysis results
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(f"[{session_id}] Starting auto scan+analyze (scan={scan_limit}, analyze_top={analyze_top_n})")

        # Step 1: Run arbitrage scan
        logger.info("[AUTO 1/2] Running arbitrage scan...")
        scan_results = await self.run_arbitrage_scan(
            limit=scan_limit,
            min_profit=min_arbitrage_profit,
            save_to_db=False,  # Will save all at end
            filter_by_priority=True
        )

        opportunities = scan_results['opportunities']

        if not opportunities:
            logger.warning("No arbitrage opportunities found in scan. Skipping deep analysis.")
            return {
                'session_id': session_id,
                'scan_results': scan_results,
                'deep_analyses': [],
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }

        # Step 2: Sort opportunities by profit and select top N
        opportunities.sort(
            key=lambda x: x['opportunity'].guaranteed_profit or 0,
            reverse=True
        )
        top_opportunities = opportunities[:analyze_top_n]

        logger.info(f"[AUTO 2/2] Running deep analysis on top {len(top_opportunities)} opportunities...")

        # Step 3: Run deep analysis on each
        deep_analyses = []
        for i, opp_data in enumerate(top_opportunities, 1):
            market = opp_data['market']
            market_slug = market.get('slug', market.get('id'))
            market_question = market.get('question', market.get('title'))

            logger.info(f"[AUTO 2/2] Analyzing #{i}/{len(top_opportunities)}: {market_question[:60]}...")

            try:
                analysis_result = await self.run_deep_analysis(
                    market_slug=market_slug,
                    market_question=market_question,
                    save_to_db=False
                )
                deep_analyses.append(analysis_result)
            except Exception as e:
                logger.error(f"Deep analysis failed for {market_slug}: {e}")
                deep_analyses.append({
                    'market_slug': market_slug,
                    'error': str(e)
                })

        # Save all results to database if requested
        if save_to_db:
            logger.info("Saving all results to database...")
            # TODO: Implement batch save
            logger.warning("Database save not yet implemented")

        execution_time = (datetime.now() - start_time).total_seconds()

        result = {
            'session_id': session_id,
            'scan_results': scan_results,
            'deep_analyses': deep_analyses,
            'execution_time_seconds': execution_time,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"[{session_id}] Auto session complete: "
            f"{len(opportunities)} opportunities found, "
            f"{len(deep_analyses)} analyzed in {execution_time:.1f}s"
        )

        return result


# Convenience functions
async def quick_scan(limit: int = 50, min_profit: float = 0.01) -> Dict[str, Any]:
    """Quick arbitrage scan with default settings"""
    controller = POLYSEERController()
    return await controller.run_arbitrage_scan(limit=limit, min_profit=min_profit)


async def analyze_market(market_slug: str, market_question: str) -> Dict[str, Any]:
    """Run deep analysis on a single market"""
    controller = POLYSEERController()
    return await controller.run_deep_analysis(market_slug, market_question)


async def auto_mode(scan_limit: int = 100, analyze_top: int = 3) -> Dict[str, Any]:
    """Auto scan and analyze top opportunities"""
    controller = POLYSEERController()
    return await controller.auto_scan_and_analyze(scan_limit=scan_limit, analyze_top_n=analyze_top)
