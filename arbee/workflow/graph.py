"""
LangGraph Workflow Orchestration for POLYSEER
Coordinates all agents in the proper sequence with parallel execution
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from arbee.models.schemas import WorkflowState
from arbee.agents.planner import PlannerAgent
from arbee.agents.researcher import run_parallel_research
from arbee.agents.critic import CriticAgent
from arbee.agents.analyst import AnalystAgent
from arbee.agents.arbitrage import ArbitrageDetector
from arbee.agents.reporter import ReporterAgent

from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner Agent Node
    Decomposes market question into research plan
    """
    logger.info("=== PLANNER NODE ===")

    agent = PlannerAgent()

    try:
        result = await agent.plan(
            market_question=state['market_question'],
            market_url=state.get('market_url', ''),
            market_slug=state.get('market_slug', ''),
            context=state.get('context', {})
        )

        logger.info(f"Planner completed: {len(result.subclaims)} subclaims, prior={result.p0_prior:.2%}")

        return {
            'planner_output': result,
            'p0_prior': result.p0_prior,
            'search_seeds': result.search_seeds,
            'subclaims': result.subclaims
        }

    except Exception as e:
        logger.error(f"Planner node failed: {e}")
        raise


async def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Researcher Agents Node (Parallel Execution)
    Runs PRO, CON, and GENERAL researchers in parallel
    """
    logger.info("=== RESEARCHER NODE (PARALLEL) ===")

    planner_output = state['planner_output']
    search_seeds = planner_output.search_seeds
    subclaims = [sc.model_dump() for sc in planner_output.subclaims]

    try:
        # Run all researchers in parallel
        results = await run_parallel_research(
            search_seeds_pro=search_seeds.pro,
            search_seeds_con=search_seeds.con,
            search_seeds_general=search_seeds.general,
            subclaims=subclaims,
            market_question=state['market_question']
        )

        # Combine all evidence
        all_evidence = []
        for direction in ['pro', 'con', 'general']:
            all_evidence.extend(results[direction].evidence)

        logger.info(
            f"Researchers completed: {len(all_evidence)} total evidence items "
            f"(PRO: {len(results['pro'].evidence)}, "
            f"CON: {len(results['con'].evidence)}, "
            f"GENERAL: {len(results['general'].evidence)})"
        )

        return {
            'researcher_output': results,
            'all_evidence': all_evidence
        }

    except Exception as e:
        logger.error(f"Researcher node failed: {e}")
        raise


async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critic Agent Node
    Analyzes evidence for correlations and gaps
    """
    logger.info("=== CRITIC NODE ===")

    agent = CriticAgent()

    try:
        result = await agent.critique(
            evidence_items=state['all_evidence'],
            planner_output=state['planner_output'].model_dump(),
            market_question=state['market_question']
        )

        logger.info(
            f"Critic completed: {len(result.correlation_warnings)} warnings, "
            f"{len(result.missing_topics)} gaps"
        )

        return {'critic_output': result}

    except Exception as e:
        logger.error(f"Critic node failed: {e}")
        raise


async def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyst Agent Node
    Performs Bayesian aggregation
    """
    logger.info("=== ANALYST NODE ===")

    agent = AnalystAgent()

    try:
        result = await agent.analyze(
            prior_p=state['p0_prior'],
            evidence_items=state['all_evidence'],
            critic_output=state['critic_output'],
            market_question=state['market_question']
        )

        logger.info(f"Analyst completed: p_bayesian={result.p_bayesian:.2%}")

        return {
            'analyst_output': result,
            'p_bayesian': result.p_bayesian
        }

    except Exception as e:
        logger.error(f"Analyst node failed: {e}")
        raise


async def arbitrage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Arbitrage Detector Node
    Finds mispricing opportunities
    """
    logger.info("=== ARBITRAGE NODE ===")

    agent = ArbitrageDetector()

    try:
        opportunities = await agent.detect_arbitrage(
            p_bayesian=state['p_bayesian'],
            market_slug=state.get('market_slug', ''),
            market_question=state['market_question'],
            providers=state.get('providers', ['polymarket', 'kalshi']),
            bankroll=state.get('bankroll', settings.DEFAULT_BANKROLL),
            max_kelly=state.get('max_kelly', settings.MAX_KELLY_FRACTION),
            min_edge_threshold=state.get('min_edge_threshold', settings.MIN_EDGE_THRESHOLD)
        )

        logger.info(f"Arbitrage completed: {len(opportunities)} opportunities found")

        return {'arbitrage_output': opportunities}

    except Exception as e:
        logger.error(f"Arbitrage node failed: {e}")
        # Don't fail the entire workflow if arbitrage detection fails
        logger.warning("Continuing without arbitrage analysis")
        return {'arbitrage_output': []}


async def reporter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reporter Agent Node
    Generates final JSON + Markdown report
    """
    logger.info("=== REPORTER NODE ===")

    agent = ReporterAgent()

    try:
        result = await agent.generate_report(
            market_question=state['market_question'],
            planner_output=state['planner_output'].model_dump(),
            researcher_output={
                k: v.model_dump() for k, v in state['researcher_output'].items()
            },
            critic_output=state['critic_output'].model_dump(),
            analyst_output=state['analyst_output'].model_dump(),
            arbitrage_opportunities=[opp.model_dump() for opp in state['arbitrage_output']],
            timestamp=state.get('timestamp', datetime.now().isoformat()),
            workflow_id=state.get('workflow_id', '')
        )

        logger.info("Reporter completed: Final report generated")

        return {'reporter_output': result}

    except Exception as e:
        logger.error(f"Reporter node failed: {e}")
        raise


# ============================================================================
# WORKFLOW CONSTRUCTION
# ============================================================================

def create_polyseer_workflow() -> StateGraph:
    """
    Create the POLYSEER LangGraph workflow

    Workflow Structure:
    START → planner → researcher (parallel) → critic → analyst → arbitrage → reporter → END

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating POLYSEER workflow graph")

    # Create graph with WorkflowState
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("arbitrage", arbitrage_node)
    workflow.add_node("reporter", reporter_node)

    # Define edges (linear flow, researchers execute parallel internally)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "analyst")
    workflow.add_edge("analyst", "arbitrage")
    workflow.add_edge("arbitrage", "reporter")
    workflow.add_edge("reporter", END)

    # Compile with checkpointing
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("Workflow graph compiled successfully")

    return app


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def run_workflow(
    market_question: str,
    market_url: str = "",
    market_slug: str = "",
    providers: List[str] = None,
    bankroll: float = None,
    max_kelly: float = None,
    min_edge_threshold: float = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the complete POLYSEER workflow

    Args:
        market_question: The prediction market question to analyze
        market_url: URL to the market (optional)
        market_slug: Market identifier (optional)
        providers: List of platforms to check for arbitrage
        bankroll: Available capital for position sizing
        max_kelly: Maximum Kelly fraction (default 5%)
        min_edge_threshold: Minimum edge to report arbitrage (default 2%)
        **kwargs: Additional context

    Returns:
        Dict with all workflow outputs including final report

    Example:
        >>> result = await run_workflow(
        ...     market_question="Will Donald Trump win the 2024 US Presidential Election?",
        ...     market_url="https://polymarket.com/event/trump-2024",
        ...     market_slug="trump-2024",
        ...     providers=["polymarket", "kalshi"],
        ...     bankroll=10000.0
        ... )
        >>> print(result['reporter_output'].executive_summary)
    """
    workflow_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    logger.info(f"Starting POLYSEER workflow {workflow_id}")
    logger.info(f"Market Question: {market_question}")

    # Create workflow graph
    app = create_polyseer_workflow()

    # Prepare initial state
    initial_state = {
        'workflow_id': workflow_id,
        'timestamp': timestamp,
        'market_question': market_question,
        'market_url': market_url,
        'market_slug': market_slug or market_question.lower().replace(' ', '-')[:50],
        'providers': providers or ['polymarket', 'kalshi'],
        'bankroll': bankroll or settings.DEFAULT_BANKROLL,
        'max_kelly': max_kelly or settings.MAX_KELLY_FRACTION,
        'min_edge_threshold': min_edge_threshold or settings.MIN_EDGE_THRESHOLD,
        'context': kwargs
    }

    # Execute workflow
    config = {"configurable": {"thread_id": workflow_id}}

    try:
        final_state = await app.ainvoke(initial_state, config)

        logger.info(f"Workflow {workflow_id} completed successfully")

        return final_state

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        raise


async def run_workflow_step_by_step(
    market_question: str,
    **kwargs
):
    """
    Run workflow with step-by-step streaming (useful for UI progress tracking)

    Args:
        market_question: The prediction market question
        **kwargs: Additional workflow parameters

    Yields:
        (node_name, output) tuples as each node completes
    """
    workflow_id = str(uuid.uuid4())
    app = create_polyseer_workflow()

    initial_state = {
        'workflow_id': workflow_id,
        'timestamp': datetime.now().isoformat(),
        'market_question': market_question,
        'market_url': kwargs.get('market_url', ''),
        'market_slug': kwargs.get('market_slug', ''),
        **kwargs
    }

    config = {"configurable": {"thread_id": workflow_id}}

    async for output in app.astream(initial_state, config):
        yield output
