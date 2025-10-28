"""
Autonomous LangGraph Workflow for POLYSEER
Uses autonomous reasoning agents with tool use, memory, and iterative loops.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from arbee.models.schemas import WorkflowState
from arbee.agents.autonomous_planner import AutonomousPlannerAgent
from arbee.agents.autonomous_researcher import run_parallel_autonomous_research
from arbee.agents.autonomous_critic import AutonomousCriticAgent
from arbee.agents.autonomous_analyst import AutonomousAnalystAgent
from arbee.agents.autonomous_arbitrage import AutonomousArbitrageAgent
from arbee.agents.autonomous_reporter import AutonomousReporterAgent
from arbee.utils.memory import get_memory_manager
from config.settings import settings

# Keep logging minimal per module: creation + finalization of workflow.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Small helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _trace(agent: str, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"agent_name": agent, "timestamp": _now_iso(), "action": action, "details": details}


def _msg(text: str) -> List[tuple[str, str]]:
    return [("assistant", text)]


# -----------------------------
# AUTONOMOUS AGENT NODE FUNCTIONS
# -----------------------------
async def autonomous_planner_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Plan subclaims and initial priors with bounded iteration and memory.
    """
    assert "market_question" in state, "state.market_question is required"

    cfg = (state.get("context") or {}).get("planner_agent", {})
    agent = AutonomousPlannerAgent(
        model_name=cfg.get("model_name", "gpt-4o-mini"),
        temperature=cfg.get("temperature", 0.1),
        enable_auto_memory_query=store is not None,
        store=store,
        max_iterations=cfg.get("max_iterations", 10),
        min_subclaims=cfg.get("min_subclaims", 4),
        max_subclaims=cfg.get("max_subclaims", 10),
        auto_extend_iterations=cfg.get("auto_extend_iterations", True),
        iteration_extension=cfg.get("iteration_extension", 5),
        max_iteration_cap=cfg.get("max_iteration_cap", 40),
        recursion_limit=cfg.get("recursion_limit", 100),
    )

    result = await agent.plan(
        market_question=state["market_question"],
        market_url=state.get("market_url", ""),
        market_slug=state.get("market_slug", ""),
        context=state.get("context", {}),
        max_iterations=cfg.get("run_max_iterations"),
    )

    # Store brief planning memory when available
    if store:
        mm = get_memory_manager(store=store)
        await mm.store_episode_memory(
            episode_id=state.get("workflow_id", ""),
            market_question=state["market_question"],
            memory_type="planning_strategy",
            content={
                "prior": result.p0_prior,
                "subclaim_count": len(result.subclaims),
                "balance": {
                    "pro": sum(1 for sc in result.subclaims if sc.direction == "pro"),
                    "con": sum(1 for sc in result.subclaims if sc.direction == "con"),
                },
            },
            effectiveness=0.9,
        )

    stats = agent.get_stats()
    trace = {
        "subclaim_count": len(result.subclaims),
        "search_seed_counts": {
            "pro": len(result.search_seeds.pro),
            "con": len(result.search_seeds.con),
            "general": len(result.search_seeds.general),
        },
        "prior": result.p0_prior,
        "iterations": stats.get("average_iterations", 0),
        "tool_calls": stats.get("total_tool_calls", 0),
    }

    return {
        "planner_output": result,
        "p0_prior": result.p0_prior,
        "search_seeds": result.search_seeds,
        "subclaims": result.subclaims,
        "messages": _msg(f"Autonomous Planner: {trace['subclaim_count']} subclaims, {trace['iterations']:.1f} iterations"),
        "agent_traces": [_trace("AutonomousPlannerAgent", "autonomous_planning_completed", trace)],
    }


async def autonomous_researcher_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Run pro/con/general researcher agents in parallel and collect evidence.
    """
    assert "planner_output" in state, "state.planner_output is required"
    planner_out = state["planner_output"]
    seeds = planner_out.search_seeds
    subclaims = [sc.model_dump() for sc in planner_out.subclaims]

    results = await run_parallel_autonomous_research(
        search_seeds_pro=seeds.pro,
        search_seeds_con=seeds.con,
        search_seeds_general=seeds.general,
        subclaims=subclaims,
        market_question=state["market_question"],
        store=store,
        min_evidence_items=5,
        max_search_attempts=10,
        max_iterations=15,
    )

    all_evidence: List[Any] = []
    counts: Dict[str, int] = {}
    for d in ("pro", "con", "general"):
        items = results[d].evidence_items
        all_evidence.extend(items)
        counts[d] = len(items)

    if store and len(all_evidence) >= 10:
        mm = get_memory_manager(store=store)
        await mm.store_episode_memory(
            episode_id=state.get("workflow_id", ""),
            market_question=state["market_question"],
            memory_type="research_strategy",
            content={
                "search_seeds_used": {"pro": seeds.pro, "con": seeds.con, "general": seeds.general},
                "evidence_counts": counts,
                "total_evidence": len(all_evidence),
            },
            effectiveness=min(1.0, len(all_evidence) / 15),
        )

    details = {"total_items": len(all_evidence), "direction_counts": counts}
    return {
        "researcher_output": results,
        "all_evidence": all_evidence,
        "messages": _msg(f"Autonomous Researchers: {len(all_evidence)} evidence items ({counts['pro']}P, {counts['con']}C, {counts['general']}G)"),
        "agent_traces": [_trace("AutonomousResearcherAgents", "autonomous_research_completed", details)],
    }


async def autonomous_critic_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Critique evidence for quality, duplication, and gaps.
    """
    assert "all_evidence" in state and "planner_output" in state, "state missing evidence/planner_output"

    agent = AutonomousCriticAgent(store=store, max_iterations=15, min_correlation_check_items=3, enable_auto_memory_query=store is not None)
    result = await agent.critique(
        evidence_items=state["all_evidence"],
        planner_output=state["planner_output"].model_dump(),
        market_question=state["market_question"],
    )

    stats = agent.get_stats()
    trace = {
        "duplicate_clusters": len(result.duplicate_clusters),
        "missing_topics": len(result.missing_topics),
        "over_represented_sources": len(result.over_represented_sources),
        "iterations": stats.get("average_iterations", 0),
    }

    return {
        "critic_output": result,
        "messages": _msg(f"Autonomous Critic: {trace['duplicate_clusters']} correlations, {trace['missing_topics']} gaps"),
        "agent_traces": [_trace("AutonomousCriticAgent", "autonomous_critique_completed", trace)],
    }


async def autonomous_analyst_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Aggregate evidence Bayesianly with sensitivity checks.
    """
    assert "p0_prior" in state and "all_evidence" in state and "critic_output" in state, "state missing prior/evidence/critic_output"

    agent = AutonomousAnalystAgent(store=store, max_iterations=15, max_sensitivity_range=0.3, enable_auto_memory_query=store is not None)
    result = await agent.analyze(
        prior_p=state["p0_prior"],
        evidence_items=state["all_evidence"],
        critic_output=state["critic_output"],
        market_question=state["market_question"],
    )

    stats = agent.get_stats()
    trace = {
        "p_bayesian": result.p_bayesian,
        "evidence_items": len(result.evidence_summary),
        "sensitivity_scenarios": len(result.sensitivity_analysis),
        "iterations": stats.get("average_iterations", 0),
    }

    return {
        "analyst_output": result,
        "p_bayesian": result.p_bayesian,
        "messages": _msg(f"Autonomous Analyst: p={result.p_bayesian:.2%} (from {result.p0:.2%})"),
        "agent_traces": [_trace("AutonomousAnalystAgent", "autonomous_analysis_completed", trace)],
    }


async def autonomous_arbitrage_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Detect actionable mispricing opportunities.
    """
    assert "p_bayesian" in state and "market_question" in state, "state missing p_bayesian/market_question"

    agent = AutonomousArbitrageAgent(
        store=store,
        max_iterations=10,
        min_edge_threshold=state.get("min_edge_threshold", settings.MIN_EDGE_THRESHOLD),
        enable_auto_memory_query=store is not None,
    )
    try:
        opportunities = await agent.detect_arbitrage(
            p_bayesian=state["p_bayesian"],
            market_slug=state.get("market_slug", ""),
            market_question=state["market_question"],
            providers=state.get("providers", ["polymarket", "kalshi"]),
            bankroll=state.get("bankroll", settings.DEFAULT_BANKROLL),
            max_kelly=state.get("max_kelly", settings.MAX_KELLY_FRACTION),
            min_edge_threshold=state.get("min_edge_threshold", settings.MIN_EDGE_THRESHOLD),
        )
        details = {"opportunity_count": len(opportunities), "providers_checked": state.get("providers", [])}
        return {
            "arbitrage_output": opportunities,
            "messages": _msg(f"Autonomous Arbitrage: {len(opportunities)} opportunities"),
            "agent_traces": [_trace("AutonomousArbitrageAgent", "autonomous_arbitrage_completed", details)],
        }
    except Exception:
        # Preserve original behavior: continue without arbitrage on failure.
        return {"arbitrage_output": []}


async def autonomous_reporter_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Generate a final report from all upstream outputs.
    """
    assert {"market_question", "planner_output", "researcher_output", "critic_output", "analyst_output", "arbitrage_output"} <= state.keys(), "state missing required keys for report"

    agent = AutonomousReporterAgent(store=store, max_iterations=10, enable_auto_memory_query=store is not None)
    result = await agent.generate_report(
        market_question=state["market_question"],
        planner_output=state["planner_output"].model_dump(),
        researcher_output={k: v.model_dump() for k, v in state["researcher_output"].items()},
        critic_output=state["critic_output"].model_dump(),
        analyst_output=state["analyst_output"].model_dump(),
        arbitrage_opportunities=[opp.model_dump() for opp in state["arbitrage_output"]],
        timestamp=state.get("timestamp", datetime.now().isoformat()),
        workflow_id=state.get("workflow_id", ""),
    )

    return {
        "reporter_output": result,
        "messages": _msg("Autonomous Reporter: Final report complete"),
        "agent_traces": [_trace("AutonomousReporterAgent", "autonomous_report_completed", {"summary_length": len(result.executive_summary or '')})],
    }


# -----------------------------
# AUTONOMOUS WORKFLOW CONSTRUCTION
# -----------------------------
def create_autonomous_polyseer_workflow(store: Optional[BaseStore] = None) -> StateGraph:
    """
    Create and compile the POLYSEER workflow with autonomous agents.
    START → planner → researcher → critic → analyst → arbitrage → reporter → END
    """
    logger.info("🚀 Creating AUTONOMOUS POLYSEER workflow")

    workflow = StateGraph(WorkflowState)

    async def planner_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_planner_node(s, store)

    async def researcher_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_researcher_node(s, store)

    async def critic_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_critic_node(s, store)

    async def analyst_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_analyst_node(s, store)

    async def arbitrage_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_arbitrage_node(s, store)

    async def reporter_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_reporter_node(s, store)

    workflow.add_node("planner", planner_wrapper)
    workflow.add_node("researcher", researcher_wrapper)
    workflow.add_node("critic", critic_wrapper)
    workflow.add_node("analyst", analyst_wrapper)
    workflow.add_node("arbitrage", arbitrage_wrapper)
    workflow.add_node("reporter", reporter_wrapper)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "analyst")
    workflow.add_edge("analyst", "arbitrage")
    workflow.add_edge("arbitrage", "reporter")
    workflow.add_edge("reporter", END)

    app = workflow.compile(checkpointer=MemorySaver(), debug=False)
    logger.info("✅ Autonomous workflow compiled successfully")
    return app


# -----------------------------
# CONVENIENCE RUNNER
# -----------------------------
async def run_autonomous_workflow(
    market_question: str,
    market_url: str = "",
    market_slug: str = "",
    providers: Optional[List[str]] = None,
    bankroll: Optional[float] = None,
    max_kelly: Optional[float] = None,
    min_edge_threshold: Optional[float] = None,
    store: Optional[BaseStore] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute the complete autonomous workflow and return the final state.
    """
    assert isinstance(market_question, str) and market_question.strip(), "market_question is required"

    workflow_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    logger.info(f"🚀 Starting AUTONOMOUS POLYSEER workflow {workflow_id}")

    app = create_autonomous_polyseer_workflow(store=store)
    initial_state = {
        "workflow_id": workflow_id,
        "timestamp": timestamp,
        "market_question": market_question,
        "market_url": market_url,
        "market_slug": market_slug or market_question.lower().replace(" ", "-")[:50],
        "providers": providers or ["polymarket", "kalshi"],
        "bankroll": bankroll or settings.DEFAULT_BANKROLL,
        "max_kelly": max_kelly or settings.MAX_KELLY_FRACTION,
        "min_edge_threshold": min_edge_threshold or settings.MIN_EDGE_THRESHOLD,
        "context": kwargs,
    }

    config = {"configurable": {"thread_id": workflow_id}, "recursion_limit": 150}
    final_state = await app.ainvoke(initial_state, config)
    logger.info(f"✅ Autonomous workflow {workflow_id} completed successfully")
    return final_state