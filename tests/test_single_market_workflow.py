#!/usr/bin/env python3
"""
POLYSEER Single Market Deep Analysis Test
==========================================
Tests the complete workflow on a single Polymarket market:
1. Fetch market data from Polymarket
2. Run Planner Agent → Generate research plan
3. Run Researcher Agents → Use Valyu to gather PRO/CON evidence
4. Run Critic Agent → Detect correlations and gaps
5. Run Analyst Agent → Calculate Bayesian probability
6. Compare with market price → Identify arbitrage opportunity

This demonstrates the full POLYSEER workflow as described in CLAUDE.MD
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.valyu import ValyuResearchClient
from arbee.agents.planner import PlannerAgent
from arbee.agents.researcher import ResearcherAgent
from arbee.agents.critic import CriticAgent
from arbee.agents.analyst import AnalystAgent
from config.settings import settings


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def print_step(step_num: int, total: int, description: str):
    """Print workflow step"""
    print(f"[{step_num}/{total}] {description}")


async def fetch_polymarket_market(limit: int = 20) -> Optional[Dict[str, Any]]:
    """
    Step 1: Fetch a single interesting market from Polymarket

    Returns:
        Market dict with question, current price, and metadata
    """
    print_step(1, 7, "Fetching Polymarket markets...")

    client = PolymarketClient()

    # Fetch markets with real liquidity data from CLOB API
    print("🔄 Fetching markets with real liquidity data...")
    markets = await client.get_markets_with_real_liquidity(
        active=True,
        limit=min(limit * 5, 100),
        min_liquidity=100  # $100 minimum liquidity
    )

    if not markets:
        print("❌ No tradeable markets found on Polymarket")
        print("💡 Trying with very low liquidity threshold...")
        markets = await client.get_markets_with_real_liquidity(
            active=True,
            limit=min(limit * 5, 100),
            min_liquidity=10  # $10 minimum liquidity
        )

    if not markets:
        print("❌ No suitable markets found")
        return None

    print(f"✓ Found {len(markets)} tradeable markets with real liquidity")

    # Markets already filtered by CLOB API for real liquidity
    # Additional filtering for binary markets and current topics
    filtered_markets = []

    for market in markets:
        outcomes = market.get('outcomes', [])

        # Only consider binary markets
        if len(outcomes) != 2:
            continue

        # Check for current year topics (2024-2025)
        question = market.get('question', '').lower()
        description = market.get('description', '').lower()

        # Prioritize current year topics and avoid very old topics
        is_current = any(str(year) in question or str(year) in description
                       for year in [2025, 2024])
        is_old = any(str(year) in question or str(year) in description
                    for year in [2020, 2021, 2022, 2023])

        if is_old and not is_current:
            continue  # Skip old markets unless they mention current years

        # Calculate selection score with real liquidity
        real_liquidity = market.get('real_liquidity', 0)
        real_volume = market.get('real_volume', 0)
        recency_bonus = 2 if is_current else 1
        score = (real_liquidity + real_volume) * recency_bonus

        market['_selection_score'] = score
        filtered_markets.append(market)

    if not filtered_markets:
        print("⚠️  No suitable binary markets found in current results")
        # Fall back to first market if no filtering matches
        filtered_markets = markets

    # Sort by selection score
    filtered_markets.sort(key=lambda m: m.get('_selection_score', 0), reverse=True)

    selected = filtered_markets[0]
    print(f"✓ Selected market with ${selected.get('real_liquidity', 0):,.0f} real liquidity")

    # Display market info
    question = selected.get('question', 'Unknown')
    real_liquidity = selected.get('real_liquidity', 0)
    real_volume = selected.get('real_volume', 0)
    midpoint_price = selected.get('midpoint_price', 0.5)
    outcomes = selected.get('outcomes', [])

    print(f"\n✓ Selected market:")
    print(f"  Question: {question}")
    print(f"  Outcomes: {outcomes}")
    print(f"  Midpoint price: {midpoint_price:.1%}")
    print(f"  Real liquidity: ${real_liquidity:,.0f}")
    print(f"  Real volume: ${real_volume:,.0f}")

    # Get implied probability from midpoint price
    implied_prob = midpoint_price
    print(f"  Implied probability: {implied_prob:.1%}")

    return selected


async def run_planner(market: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Run Planner Agent to create research plan

    Returns:
        PlannerOutput with prior, subclaims, search seeds
    """
    print_step(2, 7, "Running Planner Agent...")

    planner = PlannerAgent()
    question = market.get('question', '')

    print(f"  Analyzing: {question}")

    # Run planner
    result = await planner.plan(market_question=question)

    print(f"\n✓ Research plan created:")
    print(f"  Prior p0: {result.p0_prior:.2f}")
    print(f"  Subclaims: {len(result.subclaims)}")
    print(f"  Search seeds: {len(result.search_seeds.pro)} PRO, {len(result.search_seeds.con)} CON")
    print(f"  Justification: {result.prior_justification}")

    return result


async def run_researchers(planner_output, market_question: str) -> Dict[str, Any]:
    """
    Step 3: Run Researcher Agents to gather evidence using Valyu

    Returns:
        Dict with pro_evidence and con_evidence lists
    """
    print_step(3, 7, "Running Researcher Agents (searching web via Valyu)...")

    # Create separate researcher agents for PRO and CON directions
    pro_researcher = ResearcherAgent(direction="pro")
    con_researcher = ResearcherAgent(direction="con")

    print("  Gathering PRO evidence...")
    pro_evidence = await pro_researcher.research(
        search_seeds=planner_output.search_seeds.pro,
        subclaims=planner_output.subclaims,
        market_question=market_question,
        max_evidence_per_seed=3
    )

    print(f"✓ PRO evidence: {len(pro_evidence.evidence_items)} items found")

    print("  Gathering CON evidence...")
    con_evidence = await con_researcher.research(
        search_seeds=planner_output.search_seeds.con,
        subclaims=planner_output.subclaims,
        market_question=market_question,
        max_evidence_per_seed=3
    )

    print(f"✓ CON evidence: {len(con_evidence.evidence_items)} items found")

    # Combine evidence
    all_evidence = pro_evidence.evidence_items + con_evidence.evidence_items

    print(f"\n✓ Total evidence collected: {len(all_evidence)} items")
    print(f"  PRO sources: {len([e for e in all_evidence if e.support == 'pro'])}")
    print(f"  CON sources: {len([e for e in all_evidence if e.support == 'con'])}")
    print(f"  Neutral sources: {len([e for e in all_evidence if e.support == 'neutral'])}")

    return {
        'pro_evidence': pro_evidence,
        'con_evidence': con_evidence,
        'all_evidence': all_evidence
    }


async def run_critic(evidence_data: Dict[str, Any], planner_output, market_question: str) -> Dict[str, Any]:
    """
    Step 4: Run Critic Agent to analyze evidence quality

    Returns:
        CriticOutput with correlation warnings, missing topics, etc.
    """
    print_step(4, 7, "Running Critic Agent...")

    critic = CriticAgent()

    result = await critic.analyze(
        evidence_items=evidence_data['all_evidence'],
        planner_output=planner_output,
        market_question=market_question
    )

    print(f"\n✓ Critic analysis complete:")
    print(f"  Duplicate clusters: {len(result.duplicate_clusters)}")
    print(f"  Missing topics: {result.missing_topics}")
    print(f"  Over-represented sources: {result.over_represented_sources}")
    print(f"  Correlation warnings: {len(result.correlation_warnings)}")

    if result.correlation_warnings:
        print(f"\n  Correlation warnings:")
        for warning in result.correlation_warnings[:3]:
            print(f"    - {warning.note}")

    return result


async def run_analyst(
    planner_output,
    evidence_data: Dict[str, Any],
    critic_output,
    market_question: str
) -> Dict[str, Any]:
    """
    Step 5: Run Analyst Agent to calculate Bayesian probability

    Returns:
        AnalystOutput with p_bayesian, log-odds calculations, sensitivity
    """
    print_step(5, 7, "Running Analyst Agent (Bayesian calculation)...")

    analyst = AnalystAgent()

    result = await analyst.analyze(
        prior_p=planner_output.p0_prior,
        evidence_items=evidence_data['all_evidence'],
        critic_output=critic_output,
        market_question=market_question
    )

    print(f"\n✓ Bayesian aggregation complete:")
    print(f"  p_bayesian: {result.p_bayesian:.1%}")
    print(f"  Prior (p0): {result.p0:.1%}")
    print(f"  Evidence items used: {len(result.evidence_summary)}")

    # Show sensitivity analysis
    if result.sensitivity_analysis:
        print(f"\n  Sensitivity analysis:")
        for scenario in result.sensitivity_analysis[:3]:
            print(f"    {scenario.scenario}: {scenario.p:.1%}")

    return result


def analyze_arbitrage(market: Dict[str, Any], analyst_output) -> Dict[str, Any]:
    """
    Step 6: Compare Bayesian probability with market price

    Returns:
        Arbitrage analysis with edge, EV, Kelly fraction
    """
    print_step(6, 7, "Arbitrage Analysis...")

    # Get market implied probability from real CLOB data
    market_prob = market.get('midpoint_price', 0.5)

    # Bayesian estimate
    bayesian_prob = analyst_output.p_bayesian

    # Calculate edge
    edge = bayesian_prob - market_prob

    # Expected value per $1 bet
    ev_per_dollar = edge

    # Kelly fraction (conservative, capped at 5%)
    kelly_fraction = min(edge / (1 - market_prob), 0.05) if edge > 0 else 0

    # Suggested stake for $10,000 bankroll
    bankroll = 10000
    suggested_stake = bankroll * kelly_fraction

    print(f"\n✓ Arbitrage comparison:")
    print(f"  Market price: {market_prob:.1%}")
    print(f"  Bayesian estimate: {bayesian_prob:.1%}")
    print(f"  Edge: {edge:+.1%}")
    print(f"  Expected value: ${ev_per_dollar:+.2f} per $1 bet")
    print(f"  Kelly fraction: {kelly_fraction:.2%}")
    print(f"  Suggested stake (on ${bankroll:,}): ${suggested_stake:.2f}")

    # Determine if opportunity exists
    has_opportunity = abs(edge) > 0.02  # 2% threshold

    if has_opportunity:
        if edge > 0:
            print(f"\n💡 POTENTIAL OPPORTUNITY: Market underpricing YES outcome by {edge:.1%}")
            print(f"   Consider buying YES at {market_prob:.1%} (Bayesian: {bayesian_prob:.1%})")
        else:
            print(f"\n💡 POTENTIAL OPPORTUNITY: Market overpricing YES outcome by {-edge:.1%}")
            print(f"   Consider buying NO at {1-market_prob:.1%} (Bayesian NO: {1-bayesian_prob:.1%})")
    else:
        print(f"\n❌ No significant arbitrage opportunity (edge = {edge:+.1%}, threshold = ±2%)")

    return {
        'market_prob': market_prob,
        'bayesian_prob': bayesian_prob,
        'edge': edge,
        'ev_per_dollar': ev_per_dollar,
        'kelly_fraction': kelly_fraction,
        'suggested_stake': suggested_stake,
        'has_opportunity': has_opportunity
    }


def display_summary(
    market: Dict[str, Any],
    planner_output,
    evidence_data: Dict[str, Any],
    critic_output,
    analyst_output,
    arbitrage_analysis: Dict[str, Any],
    execution_time: float
):
    """
    Step 7: Display comprehensive summary
    """
    print_step(7, 7, "Summary Report")

    print(f"\n{'='*80}")
    print("POLYSEER DEEP ANALYSIS - RESULTS")
    print(f"{'='*80}\n")

    print(f"📊 MARKET:")
    print(f"   {market.get('question', 'Unknown')}")
    print(f"   Current price: {arbitrage_analysis['market_prob']:.1%}")
    print(f"   Real liquidity: ${market.get('real_liquidity', 0):,.0f}")

    print(f"\n🔍 RESEARCH:")
    print(f"   Prior p0: {planner_output.p0_prior:.1%}")
    print(f"   Evidence items: {len(evidence_data['all_evidence'])}")
    print(f"   Correlation warnings: {len(critic_output.correlation_warnings)}")

    print(f"\n🧮 BAYESIAN ANALYSIS:")
    print(f"   p_bayesian: {analyst_output.p_bayesian:.1%}")
    print(f"   Edge vs market: {arbitrage_analysis['edge']:+.1%}")

    print(f"\n💰 ARBITRAGE:")
    if arbitrage_analysis['has_opportunity']:
        print(f"   ✓ Opportunity detected!")
        print(f"   Expected value: ${arbitrage_analysis['ev_per_dollar']:+.2f} per $1")
        print(f"   Kelly stake: ${arbitrage_analysis['suggested_stake']:.2f}")
    else:
        print(f"   ❌ No significant opportunity (edge < 2%)")

    print(f"\n⏱️  EXECUTION TIME: {execution_time:.1f}s")

    print(f"\n{'='*80}")
    print("NOT FINANCIAL ADVICE. Research only.")
    print(f"{'='*80}\n")


async def main():
    """Main workflow execution"""
    print_header("POLYSEER DEEP ANALYSIS - SINGLE MARKET TEST")

    start_time = datetime.now()

    try:
        # Step 1: Fetch market
        market = await fetch_polymarket_market(limit=20)
        if not market:
            print("❌ Failed to fetch market")
            return

        # Step 2: Run Planner
        planner_output = await run_planner(market)

        # Step 3: Run Researchers
        evidence_data = await run_researchers(
            planner_output,
            market.get('question', '')
        )

        # Step 4: Run Critic
        critic_output = await run_critic(
            evidence_data,
            planner_output,
            market.get('question', '')
        )

        # Step 5: Run Analyst
        analyst_output = await run_analyst(
            planner_output,
            evidence_data,
            critic_output,
            market.get('question', '')
        )

        # Step 6: Analyze arbitrage
        arbitrage_analysis = analyze_arbitrage(market, analyst_output)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Step 7: Display summary
        display_summary(
            market,
            planner_output,
            evidence_data,
            critic_output,
            analyst_output,
            arbitrage_analysis,
            execution_time
        )

        # Save results to JSON
        results = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'market': {
                'question': market.get('question'),
                'current_price': arbitrage_analysis['market_prob'],
                'real_liquidity': market.get('real_liquidity', 0),
                'real_volume': market.get('real_volume', 0),
                'midpoint_price': market.get('midpoint_price', 0.5)
            },
            'planner': {
                'prior_p0': planner_output.p0_prior,
                'subclaims_count': len(planner_output.subclaims),
                'search_seeds_count': len(planner_output.search_seeds.pro) + len(planner_output.search_seeds.con)
            },
            'research': {
                'total_evidence': len(evidence_data['all_evidence']),
                'pro_count': len([e for e in evidence_data['all_evidence'] if e.support == 'pro']),
                'con_count': len([e for e in evidence_data['all_evidence'] if e.support == 'con'])
            },
            'critic': {
                'duplicate_clusters': len(critic_output.duplicate_clusters),
                'correlation_warnings': len(critic_output.correlation_warnings),
                'missing_topics': critic_output.missing_topics
            },
            'analyst': {
                'p_bayesian': analyst_output.p_bayesian,
                'p0': analyst_output.p0,
                'evidence_count': len(analyst_output.evidence_summary)
            },
            'arbitrage': arbitrage_analysis
        }

        output_path = Path(__file__).parent / f"polyseer_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Full results saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
