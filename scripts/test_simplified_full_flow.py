#!/usr/bin/env python3
"""
Test Complete Simplified Flow: Planner → Researcher → Analyst
End-to-end test with Bayesian probability estimation
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.agents.simplified_planner import SimplifiedPlannerAgent
from arbee.agents.simplified_researcher import SimplifiedResearcherAgent
from arbee.agents.simplified_analyst import SimplifiedAnalystAgent


async def main():
    """Run complete simplified flow on Diplo 5k market"""

    market_question = "Will Diplo run 5k in under 23 minutes?"
    market_url = "https://polymarket.com/event/diplo-5k"

    print("=" * 80)
    print("🚀 POLYSEER SIMPLIFIED SYSTEM - FULL FLOW TEST")
    print("=" * 80)
    print(f"\nMarket Question: {market_question}")
    print(f"Market URL: {market_url}\n")

    # ============================================================================
    # PHASE 1: PLANNER
    # ============================================================================
    print("\n" + "=" * 80)
    print("📋 PHASE 1: PLANNER - Generating Research Plan")
    print("=" * 80)

    planner = SimplifiedPlannerAgent()
    plan = await planner.plan(
        market_question=market_question,
        market_url=market_url,
        market_slug="diplo-5k-under-23-minutes"
    )

    print(f"\n✅ Plan generated:")
    print(f"   Market type: {plan.market_type}")
    print(f"   Subject: {plan.subject_to_profile.entity_name} ({plan.subject_to_profile.entity_type})")
    print(f"   Research questions: {len(plan.core_research_questions)}")
    for i, q in enumerate(plan.core_research_questions, 1):
        print(f"      {i}. {q}")
    print(f"   Baseline prior: {plan.baseline_prior:.2%}")

    # ============================================================================
    # PHASE 2: RESEARCHER
    # ============================================================================
    print("\n" + "=" * 80)
    print("🔬 PHASE 2: RESEARCHER - 3-Phase Adaptive Research")
    print("=" * 80)

    researcher = SimplifiedResearcherAgent(max_searches=10)
    research_output = await researcher.research(
        subject_name=plan.subject_to_profile.entity_name,
        entity_type=plan.subject_to_profile.entity_type,
        market_question=market_question,
        research_questions=plan.core_research_questions
    )

    print(f"\n✅ Research complete:")
    print(f"   Total searches: {research_output.total_searches}")
    print(f"   Benchmark evidence: {len(research_output.benchmark_evidence)} items")
    print(f"   Specific evidence: {len(research_output.specific_evidence)} items")
    print(f"   Final confidence: {research_output.research_phase.confidence:.2%}")
    print(f"\n   Subject Profile:")
    print(f"      Name: {research_output.subject_profile.entity_name}")
    print(f"      Age: {research_output.subject_profile.key_facts.get('age', 'Unknown')}")
    print(f"      Profession: {research_output.subject_profile.key_facts.get('profession', 'Unknown')}")
    print(f"      Baseline: {research_output.subject_profile.baseline_capabilities or 'Unknown'}")

    # ============================================================================
    # PHASE 3: ANALYST
    # ============================================================================
    print("\n" + "=" * 80)
    print("🧮 PHASE 3: ANALYST - Bayesian Probability Estimation")
    print("=" * 80)

    analyst = SimplifiedAnalystAgent()
    analysis = await analyst.analyze(
        market_question=market_question,
        subject_profile=research_output.subject_profile,
        benchmark_evidence=research_output.benchmark_evidence,
        specific_evidence=research_output.specific_evidence,
        baseline_prior=plan.baseline_prior
    )

    print(f"\n✅ Analysis complete:")
    print(f"\n   📊 FINAL PROBABILITY: {analysis.probability:.2%}")
    print(f"   📊 CONFIDENCE: {analysis.confidence:.2%}")
    print(f"   📊 SENSITIVITY RANGE: {analysis.sensitivity_range['low']:.2%} - {analysis.sensitivity_range['high']:.2%}")
    print(f"\n   Evidence Breakdown:")
    print(f"      YES evidence weight: {analysis.yes_evidence_weight:.2f}")
    print(f"      NO evidence weight: {analysis.no_evidence_weight:.2f}")
    print(f"      NEUTRAL evidence weight: {analysis.neutral_evidence_weight:.2f}")
    print(f"      Total evidence: {analysis.total_evidence_count} items")
    print(f"      Primary sources: {analysis.primary_evidence_count} items")

    print(f"\n   💭 Reasoning:")
    print(f"      {analysis.reasoning}")

    if analysis.top_yes_evidence:
        print(f"\n   ✅ Top YES Evidence:")
        for i, e in enumerate(analysis.top_yes_evidence[:3], 1):
            print(f"      {i}. [{e['relevance']}/10, weight={e['weight']:.1f}] {e['key_fact'][:80]}")

    if analysis.top_no_evidence:
        print(f"\n   ❌ Top NO Evidence:")
        for i, e in enumerate(analysis.top_no_evidence[:3], 1):
            print(f"      {i}. [{e['relevance']}/10, weight={e['weight']:.1f}] {e['key_fact'][:80]}")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("🎉 COMPLETE ANALYSIS")
    print("=" * 80)
    print(f"\nMarket: {market_question}")
    print(f"\n🎯 FINAL ANSWER:")
    print(f"   Probability: {analysis.probability:.2%}")
    print(f"   Confidence: {analysis.confidence:.2%}")
    print(f"   Range: {analysis.sensitivity_range['low']:.2%} - {analysis.sensitivity_range['high']:.2%}")

    print(f"\n📈 System Performance:")
    print(f"   Total searches: {research_output.total_searches}")
    print(f"   Evidence items: {analysis.total_evidence_count}")
    print(f"   Primary sources: {analysis.primary_evidence_count}")
    print(f"   Research phases: 3 (profiling → benchmarking → evidence)")

    print(f"\n📊 Recommendation:")
    if analysis.confidence < 0.3:
        print(f"   ⚠️ LOW CONFIDENCE - Insufficient evidence for reliable prediction")
    elif analysis.confidence < 0.6:
        print(f"   ⚡ MODERATE CONFIDENCE - Some evidence, but significant uncertainty")
    else:
        print(f"   ✅ HIGH CONFIDENCE - Strong evidence base for prediction")

    if analysis.probability > 0.7:
        rec = "LIKELY YES"
    elif analysis.probability > 0.5:
        rec = "LEAN YES"
    elif analysis.probability > 0.3:
        rec = "LEAN NO"
    else:
        rec = "LIKELY NO"

    print(f"   Prediction: {rec} ({analysis.probability:.1%})")

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    output_path = Path("reports") / f"diplo-5k-full-analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)

    full_output = {
        "market_question": market_question,
        "market_url": market_url,
        "timestamp": datetime.now().isoformat(),
        "plan": plan.model_dump(),
        "research": research_output.model_dump(),
        "analysis": analysis.model_dump()
    }

    with open(output_path, 'w') as f:
        json.dump(full_output, f, indent=2, default=str)

    print(f"\n💾 Full analysis saved to: {output_path}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE - SIMPLIFIED SYSTEM WORKING END-TO-END")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
