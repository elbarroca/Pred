#!/usr/bin/env python3
"""
Test the Simplified Planner Agent
"""
import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.agents.simplified_planner import SimplifiedPlannerAgent


async def main():
    """Test simplified planner on Diplo 5k market"""

    market_question = "Will Diplo run 5k in under 23 minutes?"

    print("=" * 80)
    print("🧪 TESTING SIMPLIFIED PLANNER AGENT")
    print("=" * 80)
    print(f"\nMarket Question: {market_question}\n")

    # Create planner
    planner = SimplifiedPlannerAgent()

    # Generate plan
    print("🤔 Generating research plan...\n")
    plan = await planner.plan(
        market_question=market_question,
        market_url="https://polymarket.com/event/diplo-5k",
        market_slug="diplo-5k-under-23-minutes"
    )

    # Display results
    print("=" * 80)
    print("✅ PLAN GENERATED")
    print("=" * 80)

    print(f"\n📊 Market Type: {plan.market_type}")

    print(f"\n👤 Subject to Profile:")
    print(f"   Name: {plan.subject_to_profile.entity_name}")
    print(f"   Type: {plan.subject_to_profile.entity_type}")

    print(f"\n❓ Core Research Questions ({len(plan.core_research_questions)}):")
    for i, question in enumerate(plan.core_research_questions, 1):
        print(f"   {i}. {question}")

    print(f"\n📈 Baseline Prior: {plan.baseline_prior:.2%}")
    print(f"   Reasoning: {plan.prior_reasoning}")

    print("\n" + "=" * 80)
    print("✅ TEST PASSED - Planner works correctly!")
    print("=" * 80)

    # Save to file for inspection
    output_path = Path("reports") / f"{plan.market_slug}_plan.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(plan.model_dump(), f, indent=2, default=str)

    print(f"\n💾 Plan saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
