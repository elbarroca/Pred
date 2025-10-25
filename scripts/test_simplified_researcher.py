#!/usr/bin/env python3
"""
Test the Simplified Researcher Agent with 3-Phase Adaptive Search
"""
import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.agents.simplified_researcher import SimplifiedResearcherAgent


async def main():
    """Test simplified researcher on Diplo 5k market"""

    market_question = "Will Diplo run 5k in under 23 minutes?"
    subject_name = "Diplo"
    entity_type = "person"
    research_questions = [
        "Who is Diplo and what is their fitness/running background?",
        "What is a competitive 5k time for someone of Diplo's age and profile?",
        "Has Diplo run any races recently? What were the times?",
        "Is Diplo currently training for this specific goal?"
    ]

    print("=" * 80)
    print("🧪 TESTING SIMPLIFIED RESEARCHER AGENT")
    print("=" * 80)
    print(f"\nMarket Question: {market_question}")
    print(f"Subject: {subject_name} ({entity_type})\n")

    # Create researcher
    researcher = SimplifiedResearcherAgent(max_searches=10)

    # Execute research
    print("🔬 Starting 3-phase adaptive research...\n")
    result = await researcher.research(
        subject_name=subject_name,
        entity_type=entity_type,
        market_question=market_question,
        research_questions=research_questions
    )

    # Display results
    print("\n" + "=" * 80)
    print("✅ RESEARCH RESULTS")
    print("=" * 80)

    print(f"\n👤 Subject Profile:")
    print(f"   Name: {result.subject_profile.entity_name}")
    print(f"   Type: {result.subject_profile.entity_type}")
    print(f"   Key Facts: {json.dumps(result.subject_profile.key_facts, indent=4)}")
    print(f"   Baseline: {result.subject_profile.baseline_capabilities}")

    print(f"\n📊 Research Stats:")
    print(f"   Total searches: {result.total_searches}")
    print(f"   Final confidence: {result.research_phase.confidence:.2%}")
    print(f"   Phase: {result.research_phase.phase}")

    print(f"\n📋 Benchmark Evidence ({len(result.benchmark_evidence)} items):")
    for i, evidence in enumerate(result.benchmark_evidence[:3], 1):
        print(f"   {i}. [{evidence.relevance_score}/10] {evidence.key_fact[:80]}")
        print(f"      → {evidence.support_direction}, primary={evidence.is_primary}")

    print(f"\n🎯 Specific Evidence ({len(result.specific_evidence)} items):")
    for i, evidence in enumerate(result.specific_evidence[:5], 1):
        print(f"   {i}. [{evidence.relevance_score}/10] {evidence.key_fact[:80]}")
        print(f"      → {evidence.support_direction}, primary={evidence.is_primary}")

    print(f"\n🔍 Search Queries Used:")
    for phase_queries in result.search_queries_used:
        print(f"   {phase_queries['phase'].upper()}:")
        print(f"      Reasoning: {phase_queries['reasoning'][:100]}")
        for query in phase_queries['queries']:
            print(f"      - \"{query}\"")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    # Save to file
    output_path = Path("reports") / "diplo-5k-under-23-minutes_research.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2, default=str)

    print(f"\n💾 Research saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
