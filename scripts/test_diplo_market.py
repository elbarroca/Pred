#!/usr/bin/env python3
"""
Test Script: POLYSEER Deep Analysis on Diplo 5k Race Market
Based on market from Polymarket screenshot

Market Details:
- Question: "How fast will Diplo run 5k?"
- Sub-markets:
  - <23 minutes: 90% (Yes: 90¢, No: 11¢) - $6,354 vol
  - <22 minutes: 74% (Yes: 74¢, No: 27¢) - $4,947 vol
  - <21 minutes: 32% (Yes: 32¢, No: 69¢) - $22,995 vol
- Total volume: $34,296
- Event date: October 25, 2025
- Resolution: BibTag System official race results
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.workflow.master_controller import POLYSEERController
from config.settings import settings


async def main():
    """Run deep analysis on Diplo 5k market"""

    # Market from screenshot
    market_question = "Will Diplo run 5k in under 23 minutes?"
    market_slug = "diplo-5k-under-23-minutes"
    market_url = "https://polymarket.com/event/diplo-5k"

    print("\n" + "=" * 80)
    print("POLYSEER DEEP ANALYSIS: Diplo 5k Race Market")
    print("=" * 80)
    print(f"\nMarket Question: {market_question}")
    print(f"Market URL: {market_url}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize controller
    controller = POLYSEERController()

    print("⏳ Running complete POLYSEER workflow...")
    print("   This will take approximately 5-10 minutes\n")
    print("   Steps:")
    print("   [1/6] Planner: Breaking down question into research tasks")
    print("   [2/6] Researchers: Gathering evidence from internet (PRO/CON/GENERAL)")
    print("   [3/6] Critic: Analyzing evidence for correlations and gaps")
    print("   [4/6] Analyst: Performing Bayesian probability aggregation")
    print("   [5/6] Arbitrage: Detecting mispricing opportunities")
    print("   [6/6] Reporter: Generating final report\n")

    # Run deep analysis
    try:
        result = await controller.run_deep_analysis(
            market_slug=market_slug,
            market_question=market_question,
            market_url=market_url,
            providers=["polymarket"],
            bankroll=settings.DEFAULT_BANKROLL,
            save_to_db=True
        )

        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 80)

        # Display key results
        print(f"\n📊 KEY RESULTS:")
        print(f"   Bayesian Probability (p_bayesian): {result.get('p_bayesian', 0):.2%}")
        print(f"   Execution Time: {result.get('execution_time_seconds', 0):.1f}s")

        # Display workflow outputs
        if 'workflow_outputs' in result:
            outputs = result['workflow_outputs']

            # Planner output
            if 'planner' in outputs and outputs['planner']:
                planner = outputs['planner']
                print(f"\n📋 PLANNER OUTPUT:")
                print(f"   Prior (p0): {planner.get('p0_prior', 0):.2%}")
                print(f"   Subclaims generated: {len(planner.get('subclaims', []))}")
                print(f"   Search seeds: PRO={len(planner.get('search_seeds', {}).get('pro', []))}, "
                      f"CON={len(planner.get('search_seeds', {}).get('con', []))}, "
                      f"GENERAL={len(planner.get('search_seeds', {}).get('general', []))}")

            # Researcher output
            if 'researcher' in outputs and outputs['researcher']:
                researcher = outputs['researcher']
                total_evidence = sum(len(r.get('evidence_items', [])) for r in researcher.values() if isinstance(r, dict))
                print(f"\n🔍 RESEARCHER OUTPUT:")
                print(f"   Total evidence items: {total_evidence}")
                for direction in ['pro', 'con', 'general']:
                    if direction in researcher and researcher[direction]:
                        count = len(researcher[direction].get('evidence_items', []))
                        print(f"   {direction.upper()}: {count} items")

            # Analyst output
            if 'analyst' in outputs and outputs['analyst']:
                analyst = outputs['analyst']
                print(f"\n🧮 ANALYST OUTPUT:")
                print(f"   p_bayesian: {analyst.get('p_bayesian', 0):.2%}")
                print(f"   Confidence interval: [{analyst.get('p_neutral', 0) - 0.1:.2%}, {analyst.get('p_neutral', 0) + 0.1:.2%}]")
                print(f"   Evidence items analyzed: {len(analyst.get('evidence_summary', []))}")

            # Arbitrage output
            if 'arbitrage' in outputs and outputs['arbitrage']:
                opportunities = outputs['arbitrage']
                print(f"\n💰 ARBITRAGE OUTPUT:")
                print(f"   Opportunities found: {len(opportunities)}")
                if opportunities:
                    for i, opp in enumerate(opportunities[:3], 1):
                        print(f"   [{i}] {opp.get('provider', 'N/A')}: "
                              f"EV={opp.get('expected_value_per_dollar', 0):.2%}, "
                              f"Kelly={opp.get('kelly_fraction', 0):.2%}")

        # Save outputs
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Save JSON
        json_path = reports_dir / f"{market_slug}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n💾 OUTPUTS SAVED:")
        print(f"   JSON Report: {json_path}")

        # Save Markdown if reporter output exists
        if 'reporter_output' in result and result['reporter_output']:
            reporter = result['reporter_output']
            if 'executive_summary' in reporter:
                md_path = reports_dir / f"{market_slug}_analysis.md"
                with open(md_path, 'w') as f:
                    f.write(f"# POLYSEER Analysis: {market_question}\n\n")
                    f.write(f"**Analysis ID:** {result.get('analysis_id', 'N/A')}\n")
                    f.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}\n\n")
                    f.write(f"## Executive Summary\n\n")
                    f.write(reporter['executive_summary'])
                    f.write(f"\n\n## TL;DR\n\n")
                    f.write(reporter.get('tldr', 'N/A'))
                    f.write("\n\n---\n\n")
                    f.write("**NOT FINANCIAL ADVICE.** This is research only.\n")
                print(f"   Markdown Report: {md_path}")

        print(f"\n✓ Database: Evidence and analysis saved to Supabase")

        print("\n" + "=" * 80)
        print("NOT FINANCIAL ADVICE. This is research only.")
        print("=" * 80 + "\n")

        return result

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
