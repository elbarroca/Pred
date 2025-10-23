#!/usr/bin/env python3
"""
POLYSEER CLI - Command-line interface for arbitrage scanning and market analysis
"""
import asyncio
import sys
import argparse
from typing import Optional
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.workflow.master_controller import POLYSEERController
from config.settings import settings


async def cmd_scan(args):
    """Run arbitrage scanner"""
    print("\n" + "="*80)
    print("POLYSEER ARBITRAGE SCANNER")
    print("="*80 + "\n")

    controller = POLYSEERController()

    result = await controller.run_arbitrage_scan(
        limit=args.limit,
        min_profit=args.min_profit,
        parallel_limit=args.parallel,
        save_to_db=args.save_db,
        filter_by_priority=not args.no_filter
    )

    # Display results
    opportunities = result['opportunities']

    print(f"\n📊 SCAN RESULTS")
    print(f"   Markets scanned: {result['markets_scanned']}")
    print(f"   Opportunities found: {len(opportunities)}")
    print(f"   Execution time: {result['execution_time_seconds']:.1f}s\n")

    if not opportunities:
        print("❌ No arbitrage opportunities found above threshold.\n")
        print("Try:")
        print("  • Lower the profit threshold: --min-profit 0.005")
        print("  • Scan more markets: --limit 200")
        print("  • Disable filtering: --no-filter\n")
        return

    # Sort by profit
    opportunities.sort(
        key=lambda x: x['opportunity'].guaranteed_profit or 0,
        reverse=True
    )

    # Display top opportunities
    display_limit = min(args.show_top, len(opportunities))
    print(f"🎯 TOP {display_limit} OPPORTUNITIES:\n")

    for i, opp_data in enumerate(opportunities[:display_limit], 1):
        opp = opp_data['opportunity']
        market = opp_data['market']

        print(f"[{i}] {market['question'][:70]}...")
        print(f"    Platforms: {' ↔ '.join(opp.platform_pair)}")
        print(f"    Strategy: Buy {opp.side_a.outcome} on {opp.side_a.platform} ({opp.side_a.price:.1%})")
        print(f"              Buy {opp.side_b.outcome} on {opp.side_b.platform} ({opp.side_b.price:.1%})")
        print(f"    Profit: {opp.guaranteed_profit*100:.2f}% (${opp.suggested_stake * opp.guaranteed_profit:.2f} on ${opp.suggested_stake:.0f})")
        print()

    # Save to file if requested
    if args.output:
        output_data = {
            'scan_results': result,
            'opportunities': [
                {
                    'market': opp['market'],
                    'opportunity': opp['opportunity'].dict()
                }
                for opp in opportunities
            ]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"✓ Results saved to {args.output}\n")

    print("NOT FINANCIAL ADVICE. Research only.\n")


async def cmd_analyze(args):
    """Run deep Bayesian analysis on a market"""
    print("\n" + "="*80)
    print("POLYSEER DEEP ANALYSIS")
    print("="*80 + "\n")

    controller = POLYSEERController()

    print(f"Analyzing: {args.question}\n")
    print("⏳ Running full workflow (this may take 2-5 minutes)...\n")

    result = await controller.run_deep_analysis(
        market_slug=args.slug,
        market_question=args.question,
        market_url=args.url or "",
        save_to_db=args.save_db
    )

    print("\n📊 ANALYSIS COMPLETE\n")
    print(f"   Bayesian Probability: {result['p_bayesian']:.1%}")
    print(f"   Execution time: {result['execution_time_seconds']:.1f}s\n")

    # Display arbitrage opportunities if found
    arb_opps = result.get('arbitrage_opportunities', [])
    if arb_opps:
        print(f"💰 Found {len(arb_opps)} arbitrage opportunities\n")
        for i, opp in enumerate(arb_opps[:3], 1):
            print(f"   [{i}] {opp.provider}: EV = {opp.expected_value_per_dollar:.2%}")

    # Display reporter summary
    if result.get('full_report'):
        report = result['full_report']
        print(f"\n📝 SUMMARY")
        print(f"   {report.tldr}\n")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"✓ Full analysis saved to {args.output}\n")

    print("NOT FINANCIAL ADVICE. Research only.\n")


async def cmd_auto(args):
    """Auto mode: scan + analyze top opportunities"""
    print("\n" + "="*80)
    print("POLYSEER AUTO MODE")
    print("="*80 + "\n")

    controller = POLYSEERController()

    print(f"⏳ Running auto scan+analyze (this may take 5-15 minutes)...\n")

    result = await controller.auto_scan_and_analyze(
        scan_limit=args.scan_limit,
        analyze_top_n=args.analyze_top,
        min_arbitrage_profit=args.min_profit,
        save_to_db=args.save_db
    )

    # Display results
    scan = result['scan_results']
    analyses = result['deep_analyses']

    print("\n📊 AUTO MODE RESULTS\n")
    print(f"   Markets scanned: {scan['markets_scanned']}")
    print(f"   Opportunities found: {len(scan['opportunities'])}")
    print(f"   Deep analyses completed: {len(analyses)}")
    print(f"   Total execution time: {result['execution_time_seconds']:.1f}s\n")

    # Show each analysis
    for i, analysis in enumerate(analyses, 1):
        if 'error' in analysis:
            print(f"[{i}] ❌ {analysis['market_slug']}: {analysis['error']}")
        else:
            print(f"[{i}] ✓ {analysis['market_question'][:60]}...")
            print(f"     p_bayesian: {analysis['p_bayesian']:.1%}")
            print()

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"✓ Results saved to {args.output}\n")

    print("NOT FINANCIAL ADVICE. Research only.\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="POLYSEER - Autonomous Bayesian Arbitrage Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick arbitrage scan
  python run_polyseer.py scan --limit 100 --min-profit 0.01

  # Analyze specific market
  python run_polyseer.py analyze --slug "trump-2024" --question "Will Trump win 2024?"

  # Auto mode: scan and analyze top 3
  python run_polyseer.py auto --scan-limit 200 --analyze-top 3

  # Save results to JSON
  python run_polyseer.py scan --limit 50 --output results.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # SCAN command
    scan_parser = subparsers.add_parser('scan', help='Fast arbitrage scanner')
    scan_parser.add_argument('--limit', type=int, default=50, help='Markets to fetch per platform')
    scan_parser.add_argument('--min-profit', type=float, default=0.01, help='Minimum profit threshold (0.01 = 1%%)')
    scan_parser.add_argument('--parallel', type=int, default=10, help='Concurrent API requests')
    scan_parser.add_argument('--no-filter', action='store_true', help='Skip priority filtering')
    scan_parser.add_argument('--show-top', type=int, default=10, help='Show top N opportunities')
    scan_parser.add_argument('--save-db', action='store_true', help='Save to database')
    scan_parser.add_argument('--output', type=str, help='Save JSON results to file')

    # ANALYZE command
    analyze_parser = subparsers.add_parser('analyze', help='Deep Bayesian analysis')
    analyze_parser.add_argument('--slug', required=True, help='Market slug/identifier')
    analyze_parser.add_argument('--question', required=True, help='Market question')
    analyze_parser.add_argument('--url', type=str, help='Market URL (optional)')
    analyze_parser.add_argument('--save-db', action='store_true', help='Save to database')
    analyze_parser.add_argument('--output', type=str, help='Save JSON results to file')

    # AUTO command
    auto_parser = subparsers.add_parser('auto', help='Auto scan + analyze top opportunities')
    auto_parser.add_argument('--scan-limit', type=int, default=100, help='Markets to scan')
    auto_parser.add_argument('--analyze-top', type=int, default=3, help='Analyze top N opportunities')
    auto_parser.add_argument('--min-profit', type=float, default=0.01, help='Minimum profit threshold')
    auto_parser.add_argument('--save-db', action='store_true', help='Save to database')
    auto_parser.add_argument('--output', type=str, help='Save JSON results to file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run appropriate command
    try:
        if args.command == 'scan':
            asyncio.run(cmd_scan(args))
        elif args.command == 'analyze':
            asyncio.run(cmd_analyze(args))
        elif args.command == 'auto':
            asyncio.run(cmd_auto(args))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
