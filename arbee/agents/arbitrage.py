"""
Arbitrage Detector - Cross-platform price comparison and EV calculation
Identifies mispricing opportunities using Kelly Criterion
"""
from typing import Type, List, Dict, Any, Optional
from arbee.agents.base import BaseAgent
from arbee.agents.schemas import ArbitrageOpportunity, PlatformSide
from arbee.utils.bayesian import KellyCalculator
from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient
from config.settings import settings
from pydantic import BaseModel
import asyncio


class ArbitrageOpportunityList(BaseModel):
    """Wrapper for list of arbitrage opportunities"""
    opportunities: List[ArbitrageOpportunity]


class ArbitrageDetector(BaseAgent):
    """
    Arbitrage Detector - Finds mispricing across prediction markets

    Responsibilities:
    1. Fetch current prices from multiple platforms
    2. Compare p_bayesian vs market-implied probabilities
    3. Calculate expected value (EV) per dollar
    4. Compute Kelly fractions for position sizing
    5. Rank opportunities by EV and filter by thresholds

    Kelly Criterion:
    - kelly_fraction = edge / (1 - p_market)
    - edge = p_bayesian - p_market
    - Capped at 5% for conservative risk management

    This agent "thinks deeply" by:
    - Accounting for transaction costs and slippage
    - Considering liquidity constraints
    - Providing clear trade rationales
    - Highlighting risks and uncertainties
    - Always ending with "NOT FINANCIAL ADVICE"
    """

    def __init__(self, **kwargs):
        """Initialize with Kelly calculator and API clients"""
        super().__init__(**kwargs)
        self.kelly_calc = KellyCalculator()
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient()

    def get_system_prompt(self) -> str:
        """System prompt for arbitrage detection"""
        return """You are the Arbitrage Detector in ARBEE, a Bayesian arbitrage analysis system.

Your role is to identify mispricing opportunities across prediction markets.

## Core Responsibilities

1. **Calculate Edge**
   - edge = p_bayesian - p_market
   - Positive edge = market underprices YES
   - Negative edge = market overprices YES (bet NO)

2. **Expected Value (EV)**
   - EV = edge - transaction_costs - slippage
   - Must account for ALL costs
   - Flag opportunities where EV > 0

3. **Kelly Criterion Position Sizing**
   - kelly = edge / (1 - p_market)
   - Cap at max_kelly (default 5% of bankroll)
   - Provides optimal bet size for long-term growth

4. **Risk Assessment**
   - Check liquidity (can we actually place the bet?)
   - Estimate slippage for large orders
   - Consider counterparty risk
   - Flag uncertainties in p_bayesian

## Fee Structures (Typical)

**Polymarket:**
- Trading fee: ~2% on winnings
- Gas fees: $0.50-$5 (negligible for large bets)

**Kalshi:**
- Trading fee: ~3-7% depending on volume tier
- No gas fees (centralized)

**Calci (if available):**
- Trading fee: ~1-2%

## Calculation Example

**Scenario:**
- p_bayesian = 0.65 (our estimate)
- Polymarket price = 0.55 (market-implied probability)
- Transaction costs = 2% = 0.02
- Bankroll = $10,000

**Step 1: Calculate edge**
- edge = 0.65 - 0.55 = 0.10 = 10%

**Step 2: Calculate EV**
- EV_per_dollar = edge - costs = 0.10 - 0.02 = 0.08 = 8%

**Step 3: Calculate Kelly fraction**
- kelly = edge / (1 - p_market)
- kelly = 0.10 / (1 - 0.55) = 0.10 / 0.45 = 0.222 = 22.2%
- Capped at 5%: kelly_capped = 0.05

**Step 4: Calculate stake**
- suggested_stake = bankroll × kelly_capped
- suggested_stake = $10,000 × 0.05 = $500

**Step 5: Expected profit**
- expected_profit = stake × EV_per_dollar
- expected_profit = $500 × 0.08 = $40

**Output:**
{
  "market_id": "uuid",
  "provider": "polymarket",
  "price": 0.55,
  "implied_probability": 0.55,
  "edge": 0.10,
  "transaction_costs": 0.02,
  "slippage_estimate": 0.01,
  "expected_value_per_dollar": 0.08,
  "kelly_fraction": 0.05,
  "suggested_stake": 500.0,
  "trade_rationale": "Strong positive edge (10%) after costs. Our Bayesian estimate (65%) significantly exceeds market price (55%). Kelly suggests 5% allocation ($500). Expected profit: $40."
}

## Important Guidelines

- **Be conservative**: Better to underestimate EV than overestimate
- **Account for ALL costs**: Fees, gas, slippage, spreads
- **Check liquidity**: Can't trade if there's no volume
- **Explain clearly**: Why is there an edge? What's the basis?
- **Highlight risks**: What could make p_bayesian wrong?
- **Always disclaim**: End with "NOT FINANCIAL ADVICE"
- **Filter by threshold**: Only show opportunities above min_edge (default 2%)

Remember: An arbitrage opportunity is NOT a guarantee of profit.
Our p_bayesian is an estimate with uncertainty.
Markets may have information we don't.
Position sizing (Kelly) is critical for risk management.

ALWAYS END EVERY OUTPUT WITH: "NOT FINANCIAL ADVICE. This is research only."
"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return list of arbitrage opportunities"""
        return ArbitrageOpportunityList

    async def detect_cross_platform_arbitrage(
        self,
        market_slug: str,
        market_question: str,
        providers: List[str] = ["polymarket", "kalshi"],
        bankroll: float = None,
        max_kelly: float = None,
        transaction_costs: Optional[Dict[str, float]] = None
    ) -> List[ArbitrageOpportunity]:
        """
        Detect TRUE cross-platform arbitrage (opposite sides, guaranteed profit)

        This finds opportunities where you can bet OPPOSITE outcomes on different
        platforms and guarantee profit regardless of the outcome.

        Example: If Polymarket has YES at 55% and Kalshi has NO at 50%:
        - Buy YES on Polymarket: cost 0.55
        - Buy NO on Kalshi: cost 0.50
        - Total cost: 1.05 (plus fees)
        - If total cost < 1.0 after fees, this is guaranteed profit!

        Args:
            market_slug: Market identifier
            market_question: Question text
            providers: Platforms to check
            bankroll: Available capital
            max_kelly: Max position size
            transaction_costs: Fee structure

        Returns:
            List of cross-platform arbitrage opportunities
        """
        self.logger.info(f"Detecting cross-platform arbitrage across {providers}")

        # Apply config defaults
        bankroll = bankroll or settings.DEFAULT_BANKROLL
        max_kelly = max_kelly or settings.MAX_KELLY_FRACTION

        # Default transaction costs
        if transaction_costs is None:
            transaction_costs = {
                "polymarket": 0.02,  # 2%
                "kalshi": 0.05,      # 5%
                "calci": 0.02        # 2%
            }

        # Fetch prices from all providers
        market_prices = await self._fetch_market_prices(market_slug, providers)

        opportunities = []

        # Check all platform pairs
        platform_list = list(market_prices.keys())

        for i in range(len(platform_list)):
            for j in range(i + 1, len(platform_list)):
                platform_a = platform_list[i]
                platform_b = platform_list[j]

                price_data_a = market_prices.get(platform_a)
                price_data_b = market_prices.get(platform_b)

                if not price_data_a or not price_data_b:
                    continue

                p_a = price_data_a['implied_probability']
                p_b = price_data_b['implied_probability']

                # Strategy 1: YES on A, NO on B
                cost_yes_a_no_b = p_a + (1.0 - p_b)
                fee_a = transaction_costs.get(platform_a, 0.03)
                fee_b = transaction_costs.get(platform_b, 0.03)
                total_cost_1 = cost_yes_a_no_b * (1 + fee_a + fee_b)

                # Strategy 2: NO on A, YES on B
                cost_no_a_yes_b = (1.0 - p_a) + p_b
                total_cost_2 = cost_no_a_yes_b * (1 + fee_a + fee_b)

                # Pick the cheaper strategy
                if total_cost_1 < total_cost_2:
                    best_cost = total_cost_1
                    side_a_outcome = "YES"
                    side_b_outcome = "NO"
                else:
                    best_cost = total_cost_2
                    side_a_outcome = "NO"
                    side_b_outcome = "YES"

                # Check if profitable
                if best_cost < 1.0:
                    guaranteed_profit = 1.0 - best_cost

                    # Calculate position sizing
                    # For guaranteed arbitrage, Kelly doesn't apply the same way
                    # Use conservative fraction of bankroll
                    suggested_stake = min(bankroll * max_kelly, bankroll * guaranteed_profit)

                    # Split stake across both platforms proportionally
                    if side_a_outcome == "YES":
                        stake_a = suggested_stake * (p_a / cost_yes_a_no_b) if cost_yes_a_no_b > 0 else suggested_stake / 2
                        stake_b = suggested_stake * ((1.0 - p_b) / cost_yes_a_no_b) if cost_yes_a_no_b > 0 else suggested_stake / 2
                    else:
                        stake_a = suggested_stake * ((1.0 - p_a) / cost_no_a_yes_b) if cost_no_a_yes_b > 0 else suggested_stake / 2
                        stake_b = suggested_stake * (p_b / cost_no_a_yes_b) if cost_no_a_yes_b > 0 else suggested_stake / 2

                    rationale = (
                        f"GUARANTEED PROFIT ARBITRAGE: Buy {side_a_outcome} on {platform_a} "
                        f"({p_a if side_a_outcome == 'YES' else 1.0-p_a:.1%}) and {side_b_outcome} on {platform_b} "
                        f"({p_b if side_b_outcome == 'YES' else 1.0-p_b:.1%}). "
                        f"Total cost: ${best_cost:.4f} for $1.00 payout. "
                        f"Guaranteed profit: ${guaranteed_profit:.4f} per $1 ({guaranteed_profit*100:.2f}%). "
                        f"Suggested total stake: ${suggested_stake:.2f}. "
                        f"Expected profit: ${suggested_stake * guaranteed_profit:.2f}. "
                        f"NOT FINANCIAL ADVICE. Risk-free arbitrage in theory, but requires execution on both platforms."
                    )

                    opportunities.append(ArbitrageOpportunity(
                        arbitrage_type="cross_platform",
                        platform_pair=[platform_a, platform_b],
                        side_a=PlatformSide(
                            platform=platform_a,
                            market_id=price_data_a.get('market_id', 'unknown'),
                            outcome=side_a_outcome,
                            price=p_a if side_a_outcome == "YES" else 1.0 - p_a,
                            stake=stake_a
                        ),
                        side_b=PlatformSide(
                            platform=platform_b,
                            market_id=price_data_b.get('market_id', 'unknown'),
                            outcome=side_b_outcome,
                            price=p_b if side_b_outcome == "YES" else 1.0 - p_b,
                            stake=stake_b
                        ),
                        total_cost=best_cost,
                        guaranteed_profit=guaranteed_profit,
                        transaction_costs=fee_a + fee_b,
                        slippage_estimate=0.01,
                        expected_value_per_dollar=guaranteed_profit,
                        kelly_fraction=max_kelly,
                        suggested_stake=suggested_stake,
                        trade_rationale=rationale
                    ))

        # Sort by guaranteed profit
        opportunities.sort(key=lambda x: x.guaranteed_profit or 0, reverse=True)

        self.logger.info(f"Found {len(opportunities)} cross-platform arbitrage opportunities")

        return opportunities

    async def detect_arbitrage(
        self,
        p_bayesian: float,
        market_slug: str,
        market_question: str,
        providers: List[str] = ["polymarket", "kalshi"],
        bankroll: float = None,
        max_kelly: float = None,
        min_edge_threshold: float = None,
        transaction_costs: Optional[Dict[str, float]] = None,
        include_cross_platform: bool = True
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across platforms

        Args:
            p_bayesian: Our Bayesian probability estimate
            market_slug: Market identifier
            market_question: Question text
            providers: List of platforms to check
            bankroll: Available capital
            max_kelly: Maximum Kelly fraction (default 5%)
            min_edge_threshold: Minimum edge to report (default 2%)
            transaction_costs: Dict of provider -> cost (default uses typical fees)
            include_cross_platform: Whether to include cross-platform arbitrage

        Returns:
            List of ArbitrageOpportunity objects (both mispricing and cross-platform)
        """
        self.logger.info(
            f"Detecting arbitrage for p_bayesian={p_bayesian:.2%} across {providers}"
        )

        # Apply config defaults
        bankroll = bankroll or settings.DEFAULT_BANKROLL
        max_kelly = max_kelly or settings.MAX_KELLY_FRACTION
        min_edge_threshold = min_edge_threshold or settings.MIN_EDGE_THRESHOLD

        # Default transaction costs
        if transaction_costs is None:
            transaction_costs = {
                "polymarket": 0.02,  # 2%
                "kalshi": 0.05,      # 5%
                "calci": 0.02        # 2%
            }

        # Fetch prices from all providers
        market_prices = await self._fetch_market_prices(market_slug, providers)

        # Calculate mispricing opportunities (single-sided based on Bayesian estimate)
        opportunities = []

        for provider, price_data in market_prices.items():
            if price_data is None:
                continue

            p_market = price_data['implied_probability']
            edge = p_bayesian - p_market
            costs = transaction_costs.get(provider, 0.03)

            # Calculate EV and Kelly
            stake_info = self.kelly_calc.calculate_stake(
                bankroll=bankroll,
                p_true=p_bayesian,
                p_market=p_market,
                transaction_costs=costs,
                slippage=0.01,  # Assume 1% slippage
                max_kelly=max_kelly
            )

            # Filter by minimum edge
            if abs(edge) >= min_edge_threshold and stake_info['expected_value_per_dollar'] > 0:
                direction = "YES" if edge > 0 else "NO"
                rationale = self._generate_rationale(
                    p_bayesian, p_market, edge, stake_info, direction
                )

                opportunities.append(ArbitrageOpportunity(
                    arbitrage_type="mispricing",
                    market_id=price_data.get('market_id', 'unknown'),
                    provider=provider,
                    price=p_market,
                    implied_probability=p_market,
                    edge=edge,
                    transaction_costs=costs,
                    slippage_estimate=0.01,
                    expected_value_per_dollar=stake_info['expected_value_per_dollar'],
                    kelly_fraction=stake_info['kelly_fraction'],
                    suggested_stake=stake_info['suggested_stake'],
                    trade_rationale=rationale
                ))

        # Also check for cross-platform arbitrage if requested
        if include_cross_platform:
            cross_platform_opps = await self.detect_cross_platform_arbitrage(
                market_slug=market_slug,
                market_question=market_question,
                providers=providers,
                bankroll=bankroll,
                max_kelly=max_kelly,
                transaction_costs=transaction_costs
            )
            opportunities.extend(cross_platform_opps)

        # Sort by EV (mispricing) and guaranteed profit (cross-platform)
        opportunities.sort(
            key=lambda x: x.guaranteed_profit if x.arbitrage_type == "cross_platform" else x.expected_value_per_dollar,
            reverse=True
        )

        self.logger.info(
            f"Found {len(opportunities)} total arbitrage opportunities "
            f"({sum(1 for o in opportunities if o.arbitrage_type == 'mispricing')} mispricing, "
            f"{sum(1 for o in opportunities if o.arbitrage_type == 'cross_platform')} cross-platform)"
        )

        return opportunities

    async def _fetch_market_prices(
        self,
        market_slug: str,
        providers: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch current prices from all providers

        Args:
            market_slug: Market identifier
            providers: List of platforms

        Returns:
            Dict of provider -> price data
        """
        prices = {}

        for provider in providers:
            try:
                if provider == "polymarket":
                    market = await self.polymarket.get_market_with_price(market_slug)
                    if market and 'prices' in market:
                        prices[provider] = {
                            'market_id': market.get('id', 'unknown'),
                            'implied_probability': market['prices']['implied_probability'],
                            'liquidity': market.get('liquidity', 0)
                        }

                elif provider == "kalshi":
                    price = await self.kalshi.get_market_price(market_slug)
                    if price:
                        prices[provider] = {
                            'market_id': market_slug,
                            'implied_probability': price,
                            'liquidity': None
                        }

                else:
                    self.logger.warning(f"Unknown provider: {provider}")

            except Exception as e:
                self.logger.error(f"Failed to fetch price from {provider}: {e}")
                prices[provider] = None

        return prices

    def _generate_rationale(
        self,
        p_bayesian: float,
        p_market: float,
        edge: float,
        stake_info: Dict[str, float],
        direction: str
    ) -> str:
        """Generate trade rationale string"""
        ev = stake_info['expected_value_per_dollar']
        kelly = stake_info['kelly_fraction']
        stake = stake_info['suggested_stake']
        expected_profit = stake * ev

        rationale = (
            f"Our Bayesian estimate ({p_bayesian:.1%}) "
            f"{'exceeds' if edge > 0 else 'is below'} market price ({p_market:.1%}), "
            f"creating {abs(edge):.1%} edge to bet {direction}. "
            f"After transaction costs, EV = {ev:.1%} per dollar. "
            f"Kelly criterion suggests {kelly:.1%} allocation (${stake:.0f}). "
            f"Expected profit: ${expected_profit:.2f}. "
            f"NOT FINANCIAL ADVICE. This is research only."
        )

        return rationale

    async def scan_markets_for_arbitrage(
        self,
        markets: List[Dict[str, Any]],
        providers: List[str] = ["polymarket", "kalshi"],
        min_profit_threshold: float = 0.005,
        bankroll: float = None,
        max_kelly: float = None,
        parallel_limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Scan multiple markets for cross-platform arbitrage in parallel

        This is the fast arbitrage scanner that doesn't require Bayesian analysis.
        It only checks for guaranteed profit opportunities from price differences.

        Args:
            markets: List of market dicts (with slug/id and question)
            providers: Platforms to check
            min_profit_threshold: Minimum profit to report (0.005 = 0.5%)
            bankroll: Available capital
            max_kelly: Max position size
            parallel_limit: Max concurrent requests

        Returns:
            List of dicts with {market, opportunities}
        """
        bankroll = bankroll or settings.DEFAULT_BANKROLL
        max_kelly = max_kelly or settings.MAX_KELLY_FRACTION

        self.logger.info(f"Scanning {len(markets)} markets with parallelism={parallel_limit}")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(parallel_limit)

        async def check_market(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Check single market with rate limiting"""
            async with semaphore:
                try:
                    market_slug = market.get('slug', market.get('ticker', market.get('id')))
                    question = market.get('question', market.get('title', ''))

                    # Check for cross-platform arbitrage
                    opportunities = await self.detect_cross_platform_arbitrage(
                        market_slug=market_slug,
                        market_question=question,
                        providers=providers,
                        bankroll=bankroll,
                        max_kelly=max_kelly
                    )

                    # Filter by minimum profit
                    profitable = [
                        opp for opp in opportunities
                        if (opp.guaranteed_profit or 0) >= min_profit_threshold
                    ]

                    if profitable:
                        return {
                            'market': market,
                            'opportunities': profitable
                        }

                except Exception as e:
                    self.logger.warning(f"Error checking market {market.get('slug', '?')}: {e}")
                    return None

        # Process all markets in parallel (with concurrency limit)
        results = await asyncio.gather(*[check_market(m) for m in markets])

        # Filter out None results
        found_opportunities = [r for r in results if r is not None]

        self.logger.info(
            f"Scan complete: Found arbitrage in {len(found_opportunities)}/{len(markets)} markets"
        )

        return found_opportunities
