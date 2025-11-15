"""
Trader Analytics Client

High-level trader analytics module for:
- Computing radar chart metrics (5 dimensions)
- Computing tag-specific credibility scores
- Computing per-market position details
- Computing time-windowed PnL (30d, 90d)
- Computing trade correlation with smart money
"""

import asyncio
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TraderAnalytics:
    """
    High-level trader analytics computations.

    Provides methods to compute:
    - Radar chart metrics (5 dimensions)
    - Tag credibility scores
    - Market position details
    - PnL windows (30d, 90d)
    - Trade correlation
    """

    def __init__(
        self,
        min_volume: float = 10000.0,
        min_markets: int = 20,
        min_win_rate: float = 0.60,
    ):
        """
        Initialize TraderAnalytics.

        Args:
            min_volume: Minimum volume for eligibility ($10k default)
            min_markets: Minimum markets for eligibility (20 default)
            min_win_rate: Minimum win rate for eligibility (60% default)
        """
        self.min_volume = min_volume
        self.min_markets = min_markets
        self.min_win_rate = min_win_rate

    def compute_radar_metrics(
        self,
        proxy_wallet: str,
        wallet_stats: Dict[str, Any],
        all_trades: List[Dict[str, Any]],
        all_wallet_stats: Optional[List[Dict[str, Any]]] = None,
        smart_money_positions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all 5 radar chart metrics for a wallet.

        Dimensions:
        1. Unique Markets - Diversity of markets traded
        2. Trade Correlation - Alignment with smart money (Tier A wallets)
        3. Entry Timing - Early vs late entry percentile
        4. Position Size - Relative position sizing
        5. WC/TX Delta - Profitability ratio (PnL/volume)

        Args:
            proxy_wallet: Wallet address
            wallet_stats: Global wallet statistics
            all_trades: All trades for this wallet
            all_wallet_stats: Stats for all wallets (for percentile calculations)
            smart_money_positions: Smart money aggregate positions by condition_id

        Returns:
            Dict with radar metrics (scores + raw values)
        """
        # 1. Unique Markets Score
        unique_markets_count = wallet_stats.get("n_markets", 0)
        unique_markets_score = min(1.0, unique_markets_count / 50.0)

        # 2. Trade Correlation Score
        trade_correlation_score = 0.0
        avg_correlation = 0.0
        if smart_money_positions and all_trades:
            correlation = self._compute_trade_correlation(
                all_trades, smart_money_positions
            )
            avg_correlation = correlation
            # Map correlation from [-1, 1] to [0, 1]
            trade_correlation_score = (correlation + 1) / 2.0

        # 3. Entry Timing Score
        entry_timing_score = 0.0
        avg_entry_rank = 0.0
        if all_trades:
            avg_entry_rank = self._compute_entry_timing_score(all_trades)
            entry_timing_score = avg_entry_rank

        # 4. Position Size Score
        position_size_score = 0.0
        avg_size_rank = 0.0
        if all_wallet_stats and wallet_stats.get("avg_position_size"):
            avg_position_size = wallet_stats["avg_position_size"]
            all_sizes = [
                w.get("avg_position_size", 0.0)
                for w in all_wallet_stats
                if w.get("avg_position_size", 0.0) > 0
            ]
            if all_sizes:
                median_size = statistics.median(all_sizes)
                if median_size > 0:
                    # Score caps at 1.0 if 2x median or higher
                    position_size_score = min(1.0, avg_position_size / (2 * median_size))
                    avg_size_rank = self._percentile_rank(avg_position_size, all_sizes)

        # 5. WC/TX Delta Score (Wallet Change / Transaction Volume)
        wallet_tx_delta_score = 0.0
        wallet_tx_delta_ratio = 0.0
        total_volume = wallet_stats.get("total_volume", 0.0)
        realized_pnl = wallet_stats.get("realized_pnl", 0.0)
        if total_volume > 0:
            wallet_tx_delta_ratio = realized_pnl / total_volume
            # Map ratio from [-1, 1] to [0, 1]
            # -100% PnL = 0, +100% PnL = 1.0
            wallet_tx_delta_score = max(0.0, min(1.0, (wallet_tx_delta_ratio + 1) / 2.0))

        return {
            "proxy_wallet": proxy_wallet,
            # Normalized scores (0-1)
            "unique_markets_score": unique_markets_score,
            "trade_correlation_score": trade_correlation_score,
            "entry_timing_score": entry_timing_score,
            "position_size_score": position_size_score,
            "wallet_tx_delta_score": wallet_tx_delta_score,
            # Raw values
            "unique_markets_count": unique_markets_count,
            "avg_correlation": avg_correlation,
            "avg_entry_rank": avg_entry_rank,
            "avg_size_rank": avg_size_rank,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _compute_trade_correlation(
        self,
        trades: List[Dict[str, Any]],
        smart_money_positions: Dict[str, float],
    ) -> float:
        """
        Compute correlation between trader positions and smart money positions.

        Args:
            trades: Trader's trades
            smart_money_positions: {condition_id: net_position} for smart money

        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        # Build trader's net positions by condition_id
        trader_positions = defaultdict(float)
        for trade in trades:
            condition_id = trade.get("condition_id")
            side = trade.get("side", "").upper()
            size = trade.get("size", 0.0)

            if condition_id:
                if side == "BUY":
                    trader_positions[condition_id] += size
                elif side == "SELL":
                    trader_positions[condition_id] -= size

        # Find common markets
        common_markets = set(trader_positions.keys()) & set(smart_money_positions.keys())

        if len(common_markets) < 3:
            # Need at least 3 points for meaningful correlation
            return 0.0

        trader_vals = [trader_positions[m] for m in common_markets]
        smart_vals = [smart_money_positions[m] for m in common_markets]

        # Compute Pearson correlation
        try:
            correlation = np.corrcoef(trader_vals, smart_vals)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return float(correlation)
        except Exception as e:
            logger.warning(f"Error computing correlation: {e}")
            return 0.0

    def _compute_entry_timing_score(
        self,
        trades: List[Dict[str, Any]],
    ) -> float:
        """
        Compute average entry timing percentile.

        For each market, calculate how early the trader entered
        compared to all traders. Earlier = higher score.

        Args:
            trades: Trader's trades

        Returns:
            Average entry timing percentile (0-1)
        """
        # Group trades by condition_id
        market_first_entry = {}
        for trade in trades:
            condition_id = trade.get("condition_id")
            timestamp = trade.get("timestamp", 0)
            if condition_id and timestamp:
                if condition_id not in market_first_entry:
                    market_first_entry[condition_id] = timestamp
                else:
                    market_first_entry[condition_id] = min(
                        market_first_entry[condition_id], timestamp
                    )

        if not market_first_entry:
            return 0.0

        # For this implementation, we assume:
        # - Early entry (first quartile) = 0.75-1.0
        # - Late entry (last quartile) = 0.0-0.25
        # Since we don't have all traders' data here, we'll use a simplified heuristic:
        # Entry timing relative to market creation/start

        # Placeholder: Return 0.5 as default
        # In production, you'd query all traders' first entries per market
        # and compute actual percentile rank
        return 0.5

    def _percentile_rank(self, value: float, all_values: List[float]) -> float:
        """
        Calculate percentile rank of value in all_values.

        Args:
            value: The value to rank
            all_values: List of all values

        Returns:
            Percentile rank (0-1)
        """
        if not all_values:
            return 0.0

        sorted_values = sorted(all_values)
        rank = sum(1 for v in sorted_values if v < value)
        percentile = rank / len(sorted_values)
        return percentile

    def compute_tag_credibility(
        self,
        proxy_wallet: str,
        tag: str,
        tag_stats: Dict[str, Any],
        all_wallets_tag_stats: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute credibility score for trader in specific tag.

        Credibility formula:
        0.5 × ROI_percentile + 0.3 × win_rate_percentile + 0.2 × volume_percentile

        Args:
            proxy_wallet: Wallet address
            tag: Tag/category
            tag_stats: Wallet's tag-specific stats
            all_wallets_tag_stats: All wallets' stats for this tag

        Returns:
            Tag credibility dict with score, percentiles, rank
        """
        tag_roi = tag_stats.get("roi", 0.0)
        tag_win_rate = tag_stats.get("win_rate", 0.0)
        tag_volume = tag_stats.get("total_volume", 0.0)
        tag_positions = tag_stats.get("n_positions", 0)

        # Extract all ROIs, win rates, volumes for this tag
        all_rois = [w.get("roi", 0.0) for w in all_wallets_tag_stats]
        all_win_rates = [w.get("win_rate", 0.0) for w in all_wallets_tag_stats]
        all_volumes = [w.get("total_volume", 0.0) for w in all_wallets_tag_stats]

        # Compute percentile ranks
        roi_percentile = self._percentile_rank(tag_roi, all_rois)
        win_rate_percentile = self._percentile_rank(tag_win_rate, all_win_rates)
        volume_percentile = self._percentile_rank(tag_volume, all_volumes)

        # Compute composite credibility score
        credibility_score = (
            0.5 * roi_percentile + 0.3 * win_rate_percentile + 0.2 * volume_percentile
        )

        # Compute rank (1 = best)
        # Sort all wallets by credibility score descending
        all_scores = []
        for w in all_wallets_tag_stats:
            w_roi = w.get("roi", 0.0)
            w_wr = w.get("win_rate", 0.0)
            w_vol = w.get("total_volume", 0.0)
            w_roi_pct = self._percentile_rank(w_roi, all_rois)
            w_wr_pct = self._percentile_rank(w_wr, all_win_rates)
            w_vol_pct = self._percentile_rank(w_vol, all_volumes)
            w_score = 0.5 * w_roi_pct + 0.3 * w_wr_pct + 0.2 * w_vol_pct
            all_scores.append(w_score)

        all_scores_sorted = sorted(all_scores, reverse=True)
        tag_rank = all_scores_sorted.index(credibility_score) + 1 if credibility_score in all_scores_sorted else None

        return {
            "id": f"{proxy_wallet}_{tag}",
            "proxy_wallet": proxy_wallet,
            "tag": tag,
            "credibility_score": credibility_score,
            "tag_roi": tag_roi,
            "tag_win_rate": tag_win_rate,
            "tag_volume": tag_volume,
            "tag_positions": tag_positions,
            "tag_rank": tag_rank,
            "roi_percentile": roi_percentile,
            "win_rate_percentile": win_rate_percentile,
            "volume_percentile": volume_percentile,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def compute_market_position_details(
        self,
        proxy_wallet: str,
        condition_id: str,
        trades: List[Dict[str, Any]],
        open_position: Optional[Dict[str, Any]] = None,
        closed_position: Optional[Dict[str, Any]] = None,
        market_volume: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute detailed position info for wallet in specific market.

        Args:
            proxy_wallet: Wallet address
            condition_id: Market condition ID
            trades: All trades for this wallet+market
            open_position: Current open position (if any)
            closed_position: Closed position (if any)
            market_volume: Total market volume

        Returns:
            Detailed market position dict
        """
        # Filter trades for this market
        market_trades = [
            t for t in trades if t.get("condition_id") == condition_id
        ]

        if not market_trades:
            return {}

        # Calculate buy/sell volumes and prices
        buy_volume = 0.0
        sell_volume = 0.0
        buy_prices = []
        sell_prices = []
        init_price = None
        first_trade_ts = None
        last_trade_ts = None

        for trade in sorted(market_trades, key=lambda t: t.get("timestamp", 0)):
            side = trade.get("side", "").upper()
            size = trade.get("size", 0.0)
            price = trade.get("price", 0.0)
            timestamp = trade.get("timestamp", 0)
            notional = size * price

            if first_trade_ts is None:
                first_trade_ts = timestamp
                init_price = price
            last_trade_ts = timestamp

            if side == "BUY":
                buy_volume += notional
                buy_prices.append(price)
            elif side == "SELL":
                sell_volume += notional
                sell_prices.append(price)

        avg_buy_price = statistics.mean(buy_prices) if buy_prices else 0.0
        avg_sell_price = statistics.mean(sell_prices) if sell_prices else 0.0
        net_volume = buy_volume - sell_volume
        market_trades_count = len(market_trades)

        # Current position details
        current_position_size = 0.0
        current_price = 0.0
        unrealized_pnl = 0.0
        position_value = 0.0

        if open_position:
            current_position_size = open_position.get("size", 0.0)
            current_price = open_position.get("current_price", 0.0)
            unrealized_pnl = open_position.get("unrealized_pnl", 0.0)
            position_value = open_position.get("position_value", 0.0)

        # Realized PnL
        realized_pnl = 0.0
        if closed_position:
            realized_pnl = closed_position.get("realized_pnl", 0.0)

        total_pnl = realized_pnl + unrealized_pnl

        # Market concentration
        volume_share = 0.0
        if market_volume and market_volume > 0:
            wallet_volume = buy_volume + sell_volume
            volume_share = wallet_volume / market_volume

        # Market metadata
        market_title = None
        market_slug = None
        event_slug = None
        outcome = None
        if market_trades:
            first_trade = market_trades[0]
            market_title = first_trade.get("title")
            market_slug = first_trade.get("slug")
            event_slug = first_trade.get("event_slug")
            outcome = first_trade.get("outcome")

        return {
            "proxy_wallet": proxy_wallet,
            "condition_id": condition_id,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_volume": net_volume,
            "avg_buy_price": avg_buy_price,
            "avg_sell_price": avg_sell_price,
            "init_price": init_price or 0.0,
            "current_price": current_price,
            "current_position_size": current_position_size,
            "position_value": position_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "market_trades_count": market_trades_count,
            "first_trade_ts": first_trade_ts,
            "last_trade_ts": last_trade_ts,
            "market_volume": market_volume or 0.0,
            "volume_share": volume_share,
            "market_title": market_title,
            "market_slug": market_slug,
            "event_slug": event_slug,
            "outcome": outcome,
        }

    def compute_pnl_windows(
        self,
        closed_positions: List[Dict[str, Any]],
        window_30d: bool = True,
        window_90d: bool = True,
    ) -> Dict[str, float]:
        """
        Compute time-windowed PnL (30d, 90d, all-time).

        Args:
            closed_positions: All closed positions for wallet
            window_30d: Calculate 30-day PnL
            window_90d: Calculate 90-day PnL

        Returns:
            Dict with pnl_30d, pnl_90d, pnl_all_time
        """
        now = datetime.now(timezone.utc)
        pnl_30d = 0.0
        pnl_90d = 0.0
        pnl_all_time = 0.0

        for position in closed_positions:
            realized_pnl = position.get("realized_pnl", 0.0)
            timestamp = position.get("timestamp", 0)

            # Add to all-time
            pnl_all_time += realized_pnl

            if timestamp:
                position_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                days_ago = (now - position_date).days

                # Add to 30d if within last 30 days
                if window_30d and days_ago <= 30:
                    pnl_30d += realized_pnl

                # Add to 90d if within last 90 days
                if window_90d and days_ago <= 90:
                    pnl_90d += realized_pnl

        return {
            "pnl_30d": pnl_30d,
            "pnl_90d": pnl_90d,
            "pnl_all_time": pnl_all_time,
        }


    def build_smart_money_positions(
        self,
        tier_a_wallets: List[str],
        all_trades: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Build aggregate smart money positions by condition_id.

        Args:
            tier_a_wallets: List of Tier A wallet addresses
            all_trades: All trades for all wallets

        Returns:
            {condition_id: net_position} for smart money
        """
        smart_money_positions = defaultdict(float)

        for trade in all_trades:
            proxy_wallet = trade.get("proxy_wallet")
            condition_id = trade.get("condition_id")
            side = trade.get("side", "").upper()
            size = trade.get("size", 0.0)

            if proxy_wallet in tier_a_wallets and condition_id:
                if side == "BUY":
                    smart_money_positions[condition_id] += size
                elif side == "SELL":
                    smart_money_positions[condition_id] -= size

        return dict(smart_money_positions)

    def is_eligible(self, wallet_stats: Dict[str, Any]) -> bool:
        """
        Check if wallet meets eligibility thresholds.

        Args:
            wallet_stats: Wallet statistics

        Returns:
            True if eligible, False otherwise
        """
        total_volume = wallet_stats.get("total_volume", 0.0)
        n_positions = wallet_stats.get("n_positions", 0)
        win_rate = wallet_stats.get("win_rate", 0.0)

        return (
            total_volume >= self.min_volume
            and n_positions >= self.min_markets
            and win_rate >= self.min_win_rate
        )
