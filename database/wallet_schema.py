"""
Database schema for wallet tracking and copy-trading analysis.

Defines dataclasses for:
- Wallet profiles
- Trade history
- Closed positions (realized PnL)
- Wallet statistics (global & per-tag)
- Market concentration metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class Wallet:
    """Wallet profile and metadata."""

    proxy_wallet: str  # Primary key: 0x-prefixed wallet address
    first_seen_at: Optional[str] = None  # ISO timestamp of first trade
    last_seen_at: Optional[str] = None  # ISO timestamp of most recent trade
    last_sync_at: Optional[str] = None  # When we last synced this wallet's data

    # User profile metadata (from Polymarket API)
    name: Optional[str] = None
    pseudonym: Optional[str] = None
    bio: Optional[str] = None
    profile_image: Optional[str] = None
    profile_image_optimized: Optional[str] = None
    display_username_public: Optional[bool] = None

    # Quick stats (denormalized for performance)
    total_trades: int = 0
    total_markets: int = 0
    total_volume: float = 0.0

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class Trade:
    """Individual trade record from Polymarket Data API."""

    id: str  # Primary key: transaction_hash + asset + timestamp hash
    proxy_wallet: str  # Foreign key to wallets
    event_id: Optional[str] = None  # Foreign key to events (extracted from slug)
    condition_id: str = None  # Market condition ID

    # Trade details
    side: str = None  # BUY or SELL
    asset: str = None  # ERC-1155 token ID
    size: float = 0.0  # Token quantity
    price: float = 0.0  # Execution price (0-1)
    notional: float = 0.0  # size × price (USDC value)

    timestamp: int = 0  # Unix timestamp (seconds)
    transaction_hash: Optional[str] = None

    # Market metadata
    title: Optional[str] = None
    slug: Optional[str] = None
    event_slug: Optional[str] = None
    outcome: Optional[str] = None
    outcome_index: Optional[int] = None

    created_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WalletClosedPosition:
    """Closed/resolved position with realized PnL (from /closed-positions)."""

    id: str  # Primary key: proxy_wallet + condition_id + outcome_index
    proxy_wallet: str  # Foreign key to wallets
    event_id: Optional[str] = None  # Foreign key to events
    condition_id: str = None  # Market condition ID
    asset: str = None  # ERC-1155 token ID

    # Position details
    outcome: Optional[str] = None
    outcome_index: Optional[int] = None
    total_bought: float = 0.0  # Total tokens purchased
    avg_price: float = 0.0  # Average purchase price
    cur_price: float = 0.0  # Final settlement price

    # PnL
    realized_pnl: float = 0.0  # Profit/loss in USDC

    # Timestamps
    timestamp: int = 0  # Position close timestamp
    end_date: Optional[str] = None  # Market resolution date

    # Market metadata
    title: Optional[str] = None
    slug: Optional[str] = None
    event_slug: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WalletStats:
    """Aggregated wallet performance metrics (global)."""

    proxy_wallet: str  # Primary key

    # Volume metrics
    total_volume: float = 0.0  # Σ total_bought from closed positions
    avg_position_size: float = 0.0  # total_volume / n_positions

    # PnL metrics
    realized_pnl: float = 0.0  # Σ realized_pnl
    roi: float = 0.0  # realized_pnl / total_volume

    # Win rate
    n_positions: int = 0  # Count of closed positions
    n_wins: int = 0  # Positions with realized_pnl > 0
    n_losses: int = 0  # Positions with realized_pnl < 0
    win_rate: float = 0.0  # n_wins / n_positions

    # Activity metrics
    n_markets: int = 0  # Distinct markets traded
    n_events: int = 0  # Distinct events traded
    first_trade_at: Optional[str] = None
    last_trade_at: Optional[str] = None

    # Eligibility flags
    is_eligible: bool = False  # Meets thresholds (≥$10k, ≥20 markets, ≥60% win rate)
    tier: Optional[str] = None  # A/B/C based on score

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WalletTagStats:
    """Wallet performance by category/tag."""

    id: str  # Primary key: proxy_wallet + tag
    proxy_wallet: str  # Foreign key to wallets
    tag: str  # Category or tag from events

    # Volume metrics
    total_volume: float = 0.0
    avg_position_size: float = 0.0

    # PnL metrics
    realized_pnl: float = 0.0
    roi: float = 0.0

    # Win rate
    n_positions: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0

    # Activity
    n_markets: int = 0
    n_events: int = 0

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WalletMarketStats:
    """Wallet participation in specific markets (volume share)."""

    id: str  # Primary key: proxy_wallet + condition_id
    proxy_wallet: str  # Foreign key to wallets
    event_id: Optional[str] = None  # Foreign key to events
    condition_id: str = None  # Market condition ID

    # Volume metrics
    wallet_volume: float = 0.0  # Total USDC volume by this wallet
    market_volume: float = 0.0  # Total market volume
    volume_share: float = 0.0  # wallet_volume / market_volume

    # Activity
    n_trades: int = 0
    first_trade_ts: Optional[int] = None
    last_trade_ts: Optional[int] = None

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class MarketConcentration:
    """Market-level concentration and smart money metrics."""

    id: str  # Primary key: event_id + condition_id
    event_id: Optional[str] = None  # Foreign key to events
    condition_id: str = None  # Market condition ID
    platform: str = "polymarket"

    # Concentration metrics
    n_wallets: int = 0  # Total unique wallets
    total_volume: float = 0.0  # Total market volume

    herfindahl_index: float = 0.0  # Σ(volume_share²) - concentration measure
    top_1_share: float = 0.0  # Largest wallet's volume share
    top_5_share: float = 0.0  # Top 5 wallets' combined share
    top_10_share: float = 0.0  # Top 10 wallets' combined share

    # Smart money metrics
    smart_volume: float = 0.0  # Volume from Tier A wallets
    smart_volume_share: float = 0.0  # smart_volume / total_volume
    smart_wallet_count: int = 0  # Number of Tier A wallets

    dumb_volume: float = 0.0  # Volume from low-tier wallets
    dumb_volume_share: float = 0.0

    # Top wallets (for quick reference)
    top_wallets: Optional[List[str]] = field(default_factory=list)  # Top 10 wallet addresses

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WalletScore:
    """Composite score for copy-trading eligibility."""

    proxy_wallet: str  # Primary key

    # Component scores (normalized 0-1)
    roi_score: float = 0.0
    win_rate_score: float = 0.0
    volume_score: float = 0.0
    recency_score: float = 0.0

    # Tag-specific (for current event context)
    tag: Optional[str] = None
    roi_tag_score: float = 0.0

    # Composite
    composite_score: float = 0.0  # Weighted combination

    # Eligibility
    meets_thresholds: bool = False  # Hard filters passed
    tier: Optional[str] = None  # A/B/C classification

    # Context
    for_event_id: Optional[str] = None  # If scored in context of specific event

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


# Type aliases for convenience
WalletList = List[Wallet]
TradeList = List[Trade]
PositionList = List[WalletClosedPosition]
