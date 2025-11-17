# ========================================================================
# CORE PREDICTION MARKET CLIENTS
# ========================================================================

# Primary platform clients
from .polymarket import PolymarketClient
from .kalshi import KalshiClient

# Research and analytics clients
try:
    from .valyu import ValyuResearchClient
except ImportError:
    ValyuResearchClient = None

from .trader_analytics import TraderAnalytics

# Wallet tracking client
from .wallet_tracker import WalletTracker

# ========================================================================
# WALLET INTELLIGENCE FUNCTIONS
# ========================================================================

# Wallet discovery and performance orchestration
from .functions.wallets import (
    PolymarketWalletCollector,
    enrich_unenriched_events,
)

# ========================================================================
# TRADER ANALYSIS FUNCTIONS
# ========================================================================

# Trader analysis and copy-trading intelligence
from .functions.traders import (
    PolymarketTraderAnalyzer,
    filter_best_traders,
    get_best_traders_positions,
    get_copy_trade_suggestions,
    analyze_smart_money_consensus,
    execute_full_trader_analysis,
)

# ========================================================================
# DATA UTILITIES
# ========================================================================

from .functions.data import (
    PolymarketDataCollector,
    sync_polymarket_data,
)

# ========================================================================
# PUBLIC API
# ========================================================================

__all__ = [
    # Core clients
    "PolymarketClient",
    "KalshiClient",
    "ValyuResearchClient",
    "TraderAnalytics",
    "WalletTracker",

    # Data collection
    "PolymarketDataCollector",
    "sync_polymarket_data",

    # Wallet intelligence
    "PolymarketWalletCollector",
    "enrich_unenriched_events",

    # Trader analysis
    "PolymarketTraderAnalyzer",
    "filter_best_traders",
    "get_best_traders_positions",
    "get_copy_trade_suggestions",
    "analyze_smart_money_consensus",
    "execute_full_trader_analysis",
]
