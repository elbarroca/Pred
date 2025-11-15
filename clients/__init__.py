from .polymarket import PolymarketClient

# Optional platform clients
try:
    from .kalshi import KalshiClient
except ImportError:
    KalshiClient = None

try:
    from .manifold import ManifoldClient
except ImportError:
    ManifoldClient = None

try:
    from .market_enricher import MarketEnricher
except ImportError:
    MarketEnricher = None

try:
    from .valyu import ValyuResearchClient
except ImportError:
    ValyuResearchClient = None

try:
    from .web3.wallet_tracker import WalletTrackerClient
except ImportError:
    WalletTrackerClient = None

try:
    from .trade.insider_detector import InsiderDetectorClient
except ImportError:
    InsiderDetectorClient = None

try:
    from .web3.alchemy import AlchemyWebhooksClient
except ImportError:
    AlchemyWebhooksClient = None

# Optional imports for webhook providers
try:
    from .web3.quicknode import QuickNodeWebhooksClient
except ImportError:
    QuickNodeWebhooksClient = None

try:
    from .web3.moralis import MoralisStreamsClient
except ImportError:
    MoralisStreamsClient = None

try:
    from .trade.trader_analytics import TraderAnalyticsClient
except ImportError:
    TraderAnalyticsClient = None

try:
    from .trade.trade_executor import TradeExecutor
except ImportError:
    TradeExecutor = None

__all__ = [
    "PolymarketClient",
    "KalshiClient",
    "ManifoldClient",
    "MarketEnricher",
    "ValyuResearchClient",
    "WalletTrackerClient",
    "InsiderDetectorClient",
    "AlchemyWebhooksClient",
    "QuickNodeWebhooksClient",
    "MoralisStreamsClient",
    "TraderAnalyticsClient",
    "TradeExecutor",
]
