// Query key factory for consistent and type-safe caching
// Follows React Query best practices: https://tkdodo.eu/blog/effective-react-query-keys

export const queryKeys = {
  // Market-related queries
  markets: {
    all: ['markets'] as const,
    lists: () => [...queryKeys.markets.all, 'list'] as const,
    list: (filters: any) => [...queryKeys.markets.lists(), filters] as const,
    stats: () => [...queryKeys.markets.all, 'stats'] as const,
    detail: (id: string) => [...queryKeys.markets.all, 'detail', id] as const,
  },

  // Elite trader queries
  elite: {
    all: ['elite'] as const,
    positions: (wallet?: string, category?: string) =>
      [...queryKeys.elite.all, 'positions', wallet, category] as const,
    openPositions: (wallet: string) =>
      [...queryKeys.elite.all, 'open-positions', wallet] as const,
    tradeHistory: (wallet: string, page: number) =>
      [...queryKeys.elite.all, 'trade-history', wallet, page] as const,
    traders: () => [...queryKeys.elite.all, 'traders'] as const,
  },

  // Wallet-related queries
  wallet: {
    all: ['wallet'] as const,
    trades: (wallet: string, page: number) =>
      [...queryKeys.wallet.all, 'trades', wallet, page] as const,
    detail: (wallet: string) =>
      [...queryKeys.wallet.all, 'detail', wallet] as const,
  },

  // Funds/Vaults queries
  funds: {
    all: ['funds'] as const,
    available: (risk?: string) =>
      [...queryKeys.funds.all, 'available', risk] as const,
  },

  // PnL queries
  pnl: ['pnl'] as const,

  // Events queries
  events: {
    all: ['events'] as const,
    list: (filters: any) => [...queryKeys.events.all, 'list', filters] as const,
    detail: (id: string) => [...queryKeys.events.all, 'detail', id] as const,
  },
} as const;

// Helper type to extract query key types
export type QueryKeys = typeof queryKeys;
