import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/lib/supabase';
import { fetchMarketStats, fetchMarkets } from '@/lib/api/markets';
import { GlobalMarketStats, Market } from '@/types/market';
import { Fund } from '@/types/chat';
import { queryKeys } from '@/lib/api/queryKeys';

// --- 1. ELITE POSITIONS ---
export function useElitePositions(category?: string) {
    return useQuery({
        queryKey: queryKeys.elite.positions(undefined, category),
        queryFn: async () => {
            const { data: elites } = await supabase.from('elite_traders').select('proxy_wallet').eq('tier', 'S').limit(5);
            const eliteIds = elites?.map(e => e.proxy_wallet) || [];

            if (eliteIds.length === 0) return [];

            let query = supabase
                .from('elite_open_positions')
                .select('*')
                .in('proxy_wallet', eliteIds)
                .order('unrealized_pnl', { ascending: false })
                .limit(3);

            if (category) query = query.ilike('event_category', `%${category}%`);

            const { data: positions } = await query;

            if (!positions) return [];

            return positions.map(p => ({
                market: p.title,
                outcome: p.outcome,
                entry: (p.avg_entry_price || 0).toFixed(2),
                current: (p.current_price || 0).toFixed(2),
                whale: p.proxy_wallet.substring(0, 4),
                roi: ((p.unrealized_pnl || 0) / (p.size || 1) * 100).toFixed(1)
            }));
        },
        staleTime: 10 * 60 * 1000, // 10 minutes - aggressive caching
        gcTime: 30 * 60 * 1000, // Keep in cache 30 minutes
    });
}

// --- 2. FUNDS ---
export function useAvailableFunds(riskTolerance?: string) {
    return useQuery({
        queryKey: queryKeys.funds.available(riskTolerance),
        queryFn: async () => {
            // In a real app, this would fetch from an API/DB
            const allFunds: Fund[] = [
                { id: "v1", name: "Degen Crypto Index", description: "Leveraged bets on ETH/BTC.", risk: "High", roi_30d: 42.5, tvl: "2.1M" },
                { id: "v2", name: "Election Sniper", description: "Political prediction arbitrage.", risk: "Medium", roi_30d: 12.8, tvl: "850k" },
                { id: "v3", name: "Stable Arbitrage", description: "Low volatility stablecoin farming.", risk: "Low", roi_30d: 4.2, tvl: "5.4M" }
            ];

            if (riskTolerance) {
                return allFunds.filter(f => f.risk.toLowerCase() === riskTolerance.toLowerCase());
            }
            return allFunds;
        },
        staleTime: Infinity, // Never goes stale - static config data
        gcTime: Infinity, // Keep forever in cache
    });
}

// --- 3. MARKET STATS ---
export function useMarketStats() {
    return useQuery({
        queryKey: queryKeys.markets.stats(),
        queryFn: fetchMarketStats,
        staleTime: 15 * 60 * 1000, // 15 minutes - aggressive caching
        gcTime: 60 * 60 * 1000, // Keep in cache 1 hour
    });
}

// --- 4. MARKETS LIST ---
export function useMarkets(filters: {
    page?: number;
    search?: string;
    sortBy?: keyof Market;
    ascending?: boolean;
    filterExpiring?: boolean;
    categoryFilter?: string;
    minVolume?: number;
    minLiquidity?: number;
}) {
    return useQuery({
        queryKey: queryKeys.markets.list(filters),
        queryFn: () => fetchMarkets(
            filters.page,
            filters.search,
            filters.sortBy,
            filters.ascending,
            filters.filterExpiring,
            filters.categoryFilter,
            filters.minVolume,
            filters.minLiquidity
        ),
        placeholderData: (previousData) => previousData, // Keep previous data while fetching new page
        staleTime: 10 * 60 * 1000, // 10 minutes - aggressive caching
        gcTime: 30 * 60 * 1000, // Keep in cache 30 minutes
    });
}

// --- 5. PNL ---
export function usePnL() {
    return useQuery({
        queryKey: queryKeys.pnl,
        queryFn: async () => {
            // Mock data for now
            return {
                daily: "+15.4%",
                all_time: "-4.2%",
            };
        },
        staleTime: 20 * 60 * 1000, // 20 minutes - aggressive caching
        gcTime: 60 * 60 * 1000, // Keep in cache 1 hour
    });
}
