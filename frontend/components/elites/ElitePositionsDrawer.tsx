'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { X, ExternalLink, ArrowRight, Flame, History, Loader2, User } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInfiniteQuery } from '@tanstack/react-query';
import { EliteTrader } from '@/types/elite';
import { fetchElitePositions } from '@/lib/api/elites';
import { fetchWalletTrades } from '@/lib/api/wallets';
import { cn } from '@/lib/utils';
import { getWalletDisplayName, getWalletAvatar } from '@/lib/utils/wallet-display';

interface DrawerProps {
    trader: EliteTrader | null;
    highlightCategory?: string;
    onClose: () => void;
}

type Tab = 'open' | 'history';

export default function ElitePositionsDrawer({ trader, highlightCategory, onClose }: DrawerProps) {
    const [activeTab, setActiveTab] = useState<Tab>('open');
    const [openingLink, setOpeningLink] = useState<string | null>(null);
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    // Query 1: Open Positions (Infinite)
    const {
        data: positionsData,
        fetchNextPage: fetchNextOpenPage,
        hasNextPage: hasNextOpenPage,
        isFetchingNextPage: isFetchingNextOpenPage,
        isLoading: loadingOpen,
    } = useInfiniteQuery({
        queryKey: ['eliteOpenPositions', trader?.proxy_wallet],
        queryFn: ({ pageParam = 1 }) => fetchElitePositions(trader!.proxy_wallet, pageParam as number, 20),
        getNextPageParam: (lastPage, allPages) => {
            const currentPage = allPages.length;
            return currentPage < lastPage.totalPages ? currentPage + 1 : undefined;
        },
        enabled: !!trader && activeTab === 'open',
        initialPageParam: 1,
    });

    const positions = positionsData?.pages.flatMap(page => page.data) || [];
    const openTotal = positionsData?.pages[0]?.total || 0;

    // Query 2: Trade History (Infinite)
    const {
        data: historyData,
        fetchNextPage: fetchNextHistoryPage,
        hasNextPage: hasNextHistoryPage,
        isFetchingNextPage: isFetchingNextHistoryPage,
        isLoading: loadingHistory,
    } = useInfiniteQuery({
        queryKey: ['eliteTradeHistory', trader?.proxy_wallet],
        queryFn: ({ pageParam = 1 }) => fetchWalletTrades(trader!.proxy_wallet, pageParam as number, 20),
        getNextPageParam: (lastPage, allPages) => {
            const currentPage = allPages.length;
            return currentPage < lastPage.totalPages ? currentPage + 1 : undefined;
        },
        enabled: !!trader && activeTab === 'history',
        initialPageParam: 1,
    });

    const historyTrades = historyData?.pages.flatMap(page => page.data) || [];
    const historyTotal = historyData?.pages[0]?.total || 0;

    // Infinite scroll handler
    useEffect(() => {
        const scrollContainer = scrollContainerRef.current;
        if (!scrollContainer) return;

        const hasNext = activeTab === 'open' ? hasNextOpenPage : hasNextHistoryPage;
        const isFetching = activeTab === 'open' ? isFetchingNextOpenPage : isFetchingNextHistoryPage;
        const fetchNext = activeTab === 'open' ? fetchNextOpenPage : fetchNextHistoryPage;

        if (!hasNext || isFetching) return;

        const handleScroll = () => {
            const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
            // Load more when within 200px of bottom
            if (scrollHeight - scrollTop - clientHeight < 200) {
                fetchNext();
            }
        };

        scrollContainer.addEventListener('scroll', handleScroll);
        return () => scrollContainer.removeEventListener('scroll', handleScroll);
    }, [activeTab, hasNextOpenPage, hasNextHistoryPage, isFetchingNextOpenPage, isFetchingNextHistoryPage, fetchNextOpenPage, fetchNextHistoryPage]);

    const currency = (n: number | null) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n || 0);

    const cents = (n: number | null) => (n || 0).toFixed(2);

    // Count positions in the highlighted category
    const relevantCount = highlightCategory
        ? positions.filter(p => p.event_category?.toLowerCase().includes(highlightCategory.toLowerCase())).length
        : 0;

    const displayName = getWalletDisplayName(trader);
    const avatarUrl = getWalletAvatar(trader);

    return (
        <AnimatePresence>
            {trader && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[9998]"
                    />
                    <motion.div
                        initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
                        className="fixed right-0 top-0 h-full w-full md:w-[700px] bg-[#09090b] border-l border-white/10 z-[10000] shadow-2xl flex flex-col overflow-hidden"
                    >
                        {/* Header */}
                        <div className="p-4 md:p-6 border-b border-white/10 bg-[#0c0c0c]">
                            <div className="flex justify-between items-start mb-4 gap-2">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-start gap-3 mb-4">
                                        {/* Avatar */}
                                        <div className="relative flex-shrink-0">
                                            {avatarUrl ? (
                                                <Image
                                                    src={avatarUrl}
                                                    alt={displayName}
                                                    width={56}
                                                    height={56}
                                                    className="w-12 h-12 md:w-14 md:h-14 rounded-xl object-cover bg-zinc-800 border border-white/10"
                                                />
                                            ) : (
                                                <div className="w-12 h-12 md:w-14 md:h-14 rounded-xl bg-gradient-to-br from-zinc-800 to-zinc-900 border border-white/10 flex items-center justify-center">
                                                    <User className="w-6 h-6 md:w-7 md:h-7 text-zinc-500" />
                                                </div>
                                            )}
                                        </div>

                                        {/* Name & Badges */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex gap-2 mb-2 flex-wrap">
                                                <div className="relative">
                                                    <div className="absolute inset-0 bg-amber-500/20 blur-sm rounded-lg" />
                                                    <span className="relative px-3 py-1 rounded-lg bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 text-xs font-black border border-amber-500/40 whitespace-nowrap shadow-lg">
                                                        🏆 Rank #{trader.rank_in_tier}
                                                    </span>
                                                </div>
                                                <span className="px-3 py-1 rounded-lg bg-gradient-to-r from-zinc-800 to-zinc-900 text-zinc-300 text-xs font-bold border border-white/10 whitespace-nowrap shadow-md">
                                                    Tier {trader.tier}
                                                </span>
                                            </div>
                                            <h2 className="text-sm md:text-lg font-bold text-white mb-1">{displayName}</h2>
                                            <p className="text-xs font-mono text-zinc-500 break-all">{trader.proxy_wallet}</p>
                                        </div>
                                    </div>
                                </div>
                                <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors flex-shrink-0">
                                    <X className="w-5 h-5 text-zinc-500 hover:text-white" />
                                </button>
                            </div>

                            {/* Stats Grid */}
                            <div className="grid grid-cols-2 gap-3 md:gap-4">
                                <div className="bg-zinc-900 p-2.5 md:p-3 rounded-lg border border-white/5">
                                    <div className="text-xs text-zinc-500 mb-1">Total Volume</div>
                                    <div className="text-base md:text-lg font-mono font-bold text-white truncate">{currency(trader.total_volume)}</div>
                                </div>
                                <div className="bg-zinc-900 p-2.5 md:p-3 rounded-lg border border-white/5">
                                    <div className="text-xs text-zinc-500 mb-1">Win Rate</div>
                                    <div className="text-base md:text-lg font-mono font-bold text-emerald-400">{trader.win_rate?.toFixed(1)}%</div>
                                </div>
                                <div className="bg-zinc-900 p-2.5 md:p-3 rounded-lg border border-white/5">
                                    <div className="text-xs text-zinc-500 mb-1">ROI</div>
                                    <div className="text-base md:text-lg font-mono font-bold text-emerald-400">+{trader.roi?.toFixed(2)}%</div>
                                </div>
                                <div className="bg-zinc-900 p-2.5 md:p-3 rounded-lg border border-white/5">
                                    <div className="text-xs text-zinc-500 mb-1">Composite Score</div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-sm">⭐</span>
                                        <span className="text-base md:text-lg font-mono font-black text-cyan-300">{trader.composite_score?.toFixed(1)}</span>
                                    </div>
                                </div>
                            </div>

                            {/* NEW: Sector Signal Banner */}
                            {highlightCategory && relevantCount > 0 && (
                                <div className="mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center gap-3 animate-in slide-in-from-top-2">
                                    <div className="p-2 bg-amber-500/20 rounded-full text-amber-400">
                                        <Flame className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <div className="text-sm font-bold text-amber-100">Sector Match Detected</div>
                                        <div className="text-xs text-amber-500/80">
                                            This trader has <span className="font-bold text-white">{relevantCount}</span> open positions in {highlightCategory}.
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Tabs */}
                            <div className="flex gap-4 mt-6 border-b border-white/5">
                                <button
                                    onClick={() => setActiveTab('open')}
                                    className={cn(
                                        "pb-2 text-sm font-medium transition-colors relative",
                                        activeTab === 'open' ? "text-white" : "text-zinc-500 hover:text-zinc-300"
                                    )}
                                >
                                    Open Positions ({loadingOpen ? '...' : openTotal})
                                    {activeTab === 'open' && <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />}
                                </button>
                                <button
                                    onClick={() => setActiveTab('history')}
                                    className={cn(
                                        "pb-2 text-sm font-medium transition-colors relative",
                                        activeTab === 'history' ? "text-white" : "text-zinc-500 hover:text-zinc-300"
                                    )}
                                >
                                    Trade History
                                    {activeTab === 'history' && <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />}
                                </button>
                            </div>
                        </div>

                        {/* Content Section */}
                        <div className="flex-1 flex flex-col bg-[#050505] overflow-hidden max-h-full">
                            {activeTab === 'open' ? (
                                <>
                                    {/* Sticky Header with Pagination */}
                                    <div className="sticky top-0 z-10 bg-[#050505] border-b border-white/5 px-4 md:px-6 py-2 md:py-3">
                                        <div className="flex items-center justify-between gap-2">
                                            <h3 className="text-xs md:text-sm font-bold text-white flex items-center gap-2 uppercase tracking-wider">
                                                <span className="relative flex h-2.5 w-2.5">
                                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500"></span>
                                                </span>
                                                Live Open Positions
                                            </h3>
                                            <span className="text-xs text-zinc-600 whitespace-nowrap">
                                                {openTotal} records
                                            </span>
                                        </div>
                                    </div>

                                    {/* Scrollable Content */}
                                    <div ref={scrollContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 pt-0 min-h-0">

                                    {loadingOpen ? (
                                        <div className="space-y-3">{[1, 2, 3].map(i => <div key={i} className="h-24 bg-zinc-900 rounded-xl animate-pulse" />)}</div>
                                    ) : positions.length === 0 ? (
                                        <div className="text-center text-zinc-500 py-10">No open positions currently.</div>
                                    ) : (
                                        <div className="space-y-4">
                                            {positions.sort((a, b) => {
                                                const aMatch = highlightCategory && a.event_category?.toLowerCase().includes(highlightCategory.toLowerCase());
                                                const bMatch = highlightCategory && b.event_category?.toLowerCase().includes(highlightCategory.toLowerCase());
                                                return (aMatch === bMatch) ? 0 : aMatch ? -1 : 1;
                                            }).map(pos => {
                                                const isProfit = (pos.unrealized_pnl || 0) >= 0;
                                                const isMatch = highlightCategory && pos.event_category?.toLowerCase().includes(highlightCategory.toLowerCase());

                                                return (
                                                    <div
                                                        key={pos.id}
                                                        className={cn(
                                                            "p-3 md:p-4 rounded-xl border transition-all relative overflow-hidden",
                                                            isMatch
                                                                ? "bg-amber-950/10 border-amber-500/30 hover:bg-amber-900/20"
                                                                : "bg-zinc-900/20 border-white/5 hover:bg-zinc-900/40"
                                                        )}
                                                    >
                                                        {isMatch && <div className="absolute left-0 top-0 bottom-0 w-1 bg-amber-500" />}
                                                        <div className="flex justify-between items-start mb-2 md:mb-3 gap-2">
                                                            <div className="flex gap-2 md:gap-3 flex-1 min-w-0">
                                                                {pos.raw_data?.icon ? (
                                                                    <Image src={pos.raw_data.icon} alt={pos.title || 'Market icon'} width={40} height={40} className="w-8 h-8 md:w-10 md:h-10 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                                                ) : (
                                                                    <div className="w-8 h-8 md:w-10 md:h-10 rounded-md bg-zinc-800 flex items-center justify-center text-xs flex-shrink-0">PM</div>
                                                                )}
                                                                <div className="min-w-0 flex-1">
                                                                    <div className="text-xs md:text-sm font-medium text-white line-clamp-2">{pos.title}</div>
                                                                    <div className="flex items-center gap-1.5 md:gap-2 mt-1 flex-wrap">
                                                                        <span className={cn("text-[10px] px-1.5 py-0.5 rounded border font-bold uppercase whitespace-nowrap",
                                                                            pos.outcome === 'Yes' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                                                                        )}>{pos.outcome}</span>
                                                                        <span className="text-[10px] text-zinc-500 truncate">{pos.event_category}</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <div className="text-right flex-shrink-0">
                                                                <div className={cn("text-xs md:text-sm font-mono font-bold whitespace-nowrap", isProfit ? "text-emerald-400" : "text-red-400")}>
                                                                    {isProfit ? '+' : ''}{currency(pos.unrealized_pnl)}
                                                                </div>
                                                                <div className="text-[10px] text-zinc-500">PnL</div>
                                                            </div>
                                                        </div>

                                                        {/* Price Bar */}
                                                        <div className="bg-black rounded-lg p-2 md:p-2.5 border border-white/5 flex items-center justify-between gap-1.5 md:gap-2 text-xs font-mono flex-wrap">
                                                            <div className="text-zinc-500 text-[11px] md:text-xs whitespace-nowrap">
                                                                Entry: <span className="text-zinc-300">{cents(pos.avg_entry_price)}</span>
                                                            </div>
                                                            <ArrowRight className="w-3 h-3 text-zinc-600 flex-shrink-0" />
                                                            <div className="text-zinc-500 text-[11px] md:text-xs whitespace-nowrap">
                                                                Curr: <span className={cn("font-bold", isProfit ? "text-emerald-400" : "text-red-400")}>
                                                                    {cents(pos.current_price)}
                                                                </span>
                                                            </div>
                                                            <div className="h-3 w-px bg-white/10 hidden sm:block"></div>
                                                            <div className="text-zinc-400 text-[11px] md:text-xs whitespace-nowrap">
                                                                Size: {currency(pos.size)}
                                                            </div>
                                                        </div>

                                                        <div className="mt-2 text-right">
                                                            <a href={`https://polymarket.com/event/${pos.event_slug}`} target="_blank" className="text-[10px] text-blue-500 hover:text-blue-400 flex items-center justify-end gap-1">
                                                                View Market <ExternalLink className="w-3 h-3" />
                                                            </a>
                                                        </div>
                                                    </div>
                                                )
                                            })}
                                        </div>
                                    )}

                                    {/* Loading indicator for infinite scroll */}
                                    {isFetchingNextOpenPage && (
                                        <div className="flex justify-center items-center py-6">
                                            <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
                                        </div>
                                    )}

                                    {/* End of list indicator */}
                                    {!hasNextOpenPage && positions.length > 0 && (
                                        <div className="text-center py-6 text-xs text-zinc-600">
                                            End of list
                                        </div>
                                    )}
                                    </div>
                                </>
                            ) : (
                                <>
                                    {/* Sticky Header with Pagination */}
                                    <div className="sticky top-0 z-10 bg-[#050505] border-b border-white/5 px-4 md:px-6 py-2 md:py-3">
                                        <div className="flex items-center justify-between gap-2">
                                            <h3 className="text-xs md:text-sm font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
                                                <History className="w-4 h-4" /> Trade History
                                            </h3>
                                            <span className="text-xs text-zinc-600 whitespace-nowrap">
                                                {historyTotal} records
                                            </span>
                                        </div>
                                    </div>

                                    {/* Scrollable Content */}
                                    <div ref={scrollContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 pt-0 min-h-0">

                                    {loadingHistory ? (
                                        <div className="space-y-4">
                                            {[1, 2, 3, 4].map(i => <div key={i} className="h-32 bg-zinc-900/50 rounded-xl animate-pulse border border-white/5" />)}
                                        </div>
                                    ) : historyTrades.length === 0 ? (
                                        <div className="text-center py-20 text-zinc-500">No trade history available.</div>
                                    ) : (
                                        <div className="space-y-4">
                                            {historyTrades.map((trade) => {
                                                const invested = trade.total_bought || 0;
                                                const pnl = trade.realized_pnl || 0;
                                                const entry = trade.avg_price || 0;
                                                const exit = trade.cur_price || 0;

                                                // Determine Trade Status for Context
                                                let statusLabel = "Closed";
                                                let statusColor = "text-zinc-500";
                                                let isProfit = pnl >= 0;
                                                let displayPnl = pnl;

                                                if (exit >= 0.99) { 
                                                  statusLabel = "WON"; 
                                                  statusColor = "text-emerald-400";
                                                  isProfit = true;
                                                }
                                                else if (exit <= 0.01) { 
                                                  statusLabel = "LOST"; 
                                                  statusColor = "text-red-400";
                                                  isProfit = false;
                                                  // Force negative display for LOST trades
                                                  displayPnl = Math.abs(pnl) * -1;
                                                }
                                                else if (pnl >= 0) { 
                                                  statusLabel = "SOLD PROFIT"; 
                                                  statusColor = "text-emerald-400";
                                                  isProfit = true;
                                                }
                                                else { 
                                                  statusLabel = "SOLD LOSS"; 
                                                  statusColor = "text-red-400";
                                                  isProfit = false;
                                                }

                                                return (
                                                    <div key={trade.id} className="group relative p-3 md:p-5 rounded-xl border border-white/5 bg-zinc-900/20 hover:bg-zinc-900/40 transition-all hover:border-white/10">
                                                        <div className="flex justify-between items-start gap-2 md:gap-4 mb-3 md:mb-4">
                                                            <div className="flex items-start gap-2 md:gap-3 flex-1 min-w-0">
                                                                <div className="w-8 h-8 md:w-10 md:h-10 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-500 flex-shrink-0">PM</div>
                                                                <div className="min-w-0 flex-1">
                                                                    <h4 className="text-xs md:text-sm font-medium text-zinc-200 leading-snug mb-1 line-clamp-2">{trade.title}</h4>
                                                                    <div className="flex items-center gap-1.5 text-[10px] text-zinc-500 flex-wrap">
                                                                        <span className="truncate">{trade.event_category}</span>
                                                                        <span>•</span>
                                                                        <span className="whitespace-nowrap">{new Date(trade.timestamp * 1000).toLocaleDateString()}</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <div className="text-right flex-shrink-0">
                                                                <div className={cn("text-base md:text-lg font-mono font-bold whitespace-nowrap", isProfit ? "text-emerald-400" : "text-red-400")}>
                                                                    {isProfit ? '+' : ''}{currency(displayPnl)}
                                                                </div>
                                                                <div className={cn("text-[10px] font-bold uppercase tracking-wider", statusColor)}>
                                                                    {statusLabel}
                                                                </div>
                                                            </div>
                                                        </div>

                                                        <div className="bg-[#0a0a0a] rounded-lg p-2.5 md:p-3 border border-white/5 mb-2 md:mb-3">
                                                            <div className="flex items-center justify-between gap-2 text-xs flex-wrap sm:flex-nowrap">
                                                                <div className="flex flex-col gap-1">
                                                                    <span className="text-zinc-500 text-[10px] uppercase">Position</span>
                                                                    <span className={cn("px-1.5 md:px-2 py-0.5 rounded text-[10px] md:text-[11px] font-bold border self-start whitespace-nowrap",
                                                                        trade.outcome === 'Yes' || trade.outcome === 'Over' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                                                                    )}>{trade.outcome}</span>
                                                                </div>
                                                                <div className="flex items-center gap-2 md:gap-3">
                                                                    <div className="text-right">
                                                                        <div className="text-zinc-500 text-[10px] uppercase">Entry</div>
                                                                        <div className="font-mono text-zinc-300 text-xs">{cents(entry)}¢</div>
                                                                    </div>
                                                                    <ArrowRight className="w-3 h-3 md:w-4 md:h-4 text-zinc-600" />
                                                                    <div className="text-left">
                                                                        <div className="text-zinc-500 text-[10px] uppercase">Exit</div>
                                                                        <div className={cn("font-mono font-bold text-xs", statusColor)}>{cents(exit)}¢</div>
                                                                    </div>
                                                                </div>
                                                                <div className="text-right">
                                                                    <div className="text-zinc-500 text-[10px] uppercase">Size</div>
                                                                    <div className="font-mono text-zinc-300 text-xs whitespace-nowrap">{currency(invested)}</div>
                                                                </div>
                                                            </div>
                                                        </div>

                                                        <div className="flex justify-end">
                                                            {trade.event_slug && (
                                                                <button
                                                                    onClick={() => {
                                                                        setOpeningLink(trade.id);
                                                                        setTimeout(() => {
                                                                            window.open(`https://polymarket.com/event/${trade.event_slug}`, '_blank', 'noopener');
                                                                            setOpeningLink(null);
                                                                        }, 300);
                                                                    }}
                                                                    disabled={openingLink === trade.id}
                                                                    className="flex items-center gap-1 text-[11px] text-blue-500 hover:text-blue-400 transition-colors group/link disabled:opacity-50"
                                                                >
                                                                    {openingLink === trade.id ? (
                                                                        <>Opening... <Loader2 className="w-3 h-3 animate-spin" /></>
                                                                    ) : (
                                                                        <>Analyze Market <ExternalLink className="w-3 h-3 transition-transform group-hover/link:translate-x-0.5" /></>
                                                                    )}
                                                                </button>
                                                            )}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}

                                    {/* Loading indicator for infinite scroll */}
                                    {isFetchingNextHistoryPage && (
                                        <div className="flex justify-center items-center py-6">
                                            <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
                                        </div>
                                    )}

                                    {/* End of list indicator */}
                                    {!hasNextHistoryPage && historyTrades.length > 0 && (
                                        <div className="text-center py-6 text-xs text-zinc-600">
                                            End of list
                                        </div>
                                    )}
                                    </div>
                                </>
                            )}
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}