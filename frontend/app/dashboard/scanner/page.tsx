'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { fetchMarkets, fetchEvents, fetchMarketStats } from '@/lib/api/markets';
import { Market, MarketEvent } from '@/types/market';
import MarketCharts from '@/components/scanner/MarketCharts';
import { Input } from '@/components/ui/input';
import { Search, ChevronLeft, ChevronRight, ArrowUpDown, ExternalLink, Loader2, Clock, Filter, X } from 'lucide-react';
import { cn } from '@/lib/utils';

type ViewMode = 'markets' | 'events';

export default function ScannerPage() {
    // View Mode
    const [viewMode, setViewMode] = useState<ViewMode>('markets');

    // Pagination & Filtering
    const [page, setPage] = useState(1);
    const [search, setSearch] = useState('');
    const [filterExpiring, setFilterExpiring] = useState(false);
    const [showFilters, setShowFilters] = useState(false);
    const [categoryFilter, setCategoryFilter] = useState<string>('');
    const [minVolume, setMinVolume] = useState<string>('');
    const [minLiquidity, setMinLiquidity] = useState<string>('');

    const [sortConfig, setSortConfig] = useState<{ key: string; dir: 'asc' | 'desc' }>({
        key: viewMode === 'markets' ? 'volume_24h' : 'total_volume',
        dir: 'desc'
    });

    // Reset page on view mode change
    useEffect(() => {
        setPage(1);
        setSortConfig({
            key: viewMode === 'markets' ? 'volume_24h' : 'total_volume',
            dir: 'desc'
        });
    }, [viewMode]);

    // 1. Fetch Stats
    const { data: stats } = useQuery({
        queryKey: ['marketStats'],
        queryFn: fetchMarketStats,
        staleTime: 60 * 1000 * 5, // 5 minutes
    });

    // 2. Fetch Markets
    const {
        data: marketsData,
        isLoading: marketsLoading,
        isPlaceholderData: isMarketsPlaceholder
    } = useQuery({
        queryKey: ['markets', page, search, sortConfig, filterExpiring, categoryFilter, minVolume, minLiquidity],
        queryFn: () => fetchMarkets(
            page,
            search,
            sortConfig.key as keyof Market,
            sortConfig.dir === 'asc',
            filterExpiring,
            categoryFilter || undefined,
            minVolume ? parseFloat(minVolume) : undefined,
            minLiquidity ? parseFloat(minLiquidity) : undefined
        ),
        enabled: viewMode === 'markets',
        placeholderData: keepPreviousData,
    });

    // 3. Fetch Events
    const {
        data: eventsData,
        isLoading: eventsLoading,
        isPlaceholderData: isEventsPlaceholder
    } = useQuery({
        queryKey: ['events', page, search, sortConfig, filterExpiring, categoryFilter, minVolume, minLiquidity],
        queryFn: () => fetchEvents(
            page,
            search,
            sortConfig.key as keyof MarketEvent,
            sortConfig.dir === 'asc',
            filterExpiring,
            categoryFilter || undefined,
            minVolume ? parseFloat(minVolume) : undefined,
            minLiquidity ? parseFloat(minLiquidity) : undefined
        ),
        enabled: viewMode === 'events',
        placeholderData: keepPreviousData,
    });

    // Derived Data
    const markets = marketsData?.data || [];
    const events = eventsData?.data || [];
    const totalCount = viewMode === 'markets' ? (marketsData?.total || 0) : (eventsData?.total || 0);
    const loading = viewMode === 'markets' ? marketsLoading : eventsLoading;

    // Categories for filter
    const categories = stats?.categoryDistribution?.map(c => c.name) || [];

    const handleSort = (key: string) => {
        setSortConfig(curr => ({ key, dir: curr.key === key && curr.dir === 'desc' ? 'asc' : 'desc' }));
        setPage(1);
    };

    const clearFilters = () => {
        setCategoryFilter('');
        setMinVolume('');
        setMinLiquidity('');
        setFilterExpiring(false);
        setPage(1);
    };

    const hasActiveFilters = categoryFilter || minVolume || minLiquidity || filterExpiring;

    // Helpers
    const currency = (n: number | null) => n ? `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '-';
    const percent = (n: number | null) => n ? `${(n * 100).toFixed(1)}%` : '-';

    return (
        <div className="flex flex-col h-full bg-[#050505] relative overflow-hidden">
            <div className="flex-1 overflow-y-auto px-4 md:px-6 lg:px-8 py-6 md:py-8">

                {/* Header */}
                <div className="mb-6 md:mb-8">
                    <h1 className="text-xl md:text-2xl font-bold text-white mb-2">Market Scanner</h1>
                    <p className="text-zinc-400 text-sm">Real-time analysis of active liquidity and volume across the ecosystem.</p>
                </div>

                {/* Visualizations */}
                <MarketCharts stats={stats || null} />

                {/* Tabs */}
                <div className="flex gap-2 mb-4 md:mb-6 border-b border-white/5">
                    <button
                        onClick={() => setViewMode('markets')}
                        className={cn(
                            "px-3 md:px-4 py-2 text-sm font-medium transition-colors border-b-2 whitespace-nowrap",
                            viewMode === 'markets'
                                ? "text-white border-blue-500"
                                : "text-zinc-400 border-transparent hover:text-white"
                        )}
                    >
                        Markets
                    </button>
                    <button
                        onClick={() => setViewMode('events')}
                        className={cn(
                            "px-3 md:px-4 py-2 text-sm font-medium transition-colors border-b-2 whitespace-nowrap",
                            viewMode === 'events'
                                ? "text-white border-blue-500"
                                : "text-zinc-400 border-transparent hover:text-white"
                        )}
                    >
                        Events
                    </button>
                </div>

                {/* Filter Toolbar */}
                <div className="flex flex-col gap-4 mb-4 md:mb-6">
                    <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
                        <div className="flex flex-col sm:flex-row gap-3 w-full lg:w-auto">
                            {/* Search */}
                            <div className="relative w-full sm:w-96 group">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
                                <Input
                                    placeholder={`Find ${viewMode}...`}
                                    value={search}
                                    onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                                    className="pl-10"
                                />
                            </div>
                            {/* Expiring Filter Toggle */}
                            <button
                                onClick={() => { setFilterExpiring(!filterExpiring); setPage(1); }}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-2 rounded-xl border transition-all text-sm font-medium whitespace-nowrap",
                                    filterExpiring
                                        ? "bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20"
                                        : "bg-zinc-900/50 border-white/10 text-zinc-400 hover:text-white hover:bg-zinc-900"
                                )}
                            >
                                <Clock className="w-4 h-4" />
                                Expiring (48h)
                            </button>
                            {/* Advanced Filters Toggle */}
                            <button
                                onClick={() => setShowFilters(!showFilters)}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-2 rounded-xl border transition-all text-sm font-medium whitespace-nowrap",
                                    showFilters || hasActiveFilters
                                        ? "bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20"
                                        : "bg-zinc-900/50 border-white/10 text-zinc-400 hover:text-white hover:bg-zinc-900"
                                )}
                            >
                                <Filter className="w-4 h-4" />
                                Filters
                                {hasActiveFilters && (
                                    <span className="ml-1 px-1.5 py-0.5 bg-blue-500/20 rounded text-xs">
                                        {[categoryFilter, minVolume, minLiquidity, filterExpiring].filter(Boolean).length}
                                    </span>
                                )}
                            </button>
                        </div>
                        {/* Pagination */}
                        <div className="flex items-center gap-3 ml-auto">
                            <span className="text-xs text-zinc-500 font-mono">
                                {totalCount === 0 ? '0-0' : `${(page - 1) * 50 + 1}-${Math.min(page * 50, totalCount)}`} of {totalCount}
                            </span>
                            <div className="flex gap-1">
                                <button
                                    onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1 || loading}
                                    className="p-2 border border-white/10 rounded-lg hover:bg-zinc-900 disabled:opacity-30 transition-colors"
                                >
                                    <ChevronLeft className="w-4 h-4 text-zinc-400" />
                                </button>
                                <button
                                    onClick={() => setPage(p => p + 1)} disabled={(viewMode === 'markets' ? markets.length : events.length) < 50 || loading}
                                    className="p-2 border border-white/10 rounded-lg hover:bg-zinc-900 disabled:opacity-30 transition-colors"
                                >
                                    <ChevronRight className="w-4 h-4 text-zinc-400" />
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Advanced Filters Panel */}
                    {showFilters && (
                        <div className="bg-zinc-900/50 border border-white/10 rounded-xl p-4">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-sm font-medium text-white">Advanced Filters</h3>
                                {hasActiveFilters && (
                                    <button
                                        onClick={clearFilters}
                                        className="text-xs text-zinc-400 hover:text-white flex items-center gap-1"
                                    >
                                        <X className="w-3 h-3" />
                                        Clear all
                                    </button>
                                )}
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {/* Category Filter */}
                                <div>
                                    <label className="text-xs text-zinc-400 mb-1 block">Category</label>
                                    <select
                                        value={categoryFilter}
                                        onChange={(e) => { setCategoryFilter(e.target.value); setPage(1); }}
                                        className="w-full px-3 py-2 bg-zinc-800 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:border-blue-500"
                                    >
                                        <option value="">All Categories</option>
                                        {categories.map(cat => (
                                            <option key={cat} value={cat}>{cat}</option>
                                        ))}
                                    </select>
                                </div>
                                {/* Min Volume Filter */}
                                <div>
                                    <label className="text-xs text-zinc-400 mb-1 block">Min Volume ($)</label>
                                    <Input
                                        type="number"
                                        placeholder="0"
                                        value={minVolume}
                                        onChange={(e) => { setMinVolume(e.target.value); setPage(1); }}
                                        className="bg-zinc-800 border-white/10"
                                    />
                                </div>
                                {/* Min Liquidity Filter */}
                                <div>
                                    <label className="text-xs text-zinc-400 mb-1 block">Min Liquidity ($)</label>
                                    <Input
                                        type="number"
                                        placeholder="0"
                                        value={minLiquidity}
                                        onChange={(e) => { setMinLiquidity(e.target.value); setPage(1); }}
                                        className="bg-zinc-800 border-white/10"
                                    />
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* MOBILE CARD VIEW (Visible on small screens) */}
                <div className="block md:hidden space-y-3">
                    {loading ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                        </div>
                    ) : (viewMode === 'markets' ? markets.length === 0 : events.length === 0) ? (
                        <div className="text-center text-zinc-500 py-12">
                            No {viewMode} found.
                        </div>
                    ) : viewMode === 'markets' ? (
                        markets.map((m) => {
                            const slug = m.event_slug || m.raw_data?.slug || m.id;
                            const marketUrl = `https://polymarket.com/event/${slug}`;
                            const imageUrl = m.raw_data?.icon || m.raw_data?.image;
                            const avgPrice = m.p_yes || 0;

                            return (
                                <div key={m.id} className="bg-zinc-900/30 border border-white/5 rounded-xl p-4">
                                    <div className="flex items-start gap-3 mb-3">
                                        {imageUrl ? (
                                            <Image src={imageUrl} alt="Icon" width={40} height={40} className="w-10 h-10 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                        ) : (
                                            <div className="w-10 h-10 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-600 flex-shrink-0">PM</div>
                                        )}
                                        <div className="min-w-0 flex-1">
                                            <div className="text-sm font-medium text-zinc-200 line-clamp-2 mb-1">{m.title}</div>
                                            <span className="inline-block px-2 py-0.5 rounded bg-zinc-800 text-[10px] text-zinc-400">
                                                {m.category}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-3 mb-3">
                                        <div className="bg-zinc-900/50 p-2 rounded-lg">
                                            <div className="text-[10px] text-zinc-500 mb-0.5">Prediction</div>
                                            <div className="flex items-center gap-2">
                                                {m.raw_data?.outcomes && m.raw_data.outcomes.length > 0 ? (
                                                    <>
                                                        <span className={cn("text-sm font-bold", avgPrice > 0.5 ? "text-emerald-400" : "text-blue-400")}>
                                                            {avgPrice > 0.5 ? m.raw_data.outcomes[0] : (m.raw_data.outcomes[1] || "No")}
                                                        </span>
                                                        <span className={cn("text-xs font-mono", avgPrice > 0.5 ? "text-emerald-400/70" : "text-blue-400/70")}>
                                                            {percent(avgPrice > 0.5 ? avgPrice : (1 - avgPrice))}
                                                        </span>
                                                    </>
                                                ) : (
                                                    <span className="text-sm font-mono text-zinc-400">{percent(avgPrice)}</span>
                                                )}
                                            </div>
                                            {/* Probability Bar */}
                                            <div className="h-1 w-full bg-zinc-800 rounded-full mt-1 overflow-hidden">
                                                <div
                                                    className={cn("h-full rounded-full", avgPrice > 0.5 ? "bg-emerald-500" : "bg-blue-500")}
                                                    style={{ width: `${avgPrice * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                        <div className="bg-zinc-900/50 p-2 rounded-lg">
                                            <div className="text-[10px] text-zinc-500 mb-0.5">24h Vol</div>
                                            <div className="text-sm font-mono text-zinc-300">{currency(m.volume_24h)}</div>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between text-xs text-zinc-500">
                                        <span>Liq: {currency(m.liquidity)}</span>
                                        <a href={marketUrl} target="_blank" className="text-blue-400 flex items-center gap-1">
                                            View <ExternalLink className="w-3 h-3" />
                                        </a>
                                    </div>
                                </div>
                            );
                        })
                    ) : (
                        events.map((e) => (
                            <div key={e.id} className="bg-zinc-900/30 border border-white/5 rounded-xl p-4">
                                <div className="flex items-start gap-3 mb-3">
                                    {e.raw_data?.icon ? (
                                        <Image src={e.raw_data.icon} alt="Icon" width={40} height={40} className="w-10 h-10 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                    ) : (
                                        <div className="w-10 h-10 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-600 flex-shrink-0">EV</div>
                                    )}
                                    <div className="min-w-0 flex-1">
                                        <div className="text-sm font-medium text-zinc-200 line-clamp-2 mb-1">{e.title}</div>
                                        <span className="inline-block px-2 py-0.5 rounded bg-zinc-800 text-[10px] text-zinc-400">
                                            {e.category}
                                        </span>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3 mb-3">
                                    <div className="bg-zinc-900/50 p-2 rounded-lg">
                                        <div className="text-[10px] text-zinc-500 mb-0.5">Total Vol</div>
                                        <div className="text-sm font-mono text-zinc-300">{currency(e.total_volume)}</div>
                                    </div>
                                    <div className="bg-zinc-900/50 p-2 rounded-lg">
                                        <div className="text-[10px] text-zinc-500 mb-0.5">Markets</div>
                                        <div className="text-sm font-mono text-zinc-300">{e.market_count}</div>
                                    </div>
                                </div>
                                <div className="flex items-center justify-end">
                                    <a href={`https://polymarket.com/event/${e.slug || e.id}`} target="_blank" className="text-blue-400 flex items-center gap-1 text-xs">
                                        View Event <ExternalLink className="w-3 h-3" />
                                    </a>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* DESKTOP TABLE VIEW (Hidden on small screens) */}
                <div className="hidden md:block bg-zinc-900/20 border border-white/5 rounded-xl overflow-x-auto min-h-[400px]">
                    <table className="w-full text-left border-collapse">
                        <thead className="bg-zinc-900/30 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold sticky top-0 z-10 backdrop-blur-sm">
                            <tr>
                                {viewMode === 'markets' ? (
                                    <>
                                        <th className="p-4 pl-6 min-w-[300px]">Market</th>
                                        <th className="p-4 min-w-[120px]">Category</th>
                                        <th className="p-4 text-right cursor-pointer hover:text-white min-w-[140px]" onClick={() => handleSort('p_yes')}>
                                            Probability <ArrowUpDown className="w-3 h-3 inline" />
                                        </th>
                                        <th className="p-4 text-right cursor-pointer hover:text-white min-w-[120px]" onClick={() => handleSort('volume_24h')}>
                                            24h Vol <ArrowUpDown className="w-3 h-3 inline" />
                                        </th>
                                        <th className="p-4 text-right cursor-pointer hover:text-white min-w-[120px]" onClick={() => handleSort('total_volume')}>
                                            Total Vol <ArrowUpDown className="w-3 h-3 inline" />
                                        </th>
                                        <th className="p-4 text-right cursor-pointer hover:text-white min-w-[120px]" onClick={() => handleSort('liquidity')}>
                                            Liquidity <ArrowUpDown className="w-3 h-3 inline" />
                                        </th>
                                        <th className="p-4 text-right min-w-[110px]">Deadline</th>
                                        <th className="p-4 min-w-[60px]"></th>
                                    </>
                                ) : (
                                    <>
                                        <th className="p-4 pl-6 min-w-[300px]">Event</th>
                                        <th className="p-4 min-w-[120px]">Category</th>
                                        <th className="p-4 text-right min-w-[100px]">Markets</th>
                                        <th className="p-4 text-right cursor-pointer hover:text-white min-w-[120px]" onClick={() => handleSort('total_volume')}>
                                            Total Vol <ArrowUpDown className="w-3 h-3 inline" />
                                        </th>
                                        <th className="p-4 text-right cursor-pointer hover:text-white min-w-[120px]" onClick={() => handleSort('total_liquidity')}>
                                            Liquidity <ArrowUpDown className="w-3 h-3 inline" />
                                        </th>
                                        <th className="p-4 text-right min-w-[110px]">Deadline</th>
                                        <th className="p-4 min-w-[60px]"></th>
                                    </>
                                )}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-sm">
                            {loading ? (
                                <tr>
                                    <td colSpan={viewMode === 'markets' ? 8 : 7} className="h-64 relative">
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                                        </div>
                                    </td>
                                </tr>
                            ) : (viewMode === 'markets' ? markets.length === 0 : events.length === 0) ? (
                                <tr>
                                    <td colSpan={viewMode === 'markets' ? 8 : 7} className="p-12 text-center text-zinc-500">
                                        No {viewMode} found matching criteria.
                                    </td>
                                </tr>
                            ) : viewMode === 'markets' ? (
                                markets.map((m) => {
                                    const slug = m.event_slug || m.raw_data?.slug || m.id;
                                    const marketUrl = `https://polymarket.com/event/${slug}`;
                                    const imageUrl = m.raw_data?.icon || m.raw_data?.image;
                                    const avgPrice = m.p_yes || 0;

                                    return (
                                        <tr key={m.id} className="group hover:bg-white/[0.02] transition-colors">
                                            <td className="p-4 pl-6">
                                                <div className="flex items-center gap-3 min-w-0">
                                                    {imageUrl ? (
                                                        <Image src={imageUrl} alt="Market icon" width={32} height={32} className="w-8 h-8 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                                    ) : (
                                                        <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-600 flex-shrink-0">PM</div>
                                                    )}
                                                    <div className="flex flex-col min-w-0 flex-1">
                                                        <span className="font-medium text-zinc-200 break-words" title={m.title}>{m.title}</span>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="p-4">
                                                <span className="px-2.5 py-1 rounded-md bg-zinc-800/50 border border-white/5 text-xs text-zinc-400 whitespace-nowrap">
                                                    {m.category}
                                                </span>
                                            </td>
                                            <td className="p-4 text-right font-mono">
                                                <div className="flex flex-col items-end gap-1">
                                                    {m.raw_data?.outcomes && m.raw_data.outcomes.length > 0 ? (
                                                        <div className="flex items-center gap-2 justify-end">
                                                            <span className={cn("text-xs font-medium", avgPrice > 0.5 ? "text-emerald-400" : "text-blue-400")}>
                                                                {avgPrice > 0.5 ? m.raw_data.outcomes[0] : (m.raw_data.outcomes[1] || "No")}
                                                            </span>
                                                            <span className={cn("font-bold", avgPrice > 0.5 ? "text-emerald-400" : "text-blue-400")}>
                                                                {percent(avgPrice > 0.5 ? avgPrice : (1 - avgPrice))}
                                                            </span>
                                                        </div>
                                                    ) : (
                                                        <span className={cn("font-bold", avgPrice > 0.5 ? "text-emerald-400" : "text-blue-400")}>
                                                            {percent(avgPrice)}
                                                        </span>
                                                    )}
                                                    <div className="w-16 h-1 bg-zinc-800 rounded-full overflow-hidden">
                                                        <div
                                                            className={cn("h-full rounded-full", avgPrice > 0.5 ? "bg-emerald-500" : "bg-blue-500")}
                                                            style={{ width: `${avgPrice * 100}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="p-4 text-right font-mono text-zinc-300">
                                                {currency(m.volume_24h)}
                                            </td>
                                            <td className="p-4 text-right font-mono text-zinc-300">
                                                {currency(m.total_volume)}
                                            </td>
                                            <td className="p-4 text-right font-mono text-zinc-400">
                                                {currency(m.liquidity)}
                                            </td>
                                            <td className="p-4 text-right text-xs text-zinc-500 font-mono whitespace-nowrap">
                                                {m.close_date ? new Date(m.close_date).toLocaleDateString() : '-'}
                                            </td>
                                            <td className="p-4 text-right">
                                                <a
                                                    href={marketUrl}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-blue-400 transition-all"
                                                >
                                                    <ExternalLink className="w-4 h-4" />
                                                </a>
                                            </td>
                                        </tr>
                                    );
                                })
                            ) : (
                                events.map((e) => {
                                    const eventUrl = `https://polymarket.com/event/${e.slug || e.id}`;
                                    const imageUrl = e.raw_data?.icon || e.raw_data?.image;

                                    return (
                                        <tr key={e.id} className="group hover:bg-white/[0.02] transition-colors">
                                            <td className="p-4 pl-6">
                                                <div className="flex items-center gap-3 min-w-0">
                                                    {imageUrl ? (
                                                        <Image src={imageUrl} alt="Event icon" width={32} height={32} className="w-8 h-8 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                                    ) : (
                                                        <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-600 flex-shrink-0">EV</div>
                                                    )}
                                                    <div className="flex flex-col min-w-0 flex-1">
                                                        <span className="font-medium text-zinc-200 break-words" title={e.title}>{e.title}</span>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="p-4">
                                                <span className="px-2.5 py-1 rounded-md bg-zinc-800/50 border border-white/5 text-xs text-zinc-400 whitespace-nowrap">
                                                    {e.category}
                                                </span>
                                            </td>
                                            <td className="p-4 text-right font-mono text-zinc-300">
                                                {e.market_count || 0}
                                            </td>
                                            <td className="p-4 text-right font-mono text-zinc-300">
                                                {currency(e.total_volume)}
                                            </td>
                                            <td className="p-4 text-right font-mono text-zinc-400">
                                                {currency(e.total_liquidity)}
                                            </td>
                                            <td className="p-4 text-right text-xs text-zinc-500 font-mono whitespace-nowrap">
                                                {e.end_date ? new Date(e.end_date).toLocaleDateString() : '-'}
                                            </td>
                                            <td className="p-4 text-right">
                                                <a
                                                    href={eventUrl}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-blue-400 transition-all"
                                                >
                                                    <ExternalLink className="w-4 h-4" />
                                                </a>
                                            </td>
                                        </tr>
                                    );
                                })
                            )}
                        </tbody>
                    </table>
                </div>

            </div>
        </div>
    );
}