'use client';

import { useState, useEffect } from 'react';
import { fetchEliteTraders, fetchEliteTags, fetchEventCategories } from '@/lib/api/elites';
import { EliteTrader, EliteTagComparison } from '@/types/elite';
import ElitePerformanceChart from '@/components/elites/ElitePerformanceChart';
import ElitePositionsDrawer from '@/components/elites/ElitePositionsDrawer';
import { ChevronLeft, ChevronRight, Crown, LayoutGrid, List, Filter, X, ChevronDown, ChevronUp, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { getWalletDisplayName, getWalletAvatar } from '@/lib/utils/wallet-display';

export default function EliteWalletsPage() {
  const [activeTab, setActiveTab] = useState<'sectors' | 'traders'>('sectors');

  // Data State
  const [tags, setTags] = useState<EliteTagComparison[]>([]);
  const [traders, setTraders] = useState<EliteTrader[]>([]);
  const [totalTraders, setTotalTraders] = useState(0);
  const [eventCategories, setEventCategories] = useState<string[]>([]);

  // Trader Table State
  const [page, setPage] = useState(1);
  const [selectedTrader, setSelectedTrader] = useState<EliteTrader | null>(null);

  // Filter State - Multiple categories can be selected
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [selectedSector, setSelectedSector] = useState(''); // For sector click navigation
  const [filterExpanded, setFilterExpanded] = useState(false);
  const [categorySearch, setCategorySearch] = useState('');

  // Sector Table Sort State
  const [sectorSort, setSectorSort] = useState<'events' | 'edge' | 'volume'>('events');

  // Initial Load
  useEffect(() => {
    const init = async () => {
      const [tagsData, tradersData, categoriesData] = await Promise.all([
        fetchEliteTags(),
        fetchEliteTraders(1),
        fetchEventCategories()
      ]);
      setTags(tagsData);
      setTraders(tradersData.data);
      setTotalTraders(tradersData.total);
      setEventCategories(categoriesData);
    };
    init();
  }, []);

  // Pagination Effect for Traders
  useEffect(() => {
    if (activeTab === 'traders') {
      fetchEliteTraders(page, selectedCategories).then(res => {
        setTraders(res.data);
        setTotalTraders(res.total);
      });
    }
  }, [page, activeTab, selectedCategories]);

  // Currency Formatter
  const currency = (n: number | null) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n || 0);

  // Sort Function for Sectors
  const sortedTags = [...tags].sort((a, b) => {
    if (sectorSort === 'events') return (b.number_of_events || 0) - (a.number_of_events || 0);
    if (sectorSort === 'edge') return b.performance_edge - a.performance_edge;
    if (sectorSort === 'volume') return b.volume_concentration - a.volume_concentration;
    return 0;
  });

  // Handler for Sector Click
  const handleSectorClick = (tag: string) => {
    setSelectedSector(tag);
    // Convert sector tag to category filter if it matches an event category
    // Check if tag matches any event category (case-insensitive)
    const matchingCategory = eventCategories.find(cat =>
      cat.toLowerCase() === tag.toLowerCase() ||
      tag.toLowerCase().includes(cat.toLowerCase()) ||
      cat.toLowerCase().includes(tag.toLowerCase())
    );
    if (matchingCategory) {
      setSelectedCategories([matchingCategory]);
    } else {
      // If no exact match, try to set it anyway (might be a sector tag that maps to category)
      setSelectedCategories([tag]);
    }
    setActiveTab('traders');
    setPage(1);
  };

  // Handler for Category Toggle
  const toggleCategory = (category: string) => {
    setSelectedCategories(prev => {
      if (prev.includes(category)) {
        return prev.filter(c => c !== category);
      } else {
        return [...prev, category];
      }
    });
    setPage(1);
  };

  // Clear all category filters
  const clearCategoryFilters = () => {
    setSelectedCategories([]);
    setSelectedSector('');
    setPage(1);
  };

  // Filter categories by search term
  const filteredCategories = eventCategories.filter(cat =>
    cat.toLowerCase().includes(categorySearch.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full bg-[#050505] overflow-hidden">

      {/* Page Header */}
      <div className="px-6 md:px-8 pt-8 pb-6 border-b border-white/5">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
              <Crown className="w-6 h-6 text-amber-400 fill-amber-400/20" /> Elite Intelligence
            </h1>
            <p className="text-zinc-400 text-sm">
              Analyze top-performing sectors or track individual elite trader positions.
            </p>
          </div>

          {/* Custom Tab Switcher */}
          <div className="bg-zinc-900/50 p-1 rounded-xl border border-white/5 flex gap-1">
            <button
              onClick={() => setActiveTab('sectors')}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'sectors'
                  ? "bg-zinc-800 text-white shadow-lg shadow-black/20 border border-white/10"
                  : "text-zinc-500 hover:text-zinc-300"
              )}
            >
              <LayoutGrid className="w-4 h-4" /> Sector Analysis
            </button>
            <button
              onClick={() => setActiveTab('traders')}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'traders'
                  ? "bg-zinc-800 text-white shadow-lg shadow-black/20 border border-white/10"
                  : "text-zinc-500 hover:text-zinc-300"
              )}
            >
              <List className="w-4 h-4" /> Elite Traders
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 md:px-8 py-8">

        {/* --- TAB 1: SECTOR ANALYSIS --- */}
        {activeTab === 'sectors' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">

            {/* Chart Section */}
            <ElitePerformanceChart
              data={tags.slice(0, 15)}
              onSectorClick={handleSectorClick}
            />

            {/* Table */}
            <div className="mt-8">
              <div className="flex justify-between items-end mb-4">
                <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Detailed Sector Statistics</h3>

                {/* Quick Sort Toggles */}
                <div className="flex gap-2">
                  {['events', 'edge', 'volume'].map((key) => (
                    <button
                      key={key}
                      onClick={() => setSectorSort(key as 'events' | 'edge' | 'volume')}
                      className={cn(
                        "text-[10px] uppercase font-bold px-3 py-1 rounded border transition-colors",
                        sectorSort === key
                          ? "bg-zinc-800 text-white border-zinc-600"
                          : "text-zinc-600 border-transparent hover:text-zinc-400"
                      )}
                    >
                      By {key}
                    </button>
                  ))}
                </div>
              </div>

              <div className="hidden md:block bg-zinc-900/20 border border-white/5 rounded-xl overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead className="bg-zinc-900/20 text-[11px] text-zinc-500 uppercase font-semibold sticky top-0 z-10 backdrop-blur-sm">
                    <tr>
                      <th className="p-4 pl-6 min-w-[180px]">Category</th>
                      <th className="p-4 text-right min-w-[120px]">Elite Traders</th>
                      <th className="p-4 text-right min-w-[120px]">Total Traders</th>
                      <th className="p-4 text-right min-w-[130px]">Events Tracked</th>
                      <th className="p-4 text-right min-w-[100px]">Elite ROI</th>
                      <th className="p-4 text-right min-w-[100px]">Avg ROI</th>
                      <th className="p-4 text-right min-w-[100px]">Edge</th>
                      <th className="p-4 text-right min-w-[140px]">Elite Vol Share</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-sm">
                    {sortedTags.map((tag) => (
                      <tr
                        key={tag.tag}
                        onClick={() => handleSectorClick(tag.tag)}
                        className="hover:bg-white/[0.04] transition-colors group cursor-pointer"
                      >
                        <td className="p-4 pl-6 font-medium text-white">{tag.tag}</td>

                        {/* Elite vs Total Traders */}
                        <td className="p-4 text-right font-mono text-emerald-400 font-bold">
                          {tag.elite_trader_count}
                        </td>
                        <td className="p-4 text-right font-mono text-zinc-400">
                          {tag.total_trader_count || (tag.elite_trader_count + (tag.non_elite_trader_count || 0))}
                        </td>

                        {/* Activity Level */}
                        <td className="p-4 text-right font-mono text-zinc-300">
                          {tag.number_of_events?.toLocaleString() || 0}
                        </td>

                        <td className="p-4 text-right font-mono font-bold text-emerald-400">
                          {tag.elite_avg_roi.toFixed(1)}%
                        </td>
                        <td className="p-4 text-right font-mono text-zinc-500">
                          {tag.non_elite_avg_roi.toFixed(1)}%
                        </td>
                        <td className="p-4 text-right font-mono">
                          <span className={cn("px-2 py-1 rounded text-xs font-bold",
                            tag.performance_edge > 0 ? "bg-amber-500/10 text-amber-400 border border-amber-500/20" : "bg-zinc-800 text-zinc-500"
                          )}>
                            {tag.performance_edge > 0 ? '+' : ''}{tag.performance_edge.toFixed(1)}%
                          </span>
                        </td>

                        {/* Visual Bar for Concentration */}
                        <td className="p-4 text-right font-mono text-zinc-400 flex items-center justify-end gap-2">
                          <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500" style={{ width: `${tag.volume_concentration}%` }} />
                          </div>
                          {tag.volume_concentration.toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Mobile Card View for Sectors */}
              <div className="md:hidden space-y-3 mt-4">
                {sortedTags.map((tag) => (
                  <div
                    key={tag.tag}
                    onClick={() => handleSectorClick(tag.tag)}
                    className="bg-zinc-900/30 border border-white/5 rounded-xl p-4 active:bg-zinc-900/50 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <h4 className="text-sm font-bold text-white">{tag.tag}</h4>
                        <div className="text-xs text-zinc-500 mt-0.5">{tag.number_of_events?.toLocaleString() || 0} Events</div>
                      </div>
                      <span className={cn("px-2 py-1 rounded text-xs font-bold",
                        tag.performance_edge > 0 ? "bg-amber-500/10 text-amber-400 border border-amber-500/20" : "bg-zinc-800 text-zinc-500"
                      )}>
                        Edge: {tag.performance_edge > 0 ? '+' : ''}{tag.performance_edge.toFixed(1)}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-3">
                      <div>
                        <div className="text-[10px] text-zinc-500 uppercase">Elite ROI</div>
                        <div className="text-sm font-mono font-bold text-emerald-400">{tag.elite_avg_roi.toFixed(1)}%</div>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-zinc-500 uppercase">Avg ROI</div>
                        <div className="text-sm font-mono text-zinc-400">{tag.non_elite_avg_roi.toFixed(1)}%</div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-3 border-t border-white/5">
                      <div className="text-xs text-zinc-400">
                        <span className="text-white font-bold">{tag.elite_trader_count}</span> Elites
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-zinc-500 uppercase">Vol Share</span>
                        <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500" style={{ width: `${tag.volume_concentration}%` }} />
                        </div>
                        <span className="text-xs font-mono text-zinc-300">{tag.volume_concentration.toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}


        {/* --- TAB 2: TRADER LEADERBOARD --- */}
        {activeTab === 'traders' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="h-full flex flex-col">

            {/* Compact Category Filter Section */}
            <div className="mb-4">
              {/* Selected Filters Bar - Always Visible */}
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                <div className="flex items-center gap-2 text-xs text-zinc-400">
                  <Filter className="w-3 h-3" />
                  <span className="font-medium">Filters:</span>
                </div>
                {selectedCategories.length > 0 ? (
                  <>
                    {selectedCategories.map((category) => (
                      <button
                        key={category}
                        onClick={() => toggleCategory(category)}
                        className="group flex items-center gap-1.5 px-2.5 py-1 bg-blue-500/20 text-blue-300 border border-blue-500/40 rounded-md text-xs font-medium hover:bg-blue-500/30 transition-colors"
                      >
                        {category}
                        <X className="w-3 h-3 opacity-60 group-hover:opacity-100" />
                      </button>
                    ))}
                    <button
                      onClick={clearCategoryFilters}
                      className="px-2 py-1 text-xs text-zinc-400 hover:text-white underline"
                    >
                      Clear All
                    </button>
                  </>
                ) : (
                  <span className="text-xs text-zinc-500 italic">None selected</span>
                )}
                <button
                  onClick={() => setFilterExpanded(!filterExpanded)}
                  className="ml-auto flex items-center gap-1 px-2 py-1 text-xs text-zinc-400 hover:text-zinc-300 transition-colors"
                >
                  {filterExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  {filterExpanded ? 'Hide' : 'Show'} Categories ({eventCategories.length})
                </button>
              </div>

              {/* Expandable Category Filter Grid */}
              {filterExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="bg-zinc-900/30 border border-white/5 rounded-lg p-3">
                    {/* Search Input for Categories */}
                    <div className="mb-3">
                      <input
                        type="text"
                        placeholder="Search categories..."
                        value={categorySearch}
                        onChange={(e) => setCategorySearch(e.target.value)}
                        className="w-full px-3 py-1.5 bg-zinc-800/50 border border-white/5 rounded-md text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500/40 focus:bg-zinc-800 transition-colors"
                      />
                    </div>

                    {/* Scrollable Category Grid */}
                    <div className="max-h-48 overflow-y-auto pr-2">
                      <div className="flex flex-wrap gap-1.5">
                        {filteredCategories.map((category) => {
                          const isSelected = selectedCategories.includes(category);
                          return (
                            <button
                              key={category}
                              onClick={() => toggleCategory(category)}
                              className={cn(
                                "px-2.5 py-1 rounded-md text-xs font-medium transition-all border",
                                isSelected
                                  ? "bg-blue-500/20 text-blue-300 border-blue-500/40 shadow-sm shadow-blue-500/10"
                                  : "bg-zinc-800/50 text-zinc-400 border-white/5 hover:bg-zinc-700 hover:text-zinc-300 hover:border-white/10"
                              )}
                            >
                              {category}
                            </button>
                          );
                        })}
                      </div>
                      {filteredCategories.length === 0 && (
                        <div className="text-center text-zinc-500 text-xs py-4">
                          No categories found matching &quot;{categorySearch}&quot;
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </div>

            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-4">
                <span className="text-xs text-zinc-500">
                  {totalTraders} {selectedCategories.length > 0 ? 'Filtered' : 'Total'} Elites Tracked
                </span>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex gap-1">
                  <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="p-2 hover:bg-zinc-800 rounded border border-white/10 disabled:opacity-30 transition-colors"><ChevronLeft className="w-4 h-4" /></button>
                  <button onClick={() => setPage(p => p + 1)} disabled={traders.length < 50} className="p-2 hover:bg-zinc-800 rounded border border-white/10 disabled:opacity-30 transition-colors"><ChevronRight className="w-4 h-4" /></button>
                </div>
              </div>
            </div>

            <div className="hidden md:block bg-zinc-900/20 border border-white/5 rounded-xl overflow-x-auto flex-1">
              <table className="w-full text-left border-collapse">
                <thead className="bg-zinc-900/20 text-[11px] text-zinc-500 uppercase font-semibold sticky top-0 z-10 backdrop-blur-sm">
                  <tr>
                    <th className="p-4 pl-6 min-w-[80px]">Rank</th>
                    <th className="p-4 min-w-[200px]">Wallet Address</th>
                    <th className="p-4 text-center min-w-[120px]">Open Position</th>
                    <th className="p-4 text-right min-w-[120px]">Win Rate</th>
                    <th className="p-4 text-right min-w-[100px]">ROI</th>
                    <th className="p-4 text-right min-w-[140px]">Total Volume</th>
                    <th className="p-4 text-center min-w-[100px]">Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-sm">
                  {traders.map((trader) => (
                    <tr
                      key={trader.proxy_wallet}
                      onClick={() => setSelectedTrader(trader)}
                      className="group hover:bg-white/[0.04] transition-colors cursor-pointer"
                    >
                      <td className="p-4 pl-6">
                        <div className={cn("w-6 h-6 flex items-center justify-center rounded text-xs font-bold",
                          trader.rank_in_tier <= 3 ? "bg-amber-500 text-black" : "bg-zinc-800 text-zinc-500"
                        )}>
                          {trader.rank_in_tier}
                        </div>
                      </td>
                      <td className="p-4 font-mono text-zinc-300 group-hover:text-blue-400 transition-colors truncate max-w-[200px]">
                        <div className="flex items-center gap-3">
                          {getWalletAvatar(trader) ? (
                            <img
                              src={getWalletAvatar(trader)!}
                              alt={getWalletDisplayName(trader)}
                              className="w-8 h-8 rounded-lg object-cover bg-zinc-800 border border-white/10 flex-shrink-0"
                            />
                          ) : (
                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-zinc-800 to-zinc-900 border border-white/10 flex items-center justify-center flex-shrink-0">
                              <User className="w-4 h-4 text-zinc-500" />
                            </div>
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="font-medium text-zinc-200 group-hover:text-blue-400 transition-colors truncate">
                              {getWalletDisplayName(trader)}
                            </div>
                            {(trader.pseudonym || trader.name) && (
                              <div className="font-mono text-xs text-zinc-500 truncate">
                                {trader.proxy_wallet.slice(0, 10)}...{trader.proxy_wallet.slice(-4)}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="p-4 text-center">
                        {(trader.n_open_positions || 0) > 0 ? (
                          <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-green-500/10 text-green-400 rounded-lg border border-green-500/20">
                            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                            <span className="text-xs font-bold">{trader.n_open_positions}</span>
                          </div>
                        ) : (
                          <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-red-500/10 text-red-400 rounded-lg border border-red-500/20">
                            <span className="w-2 h-2 bg-red-400 rounded-full" />
                            <span className="text-xs font-bold">0</span>
                          </div>
                        )}
                      </td>
                      <td className="p-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500" style={{ width: `${trader.win_rate}%` }} />
                          </div>
                          <span className="text-xs font-mono">{trader.win_rate?.toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="p-4 text-right font-mono font-bold text-emerald-400">
                        +{trader.roi?.toFixed(1)}%
                      </td>
                      <td className="p-4 text-right font-mono text-zinc-400">
                        {currency(trader.total_volume)}
                      </td>
                      <td className="p-4 text-center">
                        <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg border border-cyan-500/30">
                          <div className="text-[10px] text-cyan-300 font-semibold">⭐</div>
                          <span className="text-sm font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-blue-300">
                            {trader.composite_score?.toFixed(0)}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile Card View for Traders */}
            <div className="md:hidden space-y-3 flex-1 overflow-y-auto">
              {traders.map((trader) => (
                <div
                  key={trader.proxy_wallet}
                  onClick={() => setSelectedTrader(trader)}
                  className="bg-zinc-900/30 border border-white/5 rounded-xl p-4 active:bg-zinc-900/50 transition-colors relative overflow-hidden"
                >
                  {/* Decorative gradient overlay for high scores */}
                  {(trader.composite_score || 0) > 80 && (
                    <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full blur-3xl -z-0" />
                  )}

                  <div className="flex justify-between items-start mb-3 relative z-10">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <div className={cn("w-10 h-10 flex items-center justify-center rounded-lg text-sm font-bold shrink-0",
                        trader.rank_in_tier <= 3 ? "bg-gradient-to-br from-amber-400 to-amber-600 text-black shadow-lg shadow-amber-500/20" : "bg-zinc-800 text-zinc-500"
                      )}>
                        {trader.rank_in_tier}
                      </div>
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        {getWalletAvatar(trader) ? (
                          <img
                            src={getWalletAvatar(trader)!}
                            alt={getWalletDisplayName(trader)}
                            className="w-10 h-10 rounded-lg object-cover bg-zinc-800 border border-white/10 flex-shrink-0"
                          />
                        ) : (
                          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-zinc-800 to-zinc-900 border border-white/10 flex items-center justify-center flex-shrink-0">
                            <User className="w-5 h-5 text-zinc-500" />
                          </div>
                        )}
                        <div className="min-w-0 flex-1">
                          <div className="font-medium text-sm text-white truncate">{getWalletDisplayName(trader)}</div>
                          <div className="text-xs text-zinc-500">
                            {trader.n_positions || 0} {(trader.n_positions || 0) === 1 ? 'Position' : 'Positions'}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col items-end gap-2 shrink-0">
                      {/* Active/Inactive Badge - Pushed to Right */}
                      {(trader.n_open_positions || 0) > 0 ? (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-500/10 text-green-400 rounded-lg text-[10px] font-bold border border-green-500/20">
                          <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                          {trader.n_open_positions} Active
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-500/10 text-red-400 rounded-lg text-[10px] font-bold border border-red-500/20">
                          <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                          Inactive
                        </span>
                      )}


                      {/* Simple Score Badge */}
                      <span className="inline-flex items-center gap-1 px-2 py-1 bg-cyan-500/10 text-cyan-300 rounded-lg text-[10px] font-bold border border-cyan-500/30">
                        <span className="text-[10px]">⭐</span>
                        <span className="text-sm font-black">{trader.composite_score?.toFixed(0)}</span>
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 py-3 border-t border-white/5 border-b mb-3 relative z-10">
                    <div>
                      <div className="text-[10px] text-zinc-500 uppercase mb-1 font-semibold">Win Rate</div>
                      <div className="text-base font-mono font-bold text-white">{trader.win_rate?.toFixed(0)}%</div>
                      <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden mt-1">
                        <div className="h-full bg-blue-500" style={{ width: `${trader.win_rate}%` }} />
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-[10px] text-zinc-500 uppercase mb-1 font-semibold">ROI</div>
                      <div className="text-base font-mono font-bold text-emerald-400">+{trader.roi?.toFixed(1)}%</div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] text-zinc-500 uppercase mb-1 font-semibold">Volume</div>
                      <div className="text-xs font-mono text-zinc-300">{currency(trader.total_volume)}</div>
                    </div>
                  </div>

                  <div className="text-center text-xs text-blue-400 font-semibold relative z-10">
                    View Positions &rarr;
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

      </div>

      {/* Drawer Component */}
      <ElitePositionsDrawer
        trader={selectedTrader}
        highlightCategory={selectedCategories.length === 1 ? selectedCategories[0] : selectedSector}
        onClose={() => setSelectedTrader(null)}
      />
    </div>
  );
}