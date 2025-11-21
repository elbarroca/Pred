'use client';

import {
  TrendingUp, Briefcase, Shield, Zap, CheckCircle2,
  ArrowRight, Loader2, AlertCircle, Wallet, Clock, Brain
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { ToolInvocation, AlphaPositionsData, FundsData, TradeResultData, PnLData, Fund } from '@/types/chat';

// --- 0. THOUGHT / REASONING LOG ---
export function ThoughtLog({ thoughts }: { thoughts: string[] }) {
  if (!thoughts || thoughts.length === 0) return null;

  return (
    <div className="my-3 mb-4 space-y-2 bg-blue-900/10 border border-blue-500/20 rounded-lg p-3">
      <div className="flex items-center gap-2 text-[10px] text-blue-400 uppercase tracking-wider font-semibold mb-2">
        <Brain className="w-3 h-3" />
        Thinking
      </div>
      {thoughts.map((thought, i) => (
        <div key={i} className="text-[11px] text-blue-300/80 font-mono animate-in fade-in slide-in-from-left-2">
          {thought}
        </div>
      ))}
    </div>
  );
}

// --- 0.5. TOOL INVOCATION LOG ---
export function ToolLog({ tools }: { tools: ToolInvocation[] }) {
  if (!tools || tools.length === 0) return null;

  return (
    <div className="my-3 mb-4 space-y-2 bg-zinc-900/30 border border-white/5 rounded-lg p-3">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2">Tool Execution</div>
      {tools.map((tool, i) => (
        <div key={i} className="flex items-center gap-2 text-[11px] font-mono text-zinc-400 animate-in fade-in slide-in-from-left-2">
          {tool.status === 'pending' ? (
            <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
          ) : tool.status === 'error' ? (
            <AlertCircle className="w-3 h-3 text-red-500" />
          ) : (
            <CheckCircle2 className="w-3 h-3 text-emerald-500" />
          )}
          
          <span className={cn(
            tool.status === 'pending' && "animate-pulse text-blue-400",
            tool.status === 'error' && "text-red-400",
            tool.status === 'complete' && "text-emerald-400"
          )}>
            {tool.status === 'pending' ? 'Invoking' : tool.status === 'error' ? 'Failed' : 'Executed'}: 
            <span className="text-zinc-300 mx-1 font-semibold">{tool.toolName}</span>
          </span>
        </div>
      ))}
    </div>
  );
}

// --- 1. ALPHA POSITIONS (High-End Trading Card) ---
export function AlphaCard({ data }: { data: AlphaPositionsData }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      className="my-4 w-full max-w-md bg-[#0a0a0a] border border-white/10 rounded-xl overflow-hidden shadow-2xl shadow-emerald-900/10"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-900/20 to-transparent px-4 py-3 border-b border-emerald-500/20 flex justify-between items-center">
        <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase tracking-widest">
          <TrendingUp className="w-3.5 h-3.5" /> Alpha Signal
        </div>
        <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-[10px] text-zinc-500 font-mono">LIVE FEED</span>
        </div>
      </div>

      {/* Table Header */}
      <div className="grid grid-cols-12 px-4 py-2 bg-zinc-900/50 text-[10px] text-zinc-500 uppercase font-medium border-b border-white/5">
         <div className="col-span-6">Market / Whale</div>
         <div className="col-span-3 text-right">Entry</div>
         <div className="col-span-3 text-right">ROI</div>
      </div>

      {/* Rows */}
      <div className="divide-y divide-white/5">
        {data.data.map((pos, i: number) => (
          <div key={i} className="grid grid-cols-12 items-center p-4 hover:bg-white/[0.02] transition-colors group cursor-pointer">
            <div className="col-span-6 pr-2">
              <div className="text-sm font-medium text-zinc-200 group-hover:text-blue-400 transition-colors truncate">{pos.market}</div>
              <div className="flex items-center gap-2 mt-1">
                 <span className={cn("px-1.5 py-0.5 rounded text-[9px] font-bold uppercase border", 
                    pos.outcome === 'Yes' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                 )}>{pos.outcome}</span>
                 <span className="text-[10px] text-zinc-600 font-mono">{pos.whale}</span>
              </div>
            </div>
            <div className="col-span-3 text-right text-xs font-mono text-zinc-400">
               {pos.entry}¢
            </div>
            <div className="col-span-3 text-right">
               <div className="text-emerald-400 font-mono text-xs font-bold bg-emerald-500/10 inline-block px-1.5 py-0.5 rounded border border-emerald-500/20">
                 +{pos.roi}%
               </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Footer */}
      <button className="w-full py-2 text-[10px] text-zinc-500 hover:text-white bg-zinc-900/30 hover:bg-zinc-900 border-t border-white/5 transition-colors">
        COPY ALL TRADES
      </button>
    </motion.div>
  );
}

// --- 2. FUNDS / VAULTS (Investment Product Style) ---
export function FundsCard({ data }: { data: FundsData }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      className="my-4 w-full max-w-md bg-[#09090b] border border-white/10 rounded-xl overflow-hidden shadow-2xl"
    >
      <div className="bg-zinc-900/80 p-4 border-b border-white/5 flex justify-between items-center backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-purple-500/10 rounded-md text-purple-400 border border-purple-500/20 shadow-[0_0_10px_rgba(168,85,247,0.15)]">
            <Briefcase className="w-3.5 h-3.5" />
          </div>
          <span className="font-bold text-zinc-200 text-sm tracking-tight">Active Vaults</span>
        </div>
        <span className="text-[10px] text-zinc-500 font-mono">
          {data.data.length} POOLS
        </span>
      </div>

      <div className="divide-y divide-white/5">
        {data.data.map((fund: Fund) => (
          <div key={fund.id} className="p-4 hover:bg-zinc-800/30 transition-all group cursor-pointer">
            <div className="flex justify-between items-start mb-1.5">
              <h4 className="text-sm font-bold text-white group-hover:text-purple-400 transition-colors">
                {fund.name}
              </h4>
              <span className={cn("text-xs font-mono font-bold", fund.roi_30d > 0 ? 'text-emerald-400' : 'text-red-400')}>
                {fund.roi_30d > 0 ? '+' : ''}{fund.roi_30d}% <span className="text-[9px] text-zinc-600 font-sans font-normal ml-0.5">30d</span>
              </span>
            </div>
            
            <p className="text-xs text-zinc-500 mb-3 line-clamp-1">{fund.description}</p>
            
            <div className="flex items-center justify-between">
              <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold uppercase border flex items-center gap-1", 
                fund.risk === 'High' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                fund.risk === 'Medium' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              )}>
                {fund.risk === 'High' && <Zap className="w-3 h-3" />}
                {fund.risk === 'Low' && <Shield className="w-3 h-3" />}
                {fund.risk}
              </span>
              <span className="text-[10px] text-zinc-600 font-mono bg-zinc-900 border border-white/5 px-1.5 py-0.5 rounded">
                TVL: <span className="text-zinc-400">{fund.tvl}</span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// --- 3. TRADE SUCCESS (Receipt Style) ---
export function TradeSuccessCard({ data }: { data: TradeResultData }) {
  return (
    <motion.div 
      initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
      className="my-4 bg-zinc-900 border border-blue-500/30 rounded-xl p-5 w-full max-w-sm relative overflow-hidden group"
    >
       {/* Glow Effect */}
       <div className="absolute top-0 right-0 w-40 h-40 bg-blue-500/10 blur-[60px] rounded-full pointer-events-none translate-x-10 -translate-y-10 group-hover:bg-blue-500/20 transition-all duration-1000"></div>
       
       <div className="flex items-center gap-3 mb-5 relative z-10">
         <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 border border-blue-500/30 shadow-[0_0_20px_rgba(59,130,246,0.3)]">
           <CheckCircle2 className="w-5 h-5" />
         </div>
         <div>
           <div className="text-white font-bold text-sm">Order Executed</div>
           <div className="text-[10px] text-blue-400 font-mono uppercase tracking-wider flex items-center gap-1">
             <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"/> Confirmed on-chain
           </div>
         </div>
       </div>

       <div className="space-y-1 relative z-10">
         {/* Receipt Details */}
         <div className="bg-black/40 rounded-lg border border-white/5 overflow-hidden">
            <div className="p-3 border-b border-white/5 flex justify-between items-center">
               <span className="text-zinc-500 text-xs">Market</span>
               <span className="text-zinc-200 text-xs font-medium text-right truncate max-w-[140px]">{data.market}</span>
            </div>
            <div className="p-3 border-b border-white/5 flex justify-between items-center">
               <span className="text-zinc-500 text-xs">Outcome</span>
               <span className="text-white font-bold text-xs px-2 py-0.5 bg-white/10 rounded">{data.outcome}</span>
            </div>
            <div className="p-3 flex justify-between items-center bg-blue-500/5">
               <span className="text-zinc-500 text-xs">Total</span>
               <span className="text-emerald-400 font-mono font-bold text-sm">${data.amount} USDC</span>
            </div>
         </div>
       </div>

       <div className="mt-4 text-center">
          <div className="text-[10px] text-zinc-600 font-mono flex items-center justify-center gap-1 hover:text-zinc-400 cursor-pointer transition-colors">
            TX: {data.txHash} <ArrowRight className="w-2.5 h-2.5" />
          </div>
       </div>
    </motion.div>
  );
}

// --- 4. PNL CARD (Performance Metrics) ---
export function PnLCard({ data }: { data: PnLData }) {
  const isDailyPositive = data.daily.includes('+');
  const isTotalPositive = !data.all_time.includes('-'); // Assuming negative has '-'

  return (
    <motion.div 
      initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
      className="my-4 w-full max-w-[280px] bg-[#0a0a0a] border border-white/10 rounded-xl overflow-hidden shadow-xl"
    >
      {/* Header */}
      <div className="bg-zinc-900/50 px-4 py-3 border-b border-white/5 flex items-center gap-2">
         <div className="p-1.5 bg-zinc-800 rounded border border-white/5">
            <Wallet className="w-3.5 h-3.5 text-zinc-400" />
         </div>
         <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">Performance</span>
      </div>

      <div className="p-4 grid grid-cols-1 gap-3">
         {/* Daily Metric */}
         <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-zinc-500">
               <Clock className="w-3 h-3" />
               <span className="text-xs font-medium">Today</span>
            </div>
            <div className={cn("font-mono font-bold text-sm px-2 py-0.5 rounded border", 
               isDailyPositive 
                 ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
                 : "bg-red-500/10 text-red-400 border-red-500/20"
            )}>
               {data.daily}
            </div>
         </div>

         <div className="h-px bg-white/5 w-full my-1" />

         {/* All Time Metric */}
         <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-zinc-500">
               <TrendingUp className="w-3 h-3" />
               <span className="text-xs font-medium">All Time</span>
            </div>
            <div className={cn("font-mono font-bold text-sm", 
               isTotalPositive ? "text-emerald-400" : "text-red-400"
            )}>
               {data.all_time}
            </div>
         </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-zinc-900/30 border-t border-white/5 text-[10px] text-zinc-500 text-center font-mono">
         REALIZED + UNREALIZED
      </div>
    </motion.div>
  );
}