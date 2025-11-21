'use client';

import { useEffect, useRef } from 'react';
import { useChat } from '@/lib/hooks/use-chat';
import { cn } from '@/lib/utils';
import { User, Sparkles, RefreshCw } from 'lucide-react';
import Markdown from 'react-markdown';
import ChatInput from '@/components/chat/ChatInput';
import { ChatMessage, AlphaPositionsData, FundsData, TradeResultData, PnLData } from '@/types/chat';
import { AlphaCard, FundsCard, TradeSuccessCard, PnLCard, ToolLog, ThoughtLog } from '@/components/chat/ChatWidgets';
import Image from 'next/image';

export default function ChatInterface() {
  const { messages, status, isTyping, sendMessage, resetSession } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages, isTyping]);

  return (
    <div className="flex flex-col h-full w-full max-w-full bg-[#09090b]">
      
      {/* Header */}
      <header className="h-14 border-b border-white/5 flex items-center justify-between px-4 md:px-6 bg-[#09090b]/50 backdrop-blur absolute top-0 w-full z-10">
        <div className="flex items-center gap-2">
          <Image src="/favicon.svg" alt="Poly" width={20} height={20} className="rounded-sm" />
          <span className="font-medium text-zinc-200">PolyAgent</span>
          <span className="bg-blue-500/10 text-blue-400 text-xs px-2 py-0.5 rounded border border-blue-500/20">v2.0</span>
        </div>
        <div className="flex items-center gap-3">
            {/* Reset Button */}
            <button onClick={resetSession} className="text-zinc-500 hover:text-white transition-colors" title="Reset Chat">
                <RefreshCw className="w-4 h-4" />
            </button>
            <div className={cn("flex items-center gap-2 text-xs",
            status === 'connected' ? "text-emerald-500" : "text-amber-500")}>
            <span className="w-2 h-2 rounded-full bg-current shadow-[0_0_8px_currentColor]" />
            {status === 'connected' ? 'Online' : 'Connecting'}
            </div>
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto pt-20 pb-40 scrollbar-thin scrollbar-thumb-zinc-800">
        
        {messages.length === 0 ? (
          <EmptyState onSuggestionClick={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto w-full px-4 md:px-0 space-y-6">
            {messages.map((msg: ChatMessage) => {
              
              // --- SYSTEM WIDGETS ---
              if (msg.role === 'system' && msg.widgetType && msg.widgetData) {
                return (
                  <div key={msg.id} className="flex w-full justify-start pl-12 animate-in fade-in slide-in-from-bottom-2">
                     {msg.widgetType === 'positions' && <AlphaCard data={msg.widgetData as AlphaPositionsData} />}
                     {msg.widgetType === 'funds' && <FundsCard data={msg.widgetData as FundsData} />}
                     {msg.widgetType === 'trade_result' && <TradeSuccessCard data={msg.widgetData as TradeResultData} />}
                     {msg.widgetType === 'pnl' && <PnLCard data={msg.widgetData as PnLData} />}
                  </div>
                );
              }

              // --- 2. RENDER CHAT BUBBLES (USER/ASSISTANT) ---
              // Skip rendering empty assistant messages unless they have tool invocations or thoughts
              if (msg.role === 'assistant' && !msg.content && 
                  (!msg.toolInvocations || msg.toolInvocations.length === 0) &&
                  (!msg.thoughts || msg.thoughts.length === 0)) {
                return null;
              }

              return (
                <div key={msg.id} className="flex gap-4 md:gap-6 w-full group animate-in fade-in slide-in-from-bottom-1">
                  {/* Avatar Side */}
                  <div className="flex-shrink-0 flex flex-col relative items-end">
                    <div className={cn(
                      "w-8 h-8 rounded-lg flex items-center justify-center shadow-lg",
                      msg.role === 'user' ? "bg-[#202123]" : "bg-black/50 border border-white/10"
                    )}>
                      {msg.role === 'user' ? (
                        <User className="w-5 h-5 text-zinc-400" />
                      ) : (
                        // Use Favicon here instead of Sparkles
                        <Image src="/favicon.svg" alt="AI" width={20} height={20} className="opacity-90" />
                      )}
                    </div>
                  </div>

                  {/* Message Content */}
                  <div className="relative flex-1 overflow-hidden pt-1">
                    <div className="text-sm font-bold text-zinc-300 mb-1">
                      {msg.role === 'user' ? 'You' : 'Poly'}
                    </div>

                    {/* RENDER THOUGHTS IF EXIST */}
                    {msg.thoughts && msg.thoughts.length > 0 && (
                        <ThoughtLog thoughts={msg.thoughts} />
                    )}

                    {/* RENDER TOOL LOGS IF EXIST */}
                    {msg.toolInvocations && msg.toolInvocations.length > 0 && (
                        <ToolLog tools={msg.toolInvocations} />
                    )}

                    {/* Only render content if it exists */}
                    {msg.content && (
                      <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-[#1a1a1a] max-w-none text-zinc-100 text-[15px]">
                        <Markdown>{msg.content}</Markdown>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Typing Indicator */}
            {isTyping && (
               <div className="flex gap-4 md:gap-6 w-full animate-in fade-in pl-0.5">
                 <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-center justify-center">
                    <Image src="/favicon.svg" alt="AI" width={20} height={20} className="opacity-50 grayscale" />
                 </div>
                 <div className="flex items-center gap-1 pt-3">
                    <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce"></div>
                 </div>
               </div>
            )}
          </div>
        )}
      </div>

      <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#09090b] via-[#09090b] to-transparent pt-10 pb-6 px-4 z-20">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={sendMessage} disabled={status !== 'connected'} />
          <div className="text-center mt-2">
            <p className="text-[10px] text-zinc-600 font-mono">
              POWERED BY CONVO AI & POLYMARKET DATA
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}

function EmptyState({ onSuggestionClick }: { onSuggestionClick: (text: string) => void }) {
  // ... (Keep your existing EmptyState logic)
  // Re-pasted for completeness if needed, but assuming you keep the existing one:
  const suggestions = [
    { label: "How many active wallets?", query: "How many wallets are we currently tracking?" },
    { label: "Analyze Elite Traders", query: "Who are the most profitable traders this week?" },
    { label: "Scan Markets", query: "List active markets with > $100k volume." },
    { label: "System Status", query: "Is the database connection healthy?" },
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full px-4 text-center animate-in fade-in zoom-in duration-500">
      <div className="w-16 h-16 bg-zinc-800 rounded-2xl flex items-center justify-center mb-6 shadow-2xl shadow-blue-900/20">
        <Sparkles className="w-8 h-8 text-blue-400" />
      </div>
      <h2 className="text-2xl font-semibold text-white mb-2">Polymarket Intelligence</h2>
      <p className="text-zinc-400 mb-10 max-w-md">
        Connected to your private Supabase cluster. Ask me about traders, markets, or wallet positions.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
        {suggestions.map((s, i) => (
          <button
            key={i}
            onClick={() => onSuggestionClick(s.query)}
            className="text-left p-4 rounded-xl border border-zinc-800 hover:bg-[#1a1a1a] hover:border-zinc-700 transition-all group"
          >
            <div className="text-sm font-medium text-zinc-200 mb-1 group-hover:text-blue-400 transition-colors">
              {s.label}
            </div>
            <div className="text-xs text-zinc-500 truncate">
              &quot;{s.query}&quot;
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}