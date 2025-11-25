// FILE: frontend/components/chat/ChatInterface.tsx
'use client';

import { useEffect, useRef } from 'react';
import { useChat } from '@/lib/hooks/use-chat';
import { cn } from '@/lib/utils';
import { User, Brain } from 'lucide-react';
import Markdown from 'react-markdown';
import ChatInput from '@/components/chat/ChatInput';
import { ChatMessage, AlphaPositionsData, FundsData, TradeResultData, PnLData } from '@/types/chat';
import { AlphaCard, FundsCard, TradeSuccessCard, PnLCard, ToolLog, ThoughtLog, GenericToolWidget } from '@/components/chat/ChatWidgets';
import Image from 'next/image';

export default function ChatInterface() {
  const { messages, status, isTyping, sendMessage } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logic
  useEffect(() => {
    if (scrollRef.current) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        if (scrollRef.current) {
          scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
        }
      });
    }
  }, [messages, isTyping]);

  return (
    <div className="relative flex flex-col h-full w-full max-w-full bg-[#09090b]">

      {/* Chat Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pt-4 pb-96 md:pb-44 px-3 md:px-0 scrollbar-thin">
        <div className="max-w-3xl mx-auto w-full space-y-4 md:space-y-6">
          {messages.map((msg: ChatMessage) => {

            // 1. SYSTEM WIDGETS (Render Full Width)
            if (msg.role === 'system' && msg.widgetType && msg.widgetData) {
              return (
                <div key={msg.id} className="flex w-full justify-center py-2 animate-in fade-in slide-in-from-bottom-4">
                  {msg.widgetType === 'positions' && <AlphaCard data={msg.widgetData as AlphaPositionsData} />}
                  {msg.widgetType === 'funds' && <FundsCard data={msg.widgetData as FundsData} />}
                  {msg.widgetType === 'trade_result' && <TradeSuccessCard data={msg.widgetData as TradeResultData} />}
                  {msg.widgetType === 'pnl' && <PnLCard data={msg.widgetData as PnLData} />}
                  {!['positions', 'funds', 'trade_result', 'pnl'].includes(msg.widgetType) && (
                    <GenericToolWidget toolName={msg.widgetType} args={{}} result={msg.widgetData} />
                  )}
                </div>
              );
            }

            // 2. STANDARD MESSAGES
            // Skip empty assistant messages unless they have tools/thoughts
            if (msg.role === 'assistant' && !msg.content && (!msg.toolInvocations?.length) && (!msg.thoughts?.length)) {
              return null;
            }

            return (
              <div key={msg.id} className={cn("flex gap-2 md:gap-5 w-full px-2 md:px-4 py-1", msg.role === 'user' ? "flex-row-reverse" : "flex-row")}>

                {/* Avatar */}
                <div className={cn(
                  "w-7 h-7 md:w-9 md:h-9 rounded-xl flex items-center justify-center flex-shrink-0 border shadow-sm",
                  msg.role === 'user'
                    ? "bg-zinc-800 border-zinc-700"
                    : "bg-gradient-to-b from-blue-900 to-blue-950 border-blue-800/30"
                )}>
                  {msg.role === 'user' ? (
                    <User className="w-3.5 h-3.5 md:w-4 md:h-4 text-zinc-400" />
                  ) : (
                    <Image src="/favicon.svg" alt="AI" width={16} height={16} className="opacity-90 invert md:w-[20px] md:h-[20px]" />
                  )}
                </div>

                {/* Content Column */}
                <div className={cn("flex flex-col max-w-[78%] md:max-w-[75%]", msg.role === 'user' ? "items-end" : "items-start")}>

                  {/* Thoughts / Reasoning Log */}
                  {msg.thoughts && msg.thoughts.length > 0 && (
                    <ThoughtLog thoughts={msg.thoughts} />
                  )}

                  {/* Tool Logs */}
                  {msg.toolInvocations && msg.toolInvocations.length > 0 && (
                    <ToolLog tools={msg.toolInvocations} />
                  )}

                  {/* Text Bubble */}
                  {msg.content && (
                    <div className={cn(
                      "px-3 py-2.5 md:px-4 md:py-3 rounded-2xl text-[14px] md:text-[15px] leading-relaxed shadow-sm",
                      msg.role === 'user'
                        ? "bg-white text-black rounded-tr-none"
                        : "bg-zinc-900 border border-white/5 text-zinc-200 rounded-tl-none"
                    )}>
                      <Markdown>{msg.content}</Markdown>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {/* Thinking / Typing Indicator */}
          {isTyping && (
            <div className="flex gap-2 md:gap-5 px-2 md:px-4 py-1 animate-in fade-in duration-300">
              <div className="w-7 h-7 md:w-9 md:h-9 rounded-xl bg-zinc-900 border border-white/5 flex items-center justify-center flex-shrink-0">
                <Image src="/favicon.svg" alt="AI" width={16} height={16} className="opacity-40 grayscale md:w-[20px] md:h-[20px]" />
              </div>
              <div className="flex flex-col gap-1.5 md:gap-2 pt-0.5 md:pt-1">
                <div className="flex items-center gap-1.5 md:gap-2 text-[10px] md:text-xs text-blue-400 font-medium uppercase tracking-wider animate-pulse">
                  <Brain className="w-2.5 h-2.5 md:w-3 md:h-3" /> Reasoning...
                </div>
                <div className="flex gap-1">
                  <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce [animation-delay:-0.3s]" />
                  <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce [animation-delay:-0.15s]" />
                  <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce" />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#09090b] from-80% via-[#09090b]/95 to-transparent pt-32 md:pt-12 pb-4 md:pb-6 px-3 md:px-4 z-20">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={sendMessage} disabled={status !== 'connected'} />
          <div className="text-center mt-2 md:mt-3">
            <p className="text-[9px] md:text-[10px] text-zinc-600 font-mono tracking-widest uppercase">
              Powered by Convo AI & Polymarket Data
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}