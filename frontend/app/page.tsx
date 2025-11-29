'use client';

import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { ArrowRight, Twitter, Zap, Waves, Target, Activity, BarChart3, User, Linkedin } from 'lucide-react';
import { Button } from '@/components/ui/button';
import InteractiveTerminal from '@/components/landing/InteractiveTerminal';
import LiveAlphaSection from '@/components/landing/LiveAlphaSection';
import Navbar from '@/components/layout/Navbar';

export default function LandingPage() {
   return (
      <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30 overflow-x-hidden font-sans">

         {/* Background */}
         <div className="fixed inset-0 z-0 pointer-events-none">
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:32px_32px]"></div>
            <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-blue-600/10 blur-[120px] rounded-full"></div>
         </div>

         <Navbar />

         {/* --- HERO SECTION --- */}
         <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-6 max-w-7xl mx-auto flex flex-col items-center text-center z-10">
            <motion.div
               initial={{ opacity: 0, y: 20 }}
               animate={{ opacity: 1, y: 0 }}
               transition={{ duration: 0.5 }}
               className="space-y-8 max-w-4xl"
            >
               <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/20 bg-blue-950/30 text-blue-400 text-xs font-medium uppercase tracking-widest">
                  <Zap className="w-3 h-3 fill-current" /> AI-Powered Alpha
               </div>

               {/* Updated Gradient Title */}
               <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1]">
                  Decode the <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 underline-offset-8">Smart Money</span> <br />
                  on Polymarket.
               </h1>

               <p className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed">
                  We track the most profitable wallets, uncover hidden liquidity, and calculate your odds.
               </p>

               <div className="pt-4">
                  <Link href="/dashboard">
                     <motion.div
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        transition={{ type: "spring", stiffness: 400, damping: 17 }}
                     >
                        <Button
                           size="lg"
                           className="group relative h-14 px-8 md:px-10 text-base md:text-lg rounded-full bg-white text-black hover:bg-blue-50 border-0 font-semibold shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
                        >
                           <span className="flex items-center gap-2">
                              Start Analyzing
                              <ArrowRight className="w-4 h-4 md:w-5 md:h-5 text-blue-600 group-hover:translate-x-1 transition-transform duration-300" />
                           </span>
                        </Button>
                     </motion.div>
                  </Link>
               </div>
            </motion.div>

            <motion.div
               initial={{ opacity: 0, y: 50 }}
               animate={{ opacity: 1, y: 0 }}
               transition={{ delay: 0.3, duration: 0.8 }}
               className="mt-20 w-full perspective-1000"
            >
               <InteractiveTerminal />
            </motion.div>
         </section>

         {/* --- IDENTIFY THE EDGE (BENTO GRID) --- */}
         <section className="py-24 px-6 max-w-7xl mx-auto relative">
            <div className="mb-16">
               <h2 className="text-3xl md:text-5xl font-bold mb-6">Identify the edge.</h2>
               <p className="text-zinc-400 max-w-xl text-lg">
                  Our engine scans thousands of markets and wallets every second.
               </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 auto-rows-[280px] md:auto-rows-[320px]">
               {/* 1. LIQUIDITY HUNT (SQUARE) */}

               <div className="md:col-span-1 rounded-2xl md:rounded-3xl border border-white/10 bg-zinc-900/20 p-6 md:p-8 group hover:border-blue-500/30 transition-all duration-500 flex flex-col overflow-hidden relative">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(59,130,246,0.1),transparent_50%)]"></div>
                  <div className="z-10">
                     <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center border border-blue-500/20 mb-6">
                        <Waves className="w-6 h-6 text-blue-400" />
                     </div>
                     <h3 className="text-xl font-bold mb-2 text-white">Liquidity Hunt</h3>
                     <p className="text-zinc-400 text-sm">Find volume surges before the crowd enters.</p>
                  </div>
                  {/* Visual: Animated Bars */}

                  <div className="mt-auto flex items-end gap-1 h-16 opacity-50">
                     {[20, 40, 60, 80, 50, 90, 30, 70, 40, 100].map((h, i) => (
                        <motion.div
                           key={i}
                           animate={{ height: [`${h}%`, `${Math.max(20, h - 20)}%`, `${h}%`] }}
                           transition={{ duration: 2, repeat: Infinity, delay: i * 0.1 }}
                           className="flex-1 bg-blue-500 rounded-t-sm"
                        />
                     ))}
                  </div>
               </div>

               {/* 2. WHALE REPLICATION (RECTANGLE) */}

               <div className="md:col-span-2 rounded-2xl md:rounded-3xl border border-white/10 bg-zinc-900/20 p-6 md:p-8 group hover:border-cyan-500/30 transition-all duration-500 flex flex-col md:flex-row items-center gap-6 md:gap-8 overflow-hidden relative">
                  <div className="flex-1 z-10 relative">
                     <div className="w-12 h-12 bg-cyan-500/10 rounded-xl flex items-center justify-center mb-6 border border-cyan-500/20">
                        <Target className="w-6 h-6 text-cyan-400" />
                     </div>
                     <h3 className="text-xl font-bold mb-2 text-white">Whale Replication</h3>
                     <p className="text-zinc-400 text-sm max-w-sm">
                        Copy the top 1% of profitable wallets. Our system filters out luck to find pure skill.
                     </p>
                  </div>
                  {/* Visual: Sync Graphic */}

                  <div className="w-full md:w-1/2 h-full bg-[#050505] rounded-xl border border-white/5 flex items-center justify-center relative overflow-hidden p-4 md:p-6">
                     <div className="flex items-center gap-3 md:gap-4 w-full max-w-xs md:max-w-none">
                        <div className="w-12 h-12 md:w-16 md:h-16 rounded-full border-2 border-cyan-500/30 flex items-center justify-center relative flex-shrink-0">
                           <User className="w-6 h-6 md:w-8 md:h-8 text-cyan-500" />
                           <div className="absolute -top-1 -right-1 w-2 h-2 md:w-3 md:h-3 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_#22d3ee] md:shadow-[0_0_10px_#22d3ee]"></div>
                        </div>
                        <div className="flex-1 h-px bg-gradient-to-r from-cyan-500/50 to-zinc-700 border-t border-dashed border-white/20 relative min-w-0">
                           <motion.div
                              animate={{ x: [-20, 100], opacity: [0, 1, 0] }}
                              transition={{ duration: 1.5, repeat: Infinity }}
                              className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 md:w-2 md:h-2 bg-white rounded-full shadow-[0_0_8px_white]"
                           />
                        </div>
                        <div className="w-10 h-10 md:w-12 md:h-12 rounded-full border border-zinc-700 flex items-center justify-center bg-zinc-900 flex-shrink-0">
                           <User className="w-4 h-4 md:w-5 md:h-5 text-zinc-500" />
                        </div>
                     </div>
                  </div>
               </div>

               {/* 3. RISK CALCULATOR (Fixed Animation) */}

               <div className="md:col-span-2 rounded-2xl md:rounded-3xl border border-white/10 bg-zinc-900/20 p-6 md:p-8 group hover:border-purple-500/30 transition-all duration-500 flex flex-col md:flex-row-reverse items-center gap-6 md:gap-8 overflow-hidden">
                  <div className="flex-1 text-right md:text-left z-10">
                     <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center mb-6 border border-purple-500/20 ml-auto md:ml-0">
                        <Activity className="w-6 h-6 text-purple-400" />
                     </div>
                     <h3 className="text-xl font-bold mb-2 text-white">Risk Calculator</h3>
                     <p className="text-zinc-400 text-sm max-w-sm ml-auto md:ml-0">
                        Real-time odds vs. implied probability scoring.
                     </p>
                  </div>
                  {/* Visual: Bell Curve Scan */}

                  <div className="w-full md:w-1/2 h-40 bg-[#050505] rounded-xl border border-white/5 relative overflow-hidden flex items-center justify-center">
                     <div className="absolute inset-0 flex items-center justify-center opacity-20">
                        {/* Static Bell Curve Path */}
                        <svg viewBox="0 0 200 120" className="w-full h-full">
                           <path d="M0,120 C50,120 75,15 100,15 S150,120 200,120" fill="none" stroke="#a855f7" strokeWidth="2.5" />
                        </svg>
                     </div>
                     {/* Scanning Line */}
                     <motion.div
                        animate={{ left: ['0%', '100%'] }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                        className="absolute top-0 bottom-0 w-px bg-purple-500 shadow-[0_0_10px_#a855f7]"
                     />
                     <div className="bg-zinc-900 border border-zinc-700 px-3 py-1 rounded text-xs text-purple-300 z-10">
                        EV: +5.3%
                     </div>
                  </div>
               </div>

               {/* 4. LIVE SIGNALS (Fixed Layout) */}
               <div className="md:col-span-1 rounded-2xl md:rounded-3xl border border-white/10 bg-zinc-900/20 p-6 md:p-8 group hover:border-emerald-500/30 transition-all duration-500 flex flex-col overflow-hidden relative">
                  <div className="z-20 bg-[#0c0c0c]/80 backdrop-blur-sm pb-4">
                     <div className="w-12 h-12 bg-emerald-500/10 rounded-xl flex items-center justify-center border border-emerald-500/20 mb-6">
                        <BarChart3 className="w-6 h-6 text-emerald-400" />
                     </div>
                     <h3 className="text-xl font-bold mb-2 text-white">Live Signals</h3>
                  </div>

                  {/* Visual: Ticker with Mask */}
                  <div className="absolute inset-x-0 bottom-0 top-32 px-8 overflow-hidden [mask-image:linear-gradient(to_bottom,transparent,black_20%,black_80%,transparent)]">
                     <motion.div
                        animate={{ y: [-100, 0] }}
                        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                        className="space-y-3"
                     >
                        {[...Array(2)].map((_, setIndex) => (
                           <div key={setIndex} className="space-y-3">
                              {["Buy YES: Fed", "Sell NO: BTC", "Buy YES: SpaceX", "Buy YES: Election"].map((txt, i) => (
                                 <div key={i} className="bg-[#050505] border border-white/10 p-3 rounded-lg text-[10px] font-mono flex justify-between items-center">
                                    <span className={txt.includes('Buy') ? 'text-emerald-400' : 'text-red-400'}>{txt}</span>
                                    <span className="text-zinc-600">Just now</span>
                                 </div>
                              ))}
                           </div>
                        ))}
                     </motion.div>
                  </div>
               </div>
            </div>
         </section>

         {/* --- LIVE ALPHA --- */}
         <LiveAlphaSection />

         {/* --- CTA --- */}
         <section className="py-24 md:py-32 px-4 md:px-6">
            <div className="max-w-5xl mx-auto relative bg-[#0a0a0a] border border-white/10 p-8 md:p-12 lg:p-20 rounded-2xl md:rounded-[2rem] text-center overflow-hidden">
               <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-cyan-500 to-emerald-500"></div>
               <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4 md:mb-6 text-white">Stop trading blind.</h2>
               <p className="text-zinc-400 mb-8 md:mb-10 max-w-lg mx-auto text-base md:text-lg">
                  The market moves fast. Our AI moves faster.
               </p>
               <Link href="/dashboard">
                  <motion.div
                     whileHover={{ scale: 1.02 }}
                     whileTap={{ scale: 0.98 }}
                     transition={{ type: "spring", stiffness: 400, damping: 17 }}
                  >
                     <Button
                        size="lg"
                        className="group relative h-12 md:h-14 lg:h-16 px-8 md:px-10 lg:px-12 text-base md:text-lg lg:text-xl rounded-full bg-gradient-to-r from-blue-600 via-cyan-600 to-emerald-600 hover:from-blue-700 hover:via-cyan-700 hover:to-emerald-700 text-white border-0 font-bold shadow-[0_0_50px_-10px_rgba(59,130,246,0.5)] hover:shadow-[0_0_70px_-15px_rgba(59,130,246,0.7)] transition-all duration-500 overflow-hidden"
                     >
                        <span className="relative z-10 flex items-center gap-2 md:gap-3">
                           <span className="text-white font-bold drop-shadow-sm">
                              Launch Terminal
                           </span>
                           <ArrowRight className="w-5 h-5 md:w-6 md:h-6 text-white group-hover:translate-x-1 transition-transform duration-300" />
                        </span>
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-400/30 via-cyan-400/30 to-emerald-400/30 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-full" />
                     </Button>
                  </motion.div>
               </Link>
            </div>
         </section>

         {/* --- FOOTER --- */}
         <footer className="border-t border-white/5 bg-[#020202] py-16">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between gap-12">
               <div className="space-y-4">
                  <div className="flex items-center gap-2">
                     <div className="w-6 h-6 rounded flex items-center justify-center overflow-hidden"><Image src="/logo.svg" alt="PolyTier" width={24} height={24} className="object-contain" /></div>
                     <span className="font-bold">
                       <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-500 to-blue-600">Poly</span>
                       <span className="text-white">Tier</span>
                     </span>
                  </div>
                  <p className="text-zinc-500 text-sm">The intelligence layer for prediction markets.</p>
               </div>
               <div className="flex gap-6 text-zinc-500">
                  <Twitter className="w-5 h-5 hover:text-white cursor-pointer" />
                  <Linkedin className="w-5 h-5 hover:text-white cursor-pointer" />
               </div>
            </div>
         </footer>
      </div>
   );
}