'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutGrid,
  MessageSquareText,
  MessageCircle,
  MessageSquare,
  MessageSquareMore,
  MessageSquareDashed,
  WalletMinimal,
  LogOut,
  Crown,
  ChevronLeft,
  ChevronRight,
  User,
  Settings2
} from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';

// Mock History Data (In a real app, this comes from Supabase)
const recentChats = [
  { id: 1, title: "Wallet Analysis: 0x83...9a", date: "2h ago" },
  { id: 2, title: "Top Traders Q3", date: "1d ago" },
  { id: 3, title: "US Election Volatility", date: "3d ago" },
];

// Array of chat icons for variety
const chatIcons = [
  MessageSquareText,
  MessageCircle,
  MessageSquare,
  MessageSquareMore,
  MessageSquareDashed,
];

const navItems = [
  { name: 'Elite Wallets', icon: Crown, href: '/dashboard/elites' },
  { name: 'Market Scanner', icon: LayoutGrid, href: '/dashboard/scanner' },
  { name: 'Wallet Tracker', icon: WalletMinimal, href: '/dashboard/wallets' },
];

const SIDEBAR_STORAGE_KEY = 'sidebar-collapsed';

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  
  // Initialize state from localStorage with lazy initialization
  const [isCollapsed, setIsCollapsed] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(SIDEBAR_STORAGE_KEY);
      return saved === 'true';
    }
    return false;
  });

  // User profile popup state
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);
  const profilePopupRef = useRef<HTMLDivElement>(null);

  // Close popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      const isClickOnButton = profileMenuRef.current?.contains(target);
      const isClickOnPopup = profilePopupRef.current?.contains(target);
      
      if (!isClickOnButton && !isClickOnPopup) {
        setIsProfileMenuOpen(false);
      }
    };

    if (isProfileMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isProfileMenuOpen]);

  // Assign random chat icons to each chat item (stable per chat ID)
  const chatsWithIcons = useMemo(() => {
    return recentChats.map((chat) => {
      // Use chat ID as seed for consistent icon assignment
      const iconIndex = (chat.id - 1) % chatIcons.length;
      return {
        ...chat,
        icon: chatIcons[iconIndex],
      };
    });
  }, []);

  // Save sidebar state to localStorage
  const toggleSidebar = () => {
    setIsCollapsed((prev) => {
      const newState = !prev;
      localStorage.setItem(SIDEBAR_STORAGE_KEY, String(newState));
      return newState;
    });
  };

  const handleSignOut = () => {
    router.push('/');
  };

  const handleChatClick = () => {
    // Navigate to dashboard (chat page)
    router.push('/dashboard');
    // TODO: In the future, restore specific chat session here
  };

  return (
    <div className="relative h-full hidden md:block" style={{ overflow: 'visible' }}>
      {/* Toggle Button - Positioned Outside Sidebar, Always Visible */}
      <button
        onClick={toggleSidebar}
        className={cn(
          "absolute z-[9997] w-8 h-8 rounded-full bg-white border-2 border-white flex items-center justify-center",
          "hover:bg-zinc-100 hover:scale-110 transition-all duration-200 shadow-xl",
          "text-black hover:text-zinc-800 cursor-pointer",
          "top-16"
        )}
        style={{
          pointerEvents: 'auto',
          left: isCollapsed ? '56px' : '264px', // Position at sidebar width - 16px (half button width)
          transition: 'left 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {isCollapsed ? (
          <ChevronRight className="w-4 h-4 text-black transition-transform duration-300" />
        ) : (
          <ChevronLeft className="w-4 h-4 text-black transition-transform duration-300" />
        )}
      </button>
      
      <aside 
        className={cn(
          "flex-shrink-0 bg-[#050505] flex flex-col border-r border-white/[0.08] h-full relative overflow-y-auto",
          "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
          isCollapsed ? "w-[72px]" : "w-[280px]"
        )}
        style={{
          transitionProperty: 'width, transform',
        }}
      >
      {/* 1. Logo Area */}
      <div className="h-14 flex items-center px-4 border-b border-white/[0.08] overflow-hidden">
        <div className="flex items-center gap-2.5 min-w-0 w-full">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0">
            <Image src="/icon.svg" alt="PolyTier" width={40} height={40} className="object-contain" />
          </div>
          <span 
            className={cn(
              "font-bold tracking-tight text-sm whitespace-nowrap transition-all duration-500",
              isCollapsed 
                ? "opacity-0 max-w-0 overflow-hidden scale-95" 
                : "opacity-100 max-w-full scale-100"
            )}
          >
            <span className="text-white">POLY</span><span className="text-white">ANALYTICS</span>
          </span>
        </div>
      </div>

      {/* 3. Main Navigation */}
      <div className={cn("py-2", isCollapsed ? "px-0" : "px-3")}>
        <div 
          className={cn(
            "text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-2 transition-all duration-500",
            isCollapsed ? "opacity-0 h-0 overflow-hidden mb-0 scale-95 px-0" : "opacity-100 h-auto scale-100 px-3"
          )}
        >
          Menu
        </div>
        <Link href="/dashboard">
          <div 
            className={cn(
              "flex items-center gap-3 py-2 rounded-md text-sm transition-all mb-1 group relative",
              "hover:scale-[1.02] active:scale-[0.98]",
              isCollapsed ? "justify-center px-0" : "px-3",
              pathname === '/dashboard' 
                ? "bg-zinc-900 text-white font-medium" 
                : "text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200"
            )}
            title={isCollapsed ? "AI Chat" : undefined}
          >
            <MessageSquareText className="w-4 h-4 flex-shrink-0 transition-transform duration-300" />
            <span 
              className={cn(
                "transition-all duration-500 whitespace-nowrap",
                isCollapsed 
                  ? "opacity-0 max-w-0 overflow-hidden scale-95 absolute" 
                  : "opacity-100 max-w-full scale-100 relative"
              )}
            >
              AI Chat
            </span>
          </div>
        </Link>
        
        {navItems.map((item) => (
          <Link key={item.href} href={item.href}>
            <div 
              className={cn(
                "flex items-center gap-3 py-2 rounded-md text-sm transition-all mb-1 group relative",
                "hover:scale-[1.02] active:scale-[0.98]",
                isCollapsed ? "justify-center px-0" : "px-3",
                pathname === item.href 
                  ? "bg-zinc-900 text-white font-medium" 
                  : "text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200"
              )}
              title={isCollapsed ? item.name : undefined}
            >
              <item.icon className="w-4 h-4 flex-shrink-0 transition-transform duration-300" />
              <span 
                className={cn(
                  "transition-all duration-500 whitespace-nowrap",
                  isCollapsed 
                    ? "opacity-0 max-w-0 overflow-hidden scale-95 absolute" 
                    : "opacity-100 max-w-full scale-100 relative"
                )}
              >
                {item.name}
              </span>
            </div>
          </Link>
        ))}
      </div>

      {/* 4. History Section */}
      <div className={cn("py-2 flex-1 overflow-y-auto", isCollapsed ? "px-0" : "px-3")}>
        <div 
          className={cn(
            "text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-2 mt-4 transition-all duration-500",
            isCollapsed ? "opacity-0 h-0 overflow-hidden mb-0 mt-0 scale-95 px-0" : "opacity-100 h-auto scale-100 px-3"
          )}
        >
          Recent History
        </div>
        <div className="space-y-0.5">
          {chatsWithIcons.map((chat) => {
            const ChatIcon = chat.icon;
            return (
              <button 
                key={chat.id}
                onClick={handleChatClick}
                className={cn(
                  "flex w-full flex-col items-start py-2 rounded-md text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200 transition-all group relative",
                  "hover:scale-[1.02] active:scale-[0.98] cursor-pointer",
                  isCollapsed ? "justify-center items-center px-0" : "px-3 items-start"
                )}
                title={isCollapsed ? chat.title : undefined}
              >
                {isCollapsed ? (
                  <ChatIcon className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
                ) : (
                  <>
                    <span 
                      className={cn(
                        "text-sm truncate w-full text-left transition-all duration-500",
                        "opacity-100 scale-100"
                      )}
                    >
                      {chat.title}
                    </span>
                    <span 
                      className={cn(
                        "text-[10px] text-zinc-600 group-hover:text-zinc-500 transition-all duration-500",
                        "opacity-100 scale-100"
                      )}
                    >
                      {chat.date}
                    </span>
                  </>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* 5. User Profile Footer */}
      <div className="p-3 border-t border-white/[0.08] bg-[#0a0a0a]" ref={profileMenuRef}>
        {isCollapsed ? (
          // Collapsed state: Only show icon, click to open popup
          <div className="flex justify-center">
            <button
              onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
              className="w-10 h-10 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 transition-all duration-200 hover:scale-105 active:scale-95 shadow-sm"
              title="Profile Menu"
            >
              <User className="w-5 h-5" />
            </button>
          </div>
        ) : (
          // Expanded state: Show full profile, click to open popup
          <button
            onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
            className="w-full flex items-center gap-3 px-2 py-2 rounded-md hover:bg-zinc-900 transition-all duration-200 group"
          >
            <div className="w-10 h-10 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs font-medium text-zinc-300 flex-shrink-0 group-hover:border-zinc-600 transition-colors shadow-sm">
              <User className="w-5 h-5" />
            </div>
            <div className="flex-1 text-left overflow-hidden min-w-0">
              <div className="text-sm font-medium text-zinc-200 truncate">Pro Trader</div>
              <div className="text-[10px] text-emerald-500 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Connected
              </div>
            </div>
            <ChevronRight className={cn(
              "w-4 h-4 text-zinc-500 transition-transform duration-200 flex-shrink-0",
              isProfileMenuOpen && "rotate-90"
            )} />
          </button>
        )}
      </div>
      </aside>

      {/* Profile Popup Menu - Positioned relative to wrapper */}
      <AnimatePresence>
        {isProfileMenuOpen && (
          <motion.div
            ref={profilePopupRef}
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className={cn(
              "fixed w-56 bg-[#0a0a0a] border border-white/[0.08] rounded-lg shadow-2xl z-[10000] overflow-hidden",
              isCollapsed 
                ? "left-[88px]" // Position to the right of collapsed sidebar (72px + 16px margin)
                : "left-[296px]" // Position to the right of expanded sidebar (280px + 16px margin)
            )}
            style={{
              bottom: '24px', // Position above footer
            }}
          >
            {/* Menu Items */}
            <div className="py-2">
              <Link href="/dashboard/settings">
                <button
                  onClick={() => setIsProfileMenuOpen(false)}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-zinc-300 hover:bg-zinc-900 transition-colors duration-150 group"
                >
                  <Settings2 className="w-4 h-4 text-zinc-500 group-hover:text-blue-400 transition-colors" />
                  <span>Settings</span>
                </button>
              </Link>
            </div>

            {/* Sign Out Button - Red */}
            <div className="border-t border-white/[0.08] px-2 pt-2 pb-2">
              <button
                onClick={() => {
                  setIsProfileMenuOpen(false);
                  handleSignOut();
                }}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-all duration-150 rounded-md group"
              >
                <LogOut className="w-4 h-4 text-red-400 group-hover:text-red-300 transition-colors" />
                <span className="text-red-400 group-hover:text-red-300">Log out</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}