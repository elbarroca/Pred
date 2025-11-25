'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { motion, useScroll, useMotionValueEvent } from 'framer-motion';
import { Button } from '@/components/ui/button';

export default function Navbar() {
  const { scrollY } = useScroll();
  const [hidden, setHidden] = useState(false);

  useMotionValueEvent(scrollY, "change", (latest) => {
    const previous = scrollY.getPrevious() ?? 0;
    // Hide if scrolling down and past 100px
    if (latest > previous && latest > 100) {
      setHidden(true);
    } else {
      setHidden(false);
    }
  });

  return (
    <motion.nav
      variants={{
        visible: { y: 0 },
        hidden: { y: "-100%" },
      }}
      animate={hidden ? "hidden" : "visible"}
      transition={{ duration: 0.35, ease: "easeInOut" }}
      className="fixed top-0 w-full z-50 border-b border-white/5 bg-[#050505]/80 backdrop-blur-xl"
    >
      <div className="max-w-7xl mx-auto px-4 md:px-6 h-14 md:h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg flex items-center justify-center overflow-hidden">
            <Image src="/logo.png" alt="PolyTier" width={32} height={32} className="object-contain" />
          </div>
          <span className="font-bold tracking-tight text-base md:text-lg bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-500">
            PolyTier
          </span>
        </div>
        <div className="flex items-center gap-4 md:gap-6">
          <Link href="/dashboard">
            <Button variant="default" className="rounded-full px-4 md:px-6 h-8 md:h-9 text-xs md:text-sm">
              Launch App
            </Button>
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}