'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

export default function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        // Aggressive caching for instant loading
        staleTime: 10 * 60 * 1000, // 10 minutes - data considered fresh
        gcTime: 30 * 60 * 1000, // 30 minutes - keep in cache (was cacheTime)
        retry: 1, // Retry failed requests once
        refetchOnWindowFocus: false, // Don't refetch when window regains focus
        refetchOnReconnect: false, // Don't refetch when network reconnects
        refetchOnMount: false, // Don't refetch when component mounts
      },
    },
  }));

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}
