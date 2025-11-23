import { supabase } from '@/lib/supabase';
import { WalletAnalytics, ClosedPosition } from '@/types/database';

const ITEMS_PER_PAGE = 20;

export async function fetchWallets(
  page: number = 1,
  sortBy: keyof WalletAnalytics = 'realized_pnl',
  ascending = false,
  searchQuery = ''
) {
  const from = (page - 1) * ITEMS_PER_PAGE;
  const to = from + ITEMS_PER_PAGE - 1;

  let query = supabase
    .from('wallet_analytics')
    .select('*', { count: 'exact' })
    .order(sortBy, { ascending })
    .range(from, to);

  if (searchQuery) {
    query = query.ilike('proxy_wallet', `%${searchQuery}%`);
  }

  const { data, count, error } = await query;

  if (error) throw error;

  return {
    data: data as WalletAnalytics[],
    total: count || 0,
    totalPages: Math.ceil((count || 0) / ITEMS_PER_PAGE)
  };
}

export async function fetchWalletTrades(walletId: string, page: number = 1, limit: number = 50) {
  const from = (page - 1) * limit;
  const to = from + limit - 1;

  const { data, count, error } = await supabase
    .from('wallet_closed_positions')
    .select('*', { count: 'exact' })
    .eq('proxy_wallet', walletId)
    .order('timestamp', { ascending: false })
    .range(from, to);

  if (error) throw error;

  return {
    data: data as ClosedPosition[],
    total: count || 0,
    totalPages: Math.ceil((count || 0) / limit)
  };
}