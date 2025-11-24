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

  if (!data || data.length === 0) {
    return {
      data: [] as WalletAnalytics[],
      total: count || 0,
      totalPages: 0
    };
  }

  // Fetch wallet metadata separately
  const walletAddresses = data.map(w => w.proxy_wallet);
  const { data: walletMetadata } = await supabase
    .from('wallets')
    .select('proxy_wallet, pseudonym, name, profile_image, bio')
    .in('proxy_wallet', walletAddresses);

  // Create a map for quick lookup
  const metadataMap = new Map(
    walletMetadata?.map(w => [w.proxy_wallet, w]) || []
  );

  // Merge analytics data with wallet metadata
  const mergedData = data.map(wallet => ({
    ...wallet,
    pseudonym: metadataMap.get(wallet.proxy_wallet)?.pseudonym || null,
    name: metadataMap.get(wallet.proxy_wallet)?.name || null,
    profile_image: metadataMap.get(wallet.proxy_wallet)?.profile_image || null,
    bio: metadataMap.get(wallet.proxy_wallet)?.bio || null,
  }));

  return {
    data: mergedData as WalletAnalytics[],
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