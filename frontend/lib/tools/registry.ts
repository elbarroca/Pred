import { supabase } from '@/lib/supabase';

export type ToolFunction = (args?: Record<string, unknown>) => Promise<string>;

export const toolRegistry: Record<string, ToolFunction> = {
  /**
   * 🛠️ get_elite_positions
   */
  get_elite_positions: async (args?: Record<string, unknown>) => {
    const { category } = args as { category?: string } || {};
    try {
      const { data: elites } = await supabase.from('elite_traders').select('proxy_wallet').eq('tier', 'S').limit(5);
      const eliteIds = elites?.map(e => e.proxy_wallet) || [];

      let query = supabase
        .from('elite_open_positions')
        .select('*')
        .in('proxy_wallet', eliteIds)
        .order('unrealized_pnl', { ascending: false })
        .limit(3);

      if (category) query = query.ilike('event_category', `%${category}%`);

      const { data: positions } = await query;

      if (!positions || positions.length === 0) return JSON.stringify({ message: "No elite alpha found right now." });

      return JSON.stringify({
        type: 'positions', // SIGNAL FOR WIDGET
        data: positions.map(p => ({
          market: p.title,
          outcome: p.outcome,
          entry: (p.avg_entry_price || 0).toFixed(2),
          current: (p.current_price || 0).toFixed(2),
          whale: p.proxy_wallet.substring(0,4),
          roi: ((p.unrealized_pnl || 0) / (p.size || 1) * 100).toFixed(1)
        }))
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unknown error occurred';
      return JSON.stringify({ error: message });
    }
  },

  /**
   * 🛠️ get_available_funds
   */
  get_available_funds: async (args?: Record<string, unknown>) => {
    const { risk_tolerance } = args as { risk_tolerance?: string } || {};
    const allFunds = [
      { id: "v1", name: "Degen Crypto Index", description: "Leveraged bets on ETH/BTC.", risk: "High", roi_30d: 42.5, tvl: "2.1M" },
      { id: "v2", name: "Election Sniper", description: "Political prediction arbitrage.", risk: "Medium", roi_30d: 12.8, tvl: "850k" },
      { id: "v3", name: "Stable Arbitrage", description: "Low volatility stablecoin farming.", risk: "Low", roi_30d: 4.2, tvl: "5.4M" }
    ];

    let filtered = allFunds;
    if (risk_tolerance) {
      filtered = allFunds.filter(f => f.risk.toLowerCase() === risk_tolerance.toLowerCase());
    }

    return JSON.stringify({
      type: 'funds', // SIGNAL FOR WIDGET
      data: filtered,
      message: "Vaults retrieved."
    });
  },

  /**
   * 🛠️ execute_copy_trade
   */
  execute_copy_trade: async (args?: Record<string, unknown>) => {
    const { marketSlug, amount, outcome } = args as { marketSlug?: string; amount: number; outcome: string };
    await new Promise(resolve => setTimeout(resolve, 1000)); // Sim latency
    return JSON.stringify({
      type: 'trade_result', // SIGNAL FOR WIDGET
      status: 'success',
      txHash: '0x' + Math.random().toString(16).slice(2, 14),
      market: marketSlug || "Market ID",
      amount: amount,
      outcome: outcome,
      message: "Order filled."
    });
  },

  /**
   * 🛠️ get_pnl
   */
  get_pnl: async () => {
    return JSON.stringify({
      type: 'pnl', // SIGNAL FOR WIDGET
      daily: "+15.4%",
      all_time: "-4.2%",
      message: "PnL retrieved."
    });
  }
};