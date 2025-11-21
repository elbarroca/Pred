export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export type WidgetType = 'positions' | 'funds' | 'trade_result' | 'pnl';

export interface ToolInvocation {
  toolName: string;
  status: 'pending' | 'complete' | 'error';
  args: Record<string, unknown>;
  result?: unknown;
}

// Widget data types
export interface ElitePosition {
  market: string;
  outcome: string;
  entry: string;
  current: string;
  whale: string;
  roi: string;
}

export interface AlphaPositionsData {
  type: 'positions';
  data: ElitePosition[];
}

export interface Fund {
  id: string;
  name: string;
  description: string;
  risk: 'High' | 'Medium' | 'Low';
  roi_30d: number;
  tvl: string;
}

export interface FundsData {
  type: 'funds';
  data: Fund[];
  message: string;
}

export interface TradeResultData {
  type: 'trade_result';
  status: string;
  txHash: string;
  market: string;
  amount: number;
  outcome: string;
  message: string;
}

export interface PnLData {
  type: 'pnl';
  daily: string;
  all_time: string;
  message: string;
}

export type WidgetData = AlphaPositionsData | FundsData | TradeResultData | PnLData;

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  isStreaming?: boolean;
  timestamp: number;

  // New: Track the reasoning steps
  toolInvocations?: ToolInvocation[];
  thoughts?: string[]; // Reasoning/thinking before tool calls

  // New: The final visual widget
  widgetType?: WidgetType;
  widgetData?: WidgetData;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';