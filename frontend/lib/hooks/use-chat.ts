import { useEffect, useState, useRef, useCallback } from 'react';
import { ChatClient } from 'convo-ai-sdk';
import { toolRegistry } from '../tools/registry';
import { ToolCall, ChatMessage, ConnectionStatus, ToolInvocation } from '@/types/chat';

function getSessionId(): string {
  if (typeof window === 'undefined') return 'server_session';
  let id = localStorage.getItem('poly_session_id');
  if (!id) {
    id = `user_${crypto.randomUUID().slice(0, 8)}`;
    localStorage.setItem('poly_session_id', id);
  }
  return id;
}

export function useChat() {
  const apiKey = process.env.NEXT_PUBLIC_CONVO_API_KEY;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>(!apiKey ? 'error' : 'disconnected');
  const [isTyping, setIsTyping] = useState(false);
  
  const clientRef = useRef<ChatClient | null>(null);
  const currentMsgId = useRef<string | null>(null);
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current || !apiKey) return;
    initialized.current = true;

    const client = new ChatClient({
      apiKey: apiKey,
      identifier: getSessionId(),
      dynamicVariables: { app_context: "PolyAnalytics Dashboard" }
    });

    clientRef.current = client;

    // Event Handling
    client.on('statusChange', (s: string) => setStatus(s as ConnectionStatus));

    // Handle thought events (reasoning before tool calls)
    client.on('thought', (thought: string) => {
      let targetId = currentMsgId.current;
      if (!targetId) {
        targetId = crypto.randomUUID();
        currentMsgId.current = targetId;
        setMessages(prev => [...prev, { 
          id: targetId!, role: 'assistant', content: '', timestamp: Date.now(), toolInvocations: [], thoughts: []
        }]);
      }
      setMessages(prev => prev.map(msg => {
        if (msg.id === targetId) {
          return { ...msg, thoughts: [...(msg.thoughts || []), thought] };
        }
        return msg;
      }));
    });

    client.on('messageStart', () => {
      setIsTyping(true);
      // Only create a new ID if we aren't already in the middle of a tool sequence
      if (!currentMsgId.current) {
        const id = crypto.randomUUID();
        currentMsgId.current = id;
        setMessages(prev => [...prev, { 
          id, role: 'assistant', content: '', isStreaming: true, timestamp: Date.now(), toolInvocations: []
        }]);
      }
    });

    client.on('messageData', (chunk: string) => {
      if (!currentMsgId.current) return;
      setMessages(prev => prev.map(msg => 
        msg.id === currentMsgId.current ? { ...msg, content: msg.content + chunk } : msg
      ));
    });

    client.on('messageDone', () => {
      setIsTyping(false);
      setMessages(prev => prev.map(msg => 
        msg.id === currentMsgId.current ? { ...msg, isStreaming: false } : msg
      ));
      currentMsgId.current = null;
    });

    // --- ROBUST TOOL HANDLING ---
    client.on('toolCall', async (data: unknown) => {
      const toolCall = data as ToolCall;
      const toolFn = toolRegistry[toolCall.name];
      let result = "";

      // 1. Ensure we have a message bubble to show the "Invoking..." log
      let targetId = currentMsgId.current;
      if (!targetId) {
         targetId = crypto.randomUUID();
         currentMsgId.current = targetId;
         // Create placeholder assistant message
         setIsTyping(true);
         setMessages(prev => [...prev, { 
            id: targetId!, role: 'assistant', content: '', timestamp: Date.now(), toolInvocations: [], isStreaming: true
         }]);
      }

      // 2. Update UI to "Pending"
      setMessages(prev => prev.map(msg => {
         if (msg.id === targetId) {
            const newInvocations = [...(msg.toolInvocations || []), { 
               toolName: toolCall.name, status: 'pending', args: toolCall.args 
            } as ToolInvocation];
            return { ...msg, toolInvocations: newInvocations };
         }
         return msg;
      }));
      
      try {
        if (toolFn) {
          // 3. Execute
          result = await toolFn(toolCall.args);
          
          // 4. Update UI to "Complete"
          setMessages(prev => prev.map(msg => {
             if (msg.id === targetId) {
                const updatedInvocations = (msg.toolInvocations || []).map(inv => 
                   inv.toolName === toolCall.name ? { ...inv, status: 'complete' } : inv
                );
                return { ...msg, toolInvocations: updatedInvocations as ToolInvocation[] };
             }
             return msg;
          }));

          // 5. Check for Widget Data
          try {
            const parsed = JSON.parse(result);
            if (parsed && parsed.type && ['positions', 'funds', 'trade_result', 'pnl'].includes(parsed.type)) {
               setMessages(prev => [...prev, {
                 id: crypto.randomUUID(),
                 role: 'system',
                 content: '', 
                 timestamp: Date.now(),
                 widgetType: parsed.type as 'positions' | 'funds' | 'trade_result' | 'pnl',
                 widgetData: parsed
               }]);
            }
          } catch {
            // Not JSON or no widget type - that's okay
          }

        } else {
          result = JSON.stringify({ error: `Tool ${toolCall.name} not implemented` });
        }
      } catch (err: unknown) {
        result = JSON.stringify({ error: err instanceof Error ? err.message : String(err) });
        // Update UI to "Error"
        setMessages(prev => prev.map(msg => {
           if (msg.id === targetId) {
              const updatedInvocations = (msg.toolInvocations || []).map(inv => 
                 inv.toolName === toolCall.name ? { ...inv, status: 'error' } : inv
              );
              return { ...msg, toolInvocations: updatedInvocations as ToolInvocation[] };
           }
           return msg;
        }));
      }

      await client.sendToolResult(toolCall.id, toolCall.name, result);
    });

    client.connect().catch(console.error);

    return () => {
      client.disconnect();
      initialized.current = false;
    };
  }, [apiKey]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || !clientRef.current) return;
    
    setMessages(prev => [...prev, { 
      id: crypto.randomUUID(), role: 'user', content: text, timestamp: Date.now()
    }]);

    await clientRef.current.sendMessage(text);
  }, []);

  // NEW: Reset Functionality
  const resetSession = useCallback(() => {
    setMessages([]); // Clear UI
    if (clientRef.current) {
      console.log("🔄 Resetting Convo Session...");
      clientRef.current.resetChat(); // SDK Method: Disconnects, clears, reconnects
    }
  }, []);

  return { messages, status, isTyping, sendMessage, resetSession };
}