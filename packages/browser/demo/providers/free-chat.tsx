import * as React from 'react';

export type ReasoningEffort = 'off' | 'low' | 'medium' | 'high';

export interface FreeChatContextValue {
  temperature: number;
  maxTokens: number;
  toolsEnabled: boolean;
  reasoningEffort: ReasoningEffort;
  generating: boolean;
  // Setters / actions:
  setTemperature: (v: number) => void;
  setMaxTokens: (v: number) => void;
  setToolsEnabled: (v: boolean) => void;
  cycleReasoning: () => void;
  sendMessage: () => void;
  // Refs / handles used by JSX:
  promptRef: React.RefObject<HTMLTextAreaElement | null>;
  chatRef: React.RefObject<HTMLDivElement | null>;
}

const FreeChatContext = React.createContext<FreeChatContextValue | null>(null);

export function FreeChatProvider({
  value,
  children,
}: {
  value: FreeChatContextValue;
  children: React.ReactNode;
}) {
  return <FreeChatContext.Provider value={value}>{children}</FreeChatContext.Provider>;
}

export function useFreeChat(): FreeChatContextValue {
  const ctx = React.useContext(FreeChatContext);
  if (!ctx) throw new Error('useFreeChat must be used within <FreeChatProvider>');
  return ctx;
}
