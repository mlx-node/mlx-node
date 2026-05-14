import type { ToolCallResult } from '@mlx-node/core';

export type BrowserToolCall = {
  id?: string;
  name: string;
  arguments: string;
};

export type BrowserChatMessage = {
  role: string;
  content: string;
  images?: Uint8Array[];
  toolCalls?: BrowserToolCall[];
  toolCallId?: string;
};

function assistantToolCallsForHistory(toolCalls: readonly ToolCallResult[] | undefined): BrowserToolCall[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined;
  const calls: BrowserToolCall[] = [];
  for (const call of toolCalls) {
    if (call.status !== 'ok') continue;
    calls.push({
      id: call.id,
      name: call.name,
      arguments: typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments ?? {}),
    });
  }
  return calls.length > 0 ? calls : undefined;
}

function buildAssistantHistoryMessage(
  content: string,
  toolCalls: readonly ToolCallResult[] | undefined,
): BrowserChatMessage {
  const calls = assistantToolCallsForHistory(toolCalls);
  return calls ? { role: 'assistant', content, toolCalls: calls } : { role: 'assistant', content };
}

function countOkToolCalls(toolCalls: readonly ToolCallResult[] | undefined): number {
  if (!toolCalls || toolCalls.length === 0) return 0;
  let count = 0;
  for (const call of toolCalls) {
    if (call.status === 'ok') count++;
  }
  return count;
}

export class BrowserChatSessionAdapter {
  private readonly baseSystemPrompt: string;
  private readonly resolveSystemPrompt: (toolsEnabled: boolean) => string;
  private history: BrowserChatMessage[];
  private pendingTurn: BrowserChatMessage | null = null;
  private unresolvedOkToolCallCount: number | null = null;

  constructor(baseSystemPrompt: string, resolveSystemPrompt: (toolsEnabled: boolean) => string) {
    this.baseSystemPrompt = baseSystemPrompt;
    this.resolveSystemPrompt = resolveSystemPrompt;
    this.history = [{ role: 'system', content: baseSystemPrompt }];
  }

  get pendingUnresolvedToolCallCount() {
    return this.unresolvedOkToolCallCount;
  }

  reset() {
    this.history = [{ role: 'system', content: this.baseSystemPrompt }];
    this.pendingTurn = null;
    this.unresolvedOkToolCallCount = null;
  }

  latestUserContent() {
    const pending = this.pendingTurn?.role === 'user' ? this.pendingTurn.content : null;
    if (pending) return pending;
    for (let index = this.history.length - 1; index >= 0; index--) {
      const message = this.history[index];
      if (message?.role === 'user') return message.content;
    }
    return null;
  }

  beginUserTurn(message: BrowserChatMessage) {
    this.assertNoPendingTurn();
    if (this.unresolvedOkToolCallCount != null) {
      throw new Error(
        `BrowserChatSession: previous assistant turn has ${this.unresolvedOkToolCallCount} unresolved tool call(s)`,
      );
    }
    this.pendingTurn = message;
  }

  beginToolResultTurn(message: BrowserChatMessage) {
    this.assertNoPendingTurn();
    if (this.unresolvedOkToolCallCount !== 1) {
      throw new Error('BrowserChatSession: tool continuation requires exactly one unresolved tool call');
    }
    this.pendingTurn = message;
  }

  commitToolResult(message: BrowserChatMessage) {
    if (!this.unresolvedOkToolCallCount || this.unresolvedOkToolCallCount < 1) {
      throw new Error('BrowserChatSession: no unresolved tool call to resolve');
    }
    this.history.push(message);
    this.unresolvedOkToolCallCount -= 1;
    if (this.unresolvedOkToolCallCount === 0) {
      this.unresolvedOkToolCallCount = null;
    }
  }

  commitAssistantResult(content: string, toolCalls: readonly ToolCallResult[] | undefined) {
    if (this.pendingTurn) {
      this.history.push(this.pendingTurn);
      this.pendingTurn = null;
    }
    this.history.push(buildAssistantHistoryMessage(content, toolCalls));
    const unresolved = countOkToolCalls(toolCalls);
    this.unresolvedOkToolCallCount = unresolved > 0 ? unresolved : null;
  }

  rollbackPendingTurn() {
    this.pendingTurn = null;
  }

  requestMessages(toolsEnabled: boolean) {
    const snapshot = this.pendingTurn ? [...this.history, this.pendingTurn] : [...this.history];
    return snapshot.map((message, index) =>
      index === 0 && message.role === 'system'
        ? {
            ...message,
            content: this.resolveSystemPrompt(toolsEnabled),
          }
        : message,
    );
  }

  private assertNoPendingTurn() {
    if (this.pendingTurn) {
      throw new Error('BrowserChatSession: a turn is already in flight');
    }
  }
}
