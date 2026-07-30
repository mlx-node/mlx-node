import { describe, expect, it } from "vitest";

import type {
  ChatConfig,
  ChatMessage,
  ChatResult,
  ToolCallResult,
  ToolDefinition,
} from "@mlx-node/core";

import { ChatSession, type SessionCapableModel } from "../src/chat-session.js";
import type { ChatStreamEvent } from "../src/stream.js";

function chatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: "assistant reply",
    toolCalls: [],
    thinking: "private reasoning",
    thinkingEnabled: true,
    numTokens: 4,
    promptTokens: 10,
    reasoningTokens: 2,
    finishReason: "stop",
    rawText: "assistant reply",
    cachedTokens: 0,
    ...overrides,
  };
}

class RecordingModel implements SessionCapableModel {
  readonly startCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly continueCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly continueToolCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly results: ChatResult[] = [];

  async chatSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
  ): Promise<ChatResult> {
    this.startCalls.push({ messages, config });
    return this.nextResult();
  }

  async chatSessionContinue(
    messages: ChatMessage[],
    config?: ChatConfig | null,
  ): Promise<ChatResult> {
    this.continueCalls.push({ messages, config });
    return this.nextResult();
  }

  async chatSessionContinueTool(
    messages: ChatMessage[],
    config?: ChatConfig | null,
  ): Promise<ChatResult> {
    this.continueToolCalls.push({ messages, config });
    return this.nextResult();
  }

  chatStreamSessionStart(): AsyncGenerator<ChatStreamEvent> {
    return this.emptyStream();
  }

  chatStreamSessionContinue(): AsyncGenerator<ChatStreamEvent> {
    return this.emptyStream();
  }

  chatStreamSessionContinueTool(): AsyncGenerator<ChatStreamEvent> {
    return this.emptyStream();
  }

  resetCaches(): void {}

  private nextResult(): ChatResult {
    const result = this.results.shift();
    if (result === undefined)
      throw new Error("test model has no queued result");
    return result;
  }

  private async *emptyStream(): AsyncGenerator<ChatStreamEvent> {
    for (const event of [] as ChatStreamEvent[]) yield event;
  }
}

const tools: ToolDefinition[] = [
  {
    type: "function",
    function: {
      name: "lookup",
      description: "Look up a value",
      parameters: {
        type: "object",
        properties: '{"query":{"type":"string"}}',
        required: ["query"],
      },
    },
  },
];

describe("ChatSession template-rendered continuation history", () => {
  it("passes the complete replayable transcript and preserves thinking provenance", async () => {
    const model = new RecordingModel();
    model.results.push(
      chatResult({
        text: "first",
        thinking: "reason one",
        thinkingEnabled: true,
      }),
      chatResult({
        text: "second",
        thinking: undefined,
        thinkingEnabled: false,
      }),
    );
    const session = new ChatSession(model);

    await session.send("one", { config: { tools } });
    await session.send("two");

    expect(model.continueCalls).toHaveLength(1);
    expect(model.continueCalls[0]?.messages).toEqual([
      { role: "user", content: "one" },
      {
        role: "assistant",
        content: "first",
        reasoningContent: "reason one",
        thinkingEnabled: true,
      },
      { role: "user", content: "two" },
    ]);
    expect(model.continueCalls[0]?.config?.tools).toEqual(tools);
  });

  it("passes the declaring assistant call and structured tool result to the template", async () => {
    const model = new RecordingModel();
    const toolCall: ToolCallResult = {
      id: "call_1",
      name: "lookup",
      arguments: { query: "mlx" },
      status: "ok",
      rawContent: "",
    };
    model.results.push(
      chatResult({
        text: "",
        toolCalls: [toolCall],
        thinking: "need a lookup",
        thinkingEnabled: true,
      }),
      chatResult({ text: "done", thinking: undefined, thinkingEnabled: false }),
    );
    const session = new ChatSession(model);

    await session.send("look this up", { config: { tools } });
    await session.sendToolResult("call_1", "lookup failed", { isError: true });

    expect(model.continueToolCalls).toHaveLength(1);
    expect(model.continueToolCalls[0]?.messages).toEqual([
      { role: "user", content: "look this up" },
      {
        role: "assistant",
        content: "",
        toolCalls: [
          { id: "call_1", name: "lookup", arguments: '{"query":"mlx"}' },
        ],
        reasoningContent: "need a lookup",
        thinkingEnabled: true,
      },
      {
        role: "tool",
        content: "lookup failed",
        toolCallId: "call_1",
        isError: true,
      },
    ]);
    expect(model.continueToolCalls[0]?.config?.tools).toEqual(tools);
  });
});
