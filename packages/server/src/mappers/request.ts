/**
 * Maps OpenAI Responses API request to internal ChatMessage[] + ChatConfig.
 */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type { ContentPart, ResponsesAPIRequest, ResponsesToolDefinition } from '../types.js';

/**
 * Resolve the text content of a message, which can be either a plain string
 * or an array of content parts.
 */
function resolveContent(content: string | ContentPart[]): string {
  if (typeof content === 'string') return content;
  const parts: string[] = [];
  for (const p of content) {
    if (p.type === 'input_text') {
      parts.push(p.text);
    } else {
      throw new Error(`Unsupported content part type: "${p.type as string}"`);
    }
  }
  return parts.join('');
}

/**
 * Map a Responses API tool definition to the internal ToolDefinition format.
 *
 * The NAPI layer requires `parameters.properties` to be a JSON string,
 * so we stringify the properties object here.
 */
function mapTool(tool: ResponsesToolDefinition): ToolDefinition {
  if (tool.type !== 'function') {
    throw new Error(`Unsupported tool type: "${tool.type as string}"`);
  }
  const params = tool.parameters;
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: params
        ? {
            type: 'object',
            properties: params['properties'] ? JSON.stringify(params['properties']) : undefined,
            required: Array.isArray(params['required']) ? (params['required'] as string[]) : undefined,
          }
        : undefined,
    },
  };
}

export interface MappedRequest {
  messages: ChatMessage[];
  config: ChatConfig;
}

/**
 * Convert the Responses API request into internal ChatMessage[] + ChatConfig.
 */
export function mapRequest(req: ResponsesAPIRequest, priorMessages?: ChatMessage[]): MappedRequest {
  const messages: ChatMessage[] = [];

  // System instructions go first (before any history)
  if (req.instructions) {
    messages.push({ role: 'system', content: req.instructions });
  }

  // Prepend any prior conversation messages (from previous_response_id chain)
  if (priorMessages) {
    messages.push(...priorMessages);
  }

  // Map input.
  //
  // A single assistant turn that produced both text AND tool calls
  // is serialised by the OpenAI Responses API as a RUN of sibling
  // input items: `message` (with the text content) immediately
  // followed by one or more `function_call` items. The mapper must
  // coalesce that run into ONE `assistant` ChatMessage whose
  // `content` carries the text and whose `toolCalls` carries every
  // sibling call — both because:
  //
  //   1. The hot-path `ChatSession.sendStream()` in
  //      `packages/lm/src/chat-session.ts` appends a single
  //      assistant message per completed turn, carrying both text
  //      and `toolCalls`. The cold replay must reconstruct the same
  //      shape or `primeHistory()` would reshape the conversation
  //      and produce different model output.
  //   2. The iter-20 full-history walker
  //      (`validateAndCanonicalizeHistoryToolOrder`) requires each
  //      assistant fan-out's `toolCalls` array to match the
  //      trailing tool block one-for-one. Splitting the turn into
  //      `assistant(text)` then `assistant('', toolCalls=[...])`
  //      would make the walker reject the text-carrying assistant
  //      as orphaned (its `next` message is another assistant, not
  //      a tool).
  //
  // The coalescing rules are therefore:
  //
  //   * A `function_call` item ALWAYS attaches to the most recent
  //     assistant message, IF that message is the current run's
  //     head — detected via `prevItemType`. An assistant message
  //     pushed by a `message` item immediately before a
  //     `function_call` is the head of a `message + function_call+`
  //     run; a prior `function_call` that already coalesced is also
  //     the head (continuing a pure `function_call+` run from a
  //     previous iteration). In either case `last.toolCalls` is
  //     lazily initialised to `[]` before the push so the append is
  //     uniform regardless of whether the head was pushed as
  //     `assistant(text)` or `assistant('', toolCalls=[])`.
  //   * Any other `prevItemType` (or an empty history) starts a
  //     fresh assistant turn with `content = ''` and a new
  //     `toolCalls` array.
  //   * A `message` item that arrives RIGHT AFTER a
  //     `function_call` is NOT coalesced — it starts a new turn.
  //     That shape is a subsequent user/system turn, not a
  //     continuation of the fan-out.
  if (typeof req.input === 'string') {
    messages.push({ role: 'user', content: req.input });
  } else {
    let prevItemType: string | null = null;
    for (const item of req.input) {
      if (item == null || typeof item !== 'object') {
        throw new Error('Each input item must be a non-null object');
      }
      const itemType = item.type ?? 'message';

      if (itemType === 'message') {
        const msg = item as { role: string; content: string | ContentPart[] };
        // Map "developer" role to "system" (OpenAI convention)
        const role = msg.role === 'developer' ? 'system' : msg.role;
        if (role !== 'user' && role !== 'assistant' && role !== 'system') {
          throw new Error(`Unsupported message role: "${msg.role}"`);
        }
        messages.push({
          role,
          content: resolveContent(msg.content),
        });
      } else if (itemType === 'function_call') {
        // Reconstruct / extend an assistant message with a tool call.
        // Coalesce onto the preceding `message` or `function_call`
        // assistant turn — see the block comment above this loop.
        const fc = item as { name: string; arguments: string; call_id: string };
        const last = messages[messages.length - 1];
        const canCoalesce =
          (prevItemType === 'function_call' || prevItemType === 'message') &&
          last !== undefined &&
          last.role === 'assistant';
        if (canCoalesce) {
          if (last!.toolCalls === undefined) {
            last!.toolCalls = [];
          }
          last!.toolCalls!.push({ name: fc.name, arguments: fc.arguments, id: fc.call_id });
        } else {
          messages.push({
            role: 'assistant',
            content: '',
            toolCalls: [{ name: fc.name, arguments: fc.arguments, id: fc.call_id }],
          });
        }
      } else if (itemType === 'function_call_output') {
        // Tool result message
        const fco = item as { call_id: string; output: string };
        messages.push({
          role: 'tool',
          content: fco.output,
          toolCallId: fco.call_id,
        });
      } else {
        throw new Error(`Unsupported input item type: "${itemType as string}"`);
      }

      prevItemType = itemType;
    }
  }

  // Build ChatConfig
  const config: ChatConfig = {
    reportPerformance: true,
  };

  if (req.max_output_tokens != null) {
    config.maxNewTokens = req.max_output_tokens;
  }
  if (req.temperature != null) {
    config.temperature = req.temperature;
  }
  if (req.top_p != null) {
    config.topP = req.top_p;
  }
  if (req.reasoning?.effort) {
    config.reasoningEffort = req.reasoning.effort;
  }
  if (req.tools && req.tools.length > 0) {
    if (req.tool_choice === 'none') {
      // Don't pass any tools — user explicitly disabled tool use
    } else if (typeof req.tool_choice === 'object' && req.tool_choice?.type === 'function') {
      // Only pass the specifically named tool
      const targetName = req.tool_choice.name;
      const matched = req.tools.filter((t) => t.name === targetName);
      if (matched.length > 0) {
        config.tools = matched.map(mapTool);
      }
    } else {
      // 'auto', 'required', or unspecified — pass all tools
      config.tools = req.tools.map(mapTool);
    }
  }
  if (priorMessages && priorMessages.length > 0) {
    config.reuseCache = true;
  }

  return { messages, config };
}

/**
 * Reconstruct ChatMessage[] from a stored response chain.
 *
 * Each StoredResponseRecord contains `inputJson` (the messages sent)
 * and `outputJson` (the output items produced). We reconstruct the
 * conversation by interleaving input and output messages.
 */
export function reconstructMessagesFromChain(chain: { inputJson: string; outputJson: string }[]): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const record of chain) {
    // Add the original input messages
    const inputMessages = JSON.parse(record.inputJson) as ChatMessage[];
    messages.push(...inputMessages);

    // Reconstruct assistant message from output items
    const outputItems = JSON.parse(record.outputJson) as Array<{
      type: string;
      content?: Array<{ text: string }>;
      name?: string;
      arguments?: string;
      call_id?: string;
      summary?: Array<{ text: string }>;
    }>;

    let assistantText = '';
    let thinkingText = '';
    // Track item PRESENCE separately from accumulated content. The
    // server deliberately emits `message` items with empty text for
    // successful turns that produced no output, and `ChatSession`
    // hot-path history always appends an assistant message even when
    // `result.text === ''`. The predicate below must preserve those
    // blank successful turns on cold replay — see the block comment
    // on the predicate for the full rationale (iter-25 finding 3).
    let hadMessageItem = false;
    let hadReasoningItem = false;
    const toolCalls: { name: string; arguments: string; id?: string }[] = [];

    for (const item of outputItems) {
      if (item.type === 'message') {
        hadMessageItem = true;
        if (item.content) {
          assistantText += item.content.map((c) => c.text).join('');
        }
      } else if (item.type === 'reasoning') {
        hadReasoningItem = true;
        if (item.summary) {
          thinkingText += item.summary.map((s) => s.text).join('');
        }
      } else if (item.type === 'function_call') {
        toolCalls.push({
          name: item.name!,
          arguments: item.arguments!,
          id: item.call_id,
        });
      }
    }

    // Preserve the assistant turn whenever the stored record carried
    // ANY assistant-facing item — a `message` item (even one whose
    // text is empty), a `reasoning` item, or a `function_call`. The
    // iter-24 predicate keyed on accumulated content
    // (`assistantText.length > 0 || thinkingText.length > 0 ||
    // toolCalls.length > 0`), which silently dropped stored
    // successful turns whose `message` item carried empty text —
    // and such turns are a real stored shape, not an edge case:
    //
    //  * The server emits a `message` item with empty content when
    //    a turn completes with no tool calls and no text (e.g. a
    //    tool-result continuation where the model acknowledged the
    //    result but emitted nothing, or a turn interrupted at EOS).
    //  * `ChatSession` hot-path history always appends an assistant
    //    message for every completed turn, empty text included.
    //
    // After TTL expiry or a process restart, cold replay through
    // `reconstructMessagesFromChain` would drop the blank assistant
    // turn entirely, so the reconstructed history would primeHistory
    // a different conversation shape than the live session saw —
    // silently changing model output and corrupting any downstream
    // tool-call gate that walks the reconstructed trailing assistant
    // to compute outstanding tool-call ids (iter-25 finding 3).
    //
    // A record with NO assistant-facing items at all is still
    // skipped. This is a distinct shape from "empty message item
    // present" — it occurs on records that stored only the user
    // input and no output items, and re-emitting a synthetic
    // assistant there would clutter the replayed history with
    // fake turns that the live session never generated.
    if (hadMessageItem || hadReasoningItem || toolCalls.length > 0) {
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: assistantText,
      };
      // Emit `reasoningContent` only when the reasoning item
      // carried non-empty text. A present-but-empty reasoning
      // item (the server does not emit these today, but the
      // walker above would tolerate them) is preserved through
      // `hadReasoningItem` keeping the assistant turn alive; we
      // just omit the empty `reasoningContent` field to keep the
      // reconstructed message shape identical to a plain blank
      // successful turn.
      if (thinkingText) {
        assistantMsg.reasoningContent = thinkingText;
      }
      if (toolCalls.length > 0) {
        assistantMsg.toolCalls = toolCalls;
      }
      messages.push(assistantMsg);
    }
  }

  return messages;
}
