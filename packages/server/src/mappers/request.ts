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
  // Consecutive `function_call` items from the same assistant turn
  // are coalesced into ONE assistant message with multi-element
  // `toolCalls`. The OpenAI Responses API serialises a multi-call
  // assistant response as a RUN of sibling `function_call` input
  // items, and iter-20's full-history walker (`responses.ts`,
  // `validateAndCanonicalizeHistoryToolOrder`) requires each
  // assistant fan-out's `toolCalls` array to match the trailing
  // tool block one-for-one. Pushing each `function_call` item as
  // its own `assistant` message would turn a single fan-out into
  // `assistant(call_a)`, `assistant(call_b)` — the walker would
  // then reject the first assistant turn as orphaned (its `next`
  // message is another assistant, not a tool), so stateless
  // multi-call replays would fail even when the caller shipped a
  // perfectly valid history.
  //
  // `prevItemType` is the exact coalescing invariant: a run ends
  // the moment the loop sees anything other than `function_call`.
  // The empty-content + existing-toolCalls check on the tail
  // assistant message is a belt-and-braces guard so a previously
  // pushed `message`-derived assistant turn can't accidentally
  // absorb a later function_call, but `prevItemType` is the
  // load-bearing predicate.
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
        // Reconstruct an assistant message with a tool call. Coalesce
        // onto the immediately preceding function_call item's
        // assistant turn when the previous input item was also a
        // function_call — see the block comment above this loop.
        const fc = item as { name: string; arguments: string; call_id: string };
        const last = messages[messages.length - 1];
        if (
          prevItemType === 'function_call' &&
          last !== undefined &&
          last.role === 'assistant' &&
          last.content === '' &&
          last.toolCalls !== undefined
        ) {
          last.toolCalls.push({ name: fc.name, arguments: fc.arguments, id: fc.call_id });
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
