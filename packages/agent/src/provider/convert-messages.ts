/**
 * pi `Context` → native `ChatMessage[]` / `ToolDefinition[]` conversion.
 *
 * The provider bridge replays pi's full message history through
 * `ChatSession.primeHistory()` on every LLM call, so this conversion must
 * be deterministic and byte-stable: an unstable rendering (key-order
 * churn, nondeterministic joins) would change the token prefix between
 * replays and silently kill native KV-cache reuse.
 */

import type { Context, ImageContent, Message, TextContent, Tool } from '@earendil-works/pi-ai';
import type { ChatMessage, ToolDefinition } from '@mlx-node/lm';

const IMAGE_PLACEHOLDER = '[image omitted]';

function joinParts(parts: ReadonlyArray<TextContent | ImageContent>): string {
  return parts.map((part) => (part.type === 'image' ? IMAGE_PLACEHOLDER : part.text)).join('\n');
}

/** Per-message conversion (byte-stable joins). Never drops — the drop / orphan
 * repair lives in {@link contextToChatMessages}, mirroring pi's transformMessages. */
function convertMessage(message: Message): ChatMessage {
  switch (message.role) {
    case 'user':
      return {
        role: 'user',
        content: typeof message.content === 'string' ? message.content : joinParts(message.content),
      };
    case 'assistant': {
      // Thinking blocks are dropped: the native chat template re-renders
      // reasoning through its own <think> handling, and replayed thinking
      // would invalidate the KV prefix of every later turn.
      const text = message.content
        .filter((part): part is TextContent => part.type === 'text')
        .map((part) => part.text)
        .join('\n');
      const toolCalls = message.content
        .filter((part) => part.type === 'toolCall')
        .map((part) => ({ id: part.id, name: part.name, arguments: JSON.stringify(part.arguments) }));
      const converted: ChatMessage = { role: 'assistant', content: text };
      if (toolCalls.length > 0) converted.toolCalls = toolCalls;
      return converted;
    }
    case 'toolResult':
      return {
        role: 'tool',
        content: joinParts(message.content),
        toolCallId: message.toolCallId,
        isError: message.isError,
      };
  }
}

/**
 * Convert a pi `Context` into the `ChatMessage[]` accepted by
 * `ChatSession.primeHistory()`.
 *
 * - `systemPrompt` becomes the leading `system` message.
 * - Image parts become literal `[image omitted]` lines (v1 — no VLM plumbing).
 *
 * Two-pass mirror of pi's canonical `transformMessages` (pi-ai
 * `dist/api/transform-messages.js`). That transform normally sanitizes the
 * history INSIDE pi's built-in providers, but our custom `streamSimple` bypasses
 * it (and `defaultConvertToLlm` filters by role only), so the same two passes
 * must run here or a failed/interrupted turn reaches `primeHistory` unchanged:
 *
 *  1. DROP every assistant turn whose `stopReason` is `error` or `aborted` —
 *     partial or not. These incomplete turns (partial text, a half-emitted tool
 *     call) must not be replayed: after a native error (R2-3 resets the native
 *     cache) or an Esc/abort, priming the invalid partial turn garbles the
 *     continuation or leaves a dangling `<tool_call>` and corrupts the native
 *     `unresolvedOkToolCallCount`. A dropped turn's tool calls are NOT tracked.
 *  2. ORPHAN-REPAIR: track the tool-call ids of each RETAINED assistant and,
 *     before every following user/assistant message and at the end, synthesize a
 *     native tool result (`{ role: 'tool', content: 'No result provided',
 *     isError: true }`) for any tracked call with no matching `toolResult`
 *     (pi's `insertSyntheticToolResults`), so no assistant tool call is left
 *     unanswered in the primed history.
 *
 * The happy path (every assistant completes, every tool call answered) is
 * untouched, so the byte-stable joins that keep the replayed KV prefix stable
 * are preserved.
 */
export function contextToChatMessages(context: Context): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (context.systemPrompt) {
    messages.push({ role: 'system', content: context.systemPrompt });
  }

  // Orphan-repair state: the tool-call ids awaiting a result from the most
  // recent RETAINED assistant, and the result ids seen since.
  let pendingToolCallIds: string[] = [];
  let seenToolResultIds = new Set<string>();

  const flushOrphans = (): void => {
    if (pendingToolCallIds.length === 0) return;
    for (const id of pendingToolCallIds) {
      if (!seenToolResultIds.has(id)) {
        messages.push({ role: 'tool', content: 'No result provided', toolCallId: id, isError: true });
      }
    }
    pendingToolCallIds = [];
    seenToolResultIds = new Set();
  };

  for (const message of context.messages) {
    switch (message.role) {
      case 'user':
        flushOrphans();
        messages.push(convertMessage(message));
        break;
      case 'assistant': {
        flushOrphans();
        if (message.stopReason === 'error' || message.stopReason === 'aborted') {
          break; // dropped: not primed, and its tool calls are NOT tracked
        }
        const converted = convertMessage(message);
        messages.push(converted);
        if (converted.toolCalls && converted.toolCalls.length > 0) {
          // Native ToolCall.id is optional; only ids can be matched against a
          // tool result, so an id-less call is never tracked for orphan repair.
          pendingToolCallIds = converted.toolCalls.map((tc) => tc.id).filter((id): id is string => id !== undefined);
          seenToolResultIds = new Set();
        }
        break;
      }
      case 'toolResult':
        seenToolResultIds.add(message.toolCallId);
        messages.push(convertMessage(message));
        break;
    }
  }
  flushOrphans();
  return messages;
}

/**
 * Convert pi `Tool[]` (TypeBox-built plain JSON Schema objects) into the
 * native OpenAI-style `ToolDefinition[]`.
 *
 * The NAPI layer requires `parameters.properties` as a JSON string;
 * `JSON.stringify` preserves the schema's own key order, keeping the
 * rendered tool block byte-stable across replays. Returns `undefined`
 * for an absent or empty tool list so `ChatConfig.tools` stays unset.
 */
export function toolsToDefinitions(tools: Tool[] | undefined): ToolDefinition[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((tool) => {
    // pi's Tool.parameters is a TSchema — at runtime a plain JSON Schema
    // object (TypeBox kind markers live on symbols, which JSON ignores).
    const schema = tool.parameters as { properties?: Record<string, unknown>; required?: string[] };
    return {
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: 'object' as const,
          properties: JSON.stringify(schema.properties ?? {}),
          required: schema.required,
        },
      },
    };
  });
}
