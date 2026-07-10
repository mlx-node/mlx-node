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

function convertMessage(message: Message): ChatMessage | null {
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
      const husk =
        (message.stopReason === 'aborted' || message.stopReason === 'error') && text === '' && toolCalls.length === 0;
      if (husk) return null;
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
 * - "Husk" assistant messages (stopReason `aborted`/`error` with no text and
 *   no tool calls) are skipped entirely: they carry nothing renderable and
 *   would only destabilize the replayed prefix.
 */
export function contextToChatMessages(context: Context): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (context.systemPrompt) {
    messages.push({ role: 'system', content: context.systemPrompt });
  }
  for (const message of context.messages) {
    const converted = convertMessage(message);
    if (converted) messages.push(converted);
  }
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
