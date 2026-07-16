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
const PI_NON_VISION_IMAGE_NOTE =
  '[Current model does not support images. The image will be omitted from this request.]';
const TOOL_RESULT_IMAGE_PLACEHOLDER = '(see attached image)';
const TOOL_RESULT_IMAGE_PROMPT = 'Attached image(s) from tool result:';

interface ConvertedParts {
  content: string;
  images?: Uint8Array[];
}

interface ConvertedMessage {
  message: ChatMessage;
  toolResultImages?: Uint8Array[];
}

/**
 * Convert Pi's mixed text/image blocks into the native message shape.
 *
 * Text-only models retain the historical byte-stable placeholder rendering.
 * Image-capable models keep text order and image order independently — the
 * most ordering the native `ChatMessage { content, images }` shape can express
 * — while decoding Pi's base64 payloads into the bytes consumed by NAPI.
 */
function convertParts(
  parts: ReadonlyArray<TextContent | ImageContent>,
  supportsImages: boolean,
  stripStaleToolImageNote = false,
): ConvertedParts {
  if (!supportsImages) {
    return {
      content: parts.map((part) => (part.type === 'image' ? IMAGE_PLACEHOLDER : part.text)).join('\n'),
    };
  }

  const text: string[] = [];
  const images: Uint8Array[] = [];
  for (const part of parts) {
    if (part.type === 'image') {
      images.push(Buffer.from(part.data, 'base64'));
    } else {
      // Pi added this exact standalone line to image tool results before the
      // loaded native capability could be published. A resumed pre-fix history
      // still contains it; replaying the warning contradicts the now-loaded
      // capability even when image processing failed before producing bytes.
      // Scope cleanup to tool results: identical direct-user text is literal.
      text.push(
        stripStaleToolImageNote
          ? part.text
              .split('\n')
              .filter((line) => line !== PI_NON_VISION_IMAGE_NOTE)
              .join('\n')
          : part.text,
      );
    }
  }
  return {
    content: text.join('\n'),
    ...(images.length > 0 ? { images } : {}),
  };
}

/** Per-message conversion (byte-stable joins). Never drops — the drop / orphan
 * repair and grouped tool-result image turn live in
 * {@link contextToChatMessages}, mirroring pi's transformMessages and OpenAI
 * provider conversion. */
function convertMessage(message: Message, supportsImages: boolean): ConvertedMessage {
  switch (message.role) {
    case 'user': {
      if (typeof message.content === 'string') {
        return { message: { role: 'user', content: message.content } };
      }
      return { message: { role: 'user', ...convertParts(message.content, supportsImages) } };
    }
    case 'assistant': {
      // Preserve the parser's reasoning body so thinking-capable templates can
      // reconstruct the exact channel/tag sequence generated on the prior
      // turn. The native Gemma4 parser already removes its fixed `thought\n`
      // channel label; the template adds that label back during replay.
      const reasoningContent = message.content
        .filter((part) => part.type === 'thinking')
        .map((part) => part.thinking)
        .join('');
      const text = message.content
        .filter((part): part is TextContent => part.type === 'text')
        .map((part) => part.text)
        .join('\n');
      const toolCalls = message.content
        .filter((part) => part.type === 'toolCall')
        .map((part) => ({ id: part.id, name: part.name, arguments: JSON.stringify(part.arguments) }));
      const converted: ChatMessage = { role: 'assistant', content: text };
      if (reasoningContent.length > 0) converted.reasoningContent = reasoningContent;
      const thinkingEnabled = (message as typeof message & { mlxThinkingEnabled?: boolean }).mlxThinkingEnabled;
      if (thinkingEnabled !== undefined) converted.thinkingEnabled = thinkingEnabled;
      if (toolCalls.length > 0) converted.toolCalls = toolCalls;
      return { message: converted };
    }
    case 'toolResult': {
      const converted = convertParts(message.content, supportsImages, true);
      const images = converted.images ?? [];
      return {
        message: {
          role: 'tool',
          content:
            converted.content.length > 0
              ? converted.content
              : images.length > 0
                ? TOOL_RESULT_IMAGE_PLACEHOLDER
                : converted.content,
          toolCallId: message.toolCallId,
          isError: message.isError,
        },
        ...(images.length > 0 ? { toolResultImages: images } : {}),
      };
    }
  }
}

/**
 * Convert a pi `Context` into the `ChatMessage[]` accepted by
 * `ChatSession.primeHistory()`.
 *
 * - `systemPrompt` becomes the leading `system` message.
 * - For text-only models (the default), image parts become literal
 *   `[image omitted]` lines.
 * - For an image-capable loaded model, user images stay on their native user
 *   message. Images from a consecutive tool-result run are decoded, collected
 *   in source order, and emitted on one synthetic user message after every
 *   textual tool message in that run. This mirrors pi's OpenAI conversion and
 *   avoids templates that ignore images attached to the `tool` role.
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
export function contextToChatMessages(context: Context, supportsImages = false): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (context.systemPrompt) {
    messages.push({ role: 'system', content: context.systemPrompt });
  }

  // Orphan-repair state: the tool-call ids awaiting a result from the most
  // recent RETAINED assistant, and the result ids seen since.
  let pendingToolCallIds: string[] = [];
  let seenToolResultIds = new Set<string>();
  let pendingToolResultImages: Uint8Array[] = [];

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

  const flushToolResultImages = (): void => {
    if (pendingToolResultImages.length === 0) return;
    messages.push({
      role: 'user',
      content: TOOL_RESULT_IMAGE_PROMPT,
      images: pendingToolResultImages,
    });
    pendingToolResultImages = [];
  };

  const flushToolResultBoundary = (): void => {
    // A grouped image attachment is logically a user turn. Repair any missing
    // sibling tool result before that boundary, then append the single image
    // turn after every real/synthetic tool result.
    flushOrphans();
    flushToolResultImages();
  };

  for (const message of context.messages) {
    switch (message.role) {
      case 'user':
        flushToolResultBoundary();
        messages.push(convertMessage(message, supportsImages).message);
        break;
      case 'assistant': {
        flushToolResultBoundary();
        if (message.stopReason === 'error' || message.stopReason === 'aborted') {
          break; // dropped: not primed, and its tool calls are NOT tracked
        }
        const converted = convertMessage(message, supportsImages).message;
        messages.push(converted);
        if (converted.toolCalls && converted.toolCalls.length > 0) {
          // Native ToolCall.id is optional; only ids can be matched against a
          // tool result, so an id-less call is never tracked for orphan repair.
          pendingToolCallIds = converted.toolCalls.map((tc) => tc.id).filter((id): id is string => id !== undefined);
          seenToolResultIds = new Set();
        }
        break;
      }
      case 'toolResult': {
        seenToolResultIds.add(message.toolCallId);
        const converted = convertMessage(message, supportsImages);
        messages.push(converted.message);
        if (converted.toolResultImages) {
          pendingToolResultImages.push(...converted.toolResultImages);
        }
        break;
      }
    }
  }
  flushToolResultBoundary();
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
