/** OpenAI Responses API request → internal `ChatMessage[]` + `ChatConfig`. */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type { ContentPart, ResponsesAPIRequest, ResponsesToolDefinition } from '../types.js';

/**
 * Resolve a message's `content` array into text + optional image bytes.
 *
 * Accepts both input-side (`input_text`, `input_image`) and replay-side
 * (`output_text`, `refusal`, `summary_text`) content parts. Clients that echo
 * prior assistant turns in `input[]` instead of using `previous_response_id`
 * (pi-ai, Codex) send `output_text` on assistant messages — rejecting those
 * would break cold-start replay. `input_image` with a base64 `data:` URL is
 * decoded to bytes; `http(s)://` URLs are not fetched (the mapper stays sync).
 */
function resolveMessageContent(
  content: string | ContentPart[],
  role: 'user' | 'assistant' | 'system',
): { text: string; images?: Uint8Array[] } {
  if (typeof content === 'string') return { text: content };

  const parts: string[] = [];
  const images: Uint8Array[] = [];
  // The internal `ChatMessage` shape is `{ content: string, images: Uint8Array[] }`
  // and the downstream Jinja serializer always emits `[{type:"text",...},
  // {type:"image"}*N]` — it cannot represent a text part that appears AFTER
  // an image part in the caller's content array. Detect and reject that
  // shape rather than silently reordering it and changing user intent.
  // Mirrors the existing rejection in `anthropic-request.ts` for the
  // tool_result + trailing-mixed case.
  let seenImage = false;

  for (const p of content) {
    if (p.type === 'input_text' || p.type === 'output_text' || p.type === 'summary_text') {
      if (seenImage) {
        throw new Error(
          'Unsupported: text content part after an image part in the same message is not representable ' +
            'in the internal message model. The flat ChatMessage shape and the Jinja serializer both place ' +
            'all text before all images in a user turn, so any mapping would silently reorder your content. ' +
            'Place all text parts before any image parts, or split across separate user turns.',
        );
      }
      parts.push(p.text);
    } else if (p.type === 'refusal') {
      if (seenImage) {
        throw new Error(
          'Unsupported: refusal content part after an image part in the same message is not representable ' +
            'in the internal message model; the flat ChatMessage shape would silently reorder it.',
        );
      }
      parts.push(p.refusal);
    } else if (p.type === 'input_image') {
      if (role !== 'user') {
        throw new Error(`input_image content parts are only allowed on user messages (got role="${role}")`);
      }
      if (p.file_id) {
        throw new Error('input_image.file_id is not supported — inline the image as a data URL');
      }
      if (!p.image_url) {
        throw new Error('input_image is missing image_url');
      }
      const match = /^data:[^;,]+;base64,(.+)$/s.exec(p.image_url);
      if (!match) {
        throw new Error(
          'input_image.image_url must be a base64 data URL (data:<mime>;base64,<payload>); ' +
            'remote http(s) URLs are not fetched by this server',
        );
      }
      // Wrap as a plain `Uint8Array` rather than storing the raw
      // `Buffer`. `Buffer` is a `Uint8Array` subclass, but it defines
      // its own `toJSON()` that `JSON.stringify` calls BEFORE any
      // replacer runs — so a Buffer-backed value would serialise as
      // `{type:"Buffer",data:[...]}` and skip the `__u8__` sentinel in
      // `stringifyStoredInputMessages`, corrupting image round-trip
      // through `previous_response_id` chains. A plain `Uint8Array`
      // has no `toJSON`, so the replacer fires as intended.
      images.push(new Uint8Array(Buffer.from(match[1], 'base64')));
      seenImage = true;
    } else {
      throw new Error(`Unsupported content part type: "${(p as { type: string }).type}"`);
    }
  }

  const out: { text: string; images?: Uint8Array[] } = { text: parts.join('') };
  if (images.length > 0) out.images = images;
  return out;
}

/** NAPI `ToolDefinition` requires `parameters.properties` to be a JSON string. */
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

export function mapRequest(req: ResponsesAPIRequest, priorMessages?: ChatMessage[]): MappedRequest {
  const messages: ChatMessage[] = [];

  if (req.instructions) {
    messages.push({ role: 'system', content: req.instructions });
  }

  if (priorMessages) {
    messages.push(...priorMessages);
  }

  // An assistant turn may serialise into any interleaving of `reasoning`,
  // `message` (assistant), and `function_call` items. We coalesce that run
  // into ONE assistant `ChatMessage` carrying `content` + `reasoningContent`
  // + `toolCalls`, matching the hot-path `ChatSession` shape exactly. Any
  // non-assistant item (user / system / function_call_output) flushes the
  // current turn. An assistant `message` item that appears AFTER a
  // `function_call` opens a fresh turn — preserving the pre-existing
  // convention documented in the fan-out tests.
  if (typeof req.input === 'string') {
    messages.push({ role: 'user', content: req.input });
  } else {
    let currentAssistant: ChatMessage | null = null;
    let assistantHasToolCalls = false;

    const flushAssistant = () => {
      if (currentAssistant) {
        messages.push(currentAssistant);
        currentAssistant = null;
        assistantHasToolCalls = false;
      }
    };
    const ensureAssistant = (): ChatMessage => {
      if (!currentAssistant) {
        currentAssistant = { role: 'assistant', content: '' };
      }
      return currentAssistant;
    };

    for (const item of req.input) {
      if (item == null || typeof item !== 'object') {
        throw new Error('Each input item must be a non-null object');
      }
      const itemType = (item as { type?: string }).type ?? 'message';

      if (itemType === 'message') {
        const msg = item as { role: string; content: string | ContentPart[] };
        // OpenAI "developer" maps to our "system".
        const role = msg.role === 'developer' ? 'system' : msg.role;
        if (role !== 'user' && role !== 'assistant' && role !== 'system') {
          throw new Error(`Unsupported message role: "${msg.role}"`);
        }

        if (role === 'assistant') {
          // `message` after a `function_call` opens a new turn.
          if (assistantHasToolCalls) {
            flushAssistant();
          }
          const a = ensureAssistant();
          const { text } = resolveMessageContent(msg.content, 'assistant');
          a.content = (a.content ?? '') + text;
        } else {
          flushAssistant();
          const { text, images } = resolveMessageContent(msg.content, role);
          const m: ChatMessage = { role, content: text };
          if (images) m.images = images;
          messages.push(m);
        }
      } else if (itemType === 'reasoning') {
        const r = item as { summary?: { text?: string }[] };
        const summary = (r.summary ?? []).map((s) => s.text ?? '').join('');
        const a = ensureAssistant();
        a.reasoningContent = (a.reasoningContent ?? '') + summary;
      } else if (itemType === 'function_call') {
        const fc = item as { name: string; arguments: string; call_id: string };
        const a = ensureAssistant();
        a.toolCalls ??= [];
        a.toolCalls.push({ name: fc.name, arguments: fc.arguments, id: fc.call_id });
        assistantHasToolCalls = true;
      } else if (itemType === 'function_call_output') {
        const fco = item as { call_id: string; output: string };
        flushAssistant();
        messages.push({
          role: 'tool',
          content: fco.output,
          toolCallId: fco.call_id,
        });
      } else {
        throw new Error(`Unsupported input item type: "${itemType as string}"`);
      }
    }

    flushAssistant();
  }

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
      // Caller disabled tool use.
    } else if (typeof req.tool_choice === 'object' && req.tool_choice?.type === 'function') {
      const targetName = req.tool_choice.name;
      const matched = req.tools.filter((t) => t.name === targetName);
      if (matched.length > 0) {
        config.tools = matched.map(mapTool);
      }
    } else {
      config.tools = req.tools.map(mapTool);
    }
  }
  if (priorMessages && priorMessages.length > 0) {
    config.reuseCache = true;
  }

  return { messages, config };
}

/**
 * Sentinel key used to tag base64-encoded `Uint8Array` payloads in
 * persisted `inputJson`. Plain `JSON.stringify` turns a `Uint8Array`
 * into a numeric-keyed object (e.g. `{"0":1,"1":2,...}`), which
 * (a) bloats the row ~8× vs base64 and (b) does not round-trip — the
 * parsed object fails the NAPI `Uint8Array` type check on cold replay,
 * breaking `previous_response_id` continuations that carry images.
 */
const UINT8_SENTINEL = '__u8__';

interface EncodedUint8Array {
  [UINT8_SENTINEL]: string;
}

function isEncodedUint8Array(value: unknown): value is EncodedUint8Array {
  return (
    value !== null &&
    typeof value === 'object' &&
    typeof (value as Record<string, unknown>)[UINT8_SENTINEL] === 'string'
  );
}

/**
 * Serialise a `ChatMessage[]` snapshot for `StoredResponseRecord.inputJson`,
 * preserving any `Uint8Array` image payloads as base64-encoded sentinels
 * so a later `reconstructMessagesFromChain` can revive them into real
 * `Uint8Array`s for the NAPI chat-session boundary.
 *
 * The replacer runs AFTER `toJSON`, so a `Buffer` (which defines
 * `Buffer.prototype.toJSON` returning `{type:"Buffer",data:[...]}`)
 * would otherwise slip past the `instanceof Uint8Array` check. We
 * match both shapes defensively — the production `resolveMessageContent`
 * path now wraps with `new Uint8Array(...)` at decode time, but any
 * future caller that sneaks a `Buffer` through still round-trips
 * instead of silently corrupting image state.
 */
export function stringifyStoredInputMessages(messages: ChatMessage[]): string {
  return JSON.stringify(messages, (_key, value: unknown) => {
    if (value instanceof Uint8Array) {
      return { [UINT8_SENTINEL]: Buffer.from(value).toString('base64') };
    }
    if (
      value !== null &&
      typeof value === 'object' &&
      (value as { type?: unknown }).type === 'Buffer' &&
      Array.isArray((value as { data?: unknown }).data)
    ) {
      const data = (value as { data: number[] }).data;
      return { [UINT8_SENTINEL]: Buffer.from(data).toString('base64') };
    }
    return value;
  });
}

/**
 * Reconstruct `ChatMessage[]` from a stored response chain. Each record
 * stores `inputJson` (messages sent) and `outputJson` (output items); we
 * interleave them.
 *
 * Image payloads encoded as `{__u8__: "<base64>"}` by
 * `stringifyStoredInputMessages` are rehydrated back into `Buffer`
 * (a `Uint8Array` subclass) so replayed user turns carry the same
 * runtime shape the native chat-session APIs expect.
 */
export function reconstructMessagesFromChain(chain: { inputJson: string; outputJson: string }[]): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const record of chain) {
    const inputMessages = JSON.parse(record.inputJson, (_key, value: unknown) => {
      if (isEncodedUint8Array(value)) {
        return Buffer.from(value[UINT8_SENTINEL], 'base64');
      }
      return value;
    }) as ChatMessage[];
    messages.push(...inputMessages);

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
    // Track presence vs. content separately: an empty-text `message` item
    // still represents a real successful turn (the hot-path `ChatSession`
    // always appends an assistant message per turn), so cold replay must
    // preserve it or `primeHistory` will reshape the conversation.
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

    // Preserve the assistant turn whenever the record carried any assistant-facing
    // item — message (even empty), reasoning, or function_call — because the hot-path
    // `ChatSession` always appends one assistant message per completed turn.
    // Keying on accumulated content would silently drop blank successful turns and
    // reshape the replayed conversation. Records with no assistant items (input-only)
    // are still skipped so we don't fabricate turns the live session never generated.
    if (hadMessageItem || hadReasoningItem || toolCalls.length > 0) {
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: assistantText,
      };
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
