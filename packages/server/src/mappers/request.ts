/** OpenAI Responses API request → internal `ChatMessage[]` + `ChatConfig`. */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import { decodeBase64Strict, isAllowedImageMediaType, MAX_REQUEST_IMAGES } from '../base64.js';
import { stringifyStrictJson } from '../strict-json.js';
import type { ContentPart, ResponsesAPIRequest, ResponsesToolDefinition } from '../types.js';

export { MAX_REQUEST_IMAGES };

export const MAX_CACHE_SALT_BYTES = 256;

export const VALID_REASONING_EFFORTS: ReadonlySet<string> = new Set([
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
]);

/**
 * A forced tool call (named function, `required`, or Anthropic `any`) is
 * rendered with thinking off: callers pair it with a small output cap, and
 * thinking spends that cap before the call is emitted (finish=length, no call).
 */
export function disableThinkingForForcedToolCall(config: ChatConfig): void {
  config.reasoningEffort = 'none';
  config.thinkingTokenBudget = 0;
  config.includeReasoning = false;
}

/** vLLM `ResponsesRequest.check_tool_usage`: these choices are a 400. */
export function validateToolChoice(req: Pick<ResponsesAPIRequest, 'tools' | 'tool_choice'>): void {
  const choice = req.tool_choice;
  const tools = req.tools ?? [];
  if (choice === 'required' && tools.length === 0) {
    throw new Error("Tool choice 'required' must be specified with 'tools' parameter.");
  }
  if (typeof choice === 'object' && choice !== null && choice.type === 'function') {
    if (!choice.name || !tools.some((t) => t.name === choice.name)) {
      throw new Error("Tool choice 'function' not found in 'tools' parameter.");
    }
  }
}

export function validateCacheSalt(cacheSalt: unknown): asserts cacheSalt is string {
  if (typeof cacheSalt !== 'string') {
    throw new Error('cache_salt must be a string');
  }
  if (cacheSalt.length === 0) {
    throw new Error('cache_salt must not be empty');
  }
  if (Buffer.byteLength(cacheSalt, 'utf8') > MAX_CACHE_SALT_BYTES) {
    throw new Error(`cache_salt must be at most ${MAX_CACHE_SALT_BYTES} UTF-8 bytes`);
  }
}

function decodeInputImagePart(p: Extract<ContentPart, { type: 'input_image' }>): Uint8Array {
  if (p.file_id) {
    throw new Error('input_image.file_id is not supported — inline the image as a data URL');
  }
  if (!p.image_url) {
    throw new Error('input_image is missing image_url');
  }
  const match = /^data:([^;,]+)(?:;[^;,]+)*;base64,(.*)$/is.exec(p.image_url);
  if (!match) {
    throw new Error(
      'input_image.image_url must be a base64 data URL (data:<mime>;base64,<payload>); ' +
        'remote http(s) URLs are not fetched by this server',
    );
  }
  if (!isAllowedImageMediaType(match[1])) {
    throw new Error('Could not process image');
  }
  // `decodeBase64Strict` returns a plain `Uint8Array` (not a `Buffer`) so
  // `JSON.stringify` does not invoke `Buffer.prototype.toJSON` before the
  // `__u8__` sentinel replacer in `stringifyStoredInputMessages`.
  return decodeBase64Strict(match[2], 'Could not process image');
}

/**
 * Resolve a message's `content` array into text + optional image bytes.
 *
 * Accepts both input-side (`input_text`, `input_image`) and replay-side
 * (`output_text`, `refusal`, `summary_text`) content parts. Clients that echo
 * prior assistant turns in `input[]` instead of using `previous_response_id`
 * (Codex and other agent harnesses) send `output_text` on assistant messages — rejecting those
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
      images.push(decodeInputImagePart(p));
      seenImage = true;
    } else {
      throw new Error(`Unsupported content part type: "${(p as { type: string }).type}"`);
    }
  }

  const out: { text: string; images?: Uint8Array[] } = { text: parts.join('') };
  if (images.length > 0) out.images = images;
  return out;
}

function resolveFunctionCallOutput(output: unknown): { text: string; images?: Uint8Array[] } {
  if (typeof output === 'string') return { text: output };
  if (!Array.isArray(output)) {
    throw new Error('function_call_output.output must be a string or an array of content parts');
  }
  const parts: string[] = [];
  const images: Uint8Array[] = [];
  for (const part of output as ContentPart[]) {
    if (part == null || typeof part !== 'object') {
      throw new Error('Each function_call_output content part must be an object');
    }
    if (part.type === 'input_text' || part.type === 'output_text' || part.type === 'summary_text') {
      parts.push(part.text);
    } else if (part.type === 'input_image') {
      images.push(decodeInputImagePart(part));
    } else {
      throw new Error(`Unsupported function_call_output part type: "${(part as { type: string }).type}"`);
    }
  }
  return images.length > 0 ? { text: parts.join(''), images } : { text: parts.join('') };
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

/**
 * Shared MTP-extension parser for the `extra_body` carrier on both
 * `/v1/responses` (OpenAI) and `/v1/messages` (Anthropic) request
 * shapes. Mutates the passed `config` in place:
 *
 *   * `generation_mode: "mtp"` → `enableMtp = true`
 *   * `generation_mode: "ar"`  → `enableMtp = false`
 *   * any other / absent value → `enableMtp` untouched, so the
 *     downstream `ChatSession.mergeConfig` auto-default applies.
 *     That default is two-part, not just "has a head": the model
 *     must report `hasMtpWeights()` AND opt in through
 *     `ChatSession#mtpAutoDefaultAllowed`, which prefers the native
 *     `mtpAutoEnabled()` getter when present and otherwise falls
 *     back to the `MTP_AUTO_DEFAULT_SUPPRESSED_MODELS` class-name
 *     list (both in `packages/lm/src/chat-session.ts`). NemotronH
 *     is currently suppressed — its head is a measured throughput
 *     loss — so `generation_mode: "mtp"` (or an explicit
 *     `enableMtp: true`) is the only way to speculate on it here.
 *
 *   * `mtp_depth: <positive int ≤ 64>` → `mtpDepth = value`
 *   * non-integer, out-of-range, or absent → `mtpDepth` untouched.
 *     The real clamps are per-family and owned by native
 *     `resolve_params`: qwen3.5 native MTP clamps to [1, 5], gemma4
 *     DSpark caps at the draft block size (7 on v1), and the gemma4
 *     assistant draft clamps to [1, 8]. The server therefore only
 *     rejects garbage — non-integers, non-positives, and values
 *     > 64 (a generous sanity ceiling far above any family's real
 *     clamp) — which saves a round-trip into the model thread. Qwen DFlash2
 *     uses its checkpoint block width and ignores this native-MTP override.
 *
 * Kept as a pure helper rather than inlined into each mapper so the
 * two endpoints can't drift in semantics.
 */
export function applyExtraBodyMtpOverrides(
  config: ChatConfig,
  extraBody: { generation_mode?: string | null; mtp_depth?: number | null } | undefined,
): void {
  if (!extraBody) return;
  const mode = extraBody.generation_mode;
  if (mode === 'mtp') {
    config.enableMtp = true;
  } else if (mode === 'ar') {
    config.enableMtp = false;
  }
  // Any other value (null, undefined, unknown string) → leave alone.

  const depth = extraBody.mtp_depth;
  if (depth != null && Number.isInteger(depth) && depth > 0 && depth <= 64) {
    config.mtpDepth = depth;
  }
}

export function mapRequest(req: ResponsesAPIRequest, priorMessages?: ChatMessage[]): MappedRequest {
  let totalImageCount = 0;
  if (priorMessages) {
    for (const m of priorMessages) {
      totalImageCount += m.images?.length ?? 0;
    }
  }
  if (Array.isArray(req.input)) {
    for (const item of req.input) {
      if (item == null || typeof item !== 'object') continue;
      const itemType = (item as { type?: string }).type ?? 'message';
      if (itemType === 'message') {
        const content = (item as { content?: unknown }).content;
        if (Array.isArray(content)) {
          for (const part of content) {
            if (part != null && typeof part === 'object' && (part as { type?: unknown }).type === 'input_image') {
              totalImageCount++;
            }
          }
        }
      } else if (itemType === 'function_call_output') {
        const output = (item as { output?: unknown }).output;
        if (Array.isArray(output)) {
          for (const part of output) {
            if (part != null && typeof part === 'object' && (part as { type?: unknown }).type === 'input_image') {
              totalImageCount++;
            }
          }
        }
      }
    }
  }
  if (totalImageCount > MAX_REQUEST_IMAGES) {
    throw new Error(`input: requests support at most ${MAX_REQUEST_IMAGES} images`);
  }

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
        const fco = item as { call_id: string; output: string | ContentPart[] };
        flushAssistant();
        const resolved = resolveFunctionCallOutput(fco.output);
        const toolMsg: ChatMessage = {
          role: 'tool',
          content: resolved.text,
          toolCallId: fco.call_id,
        };
        if (resolved.images) toolMsg.images = resolved.images;
        messages.push(toolMsg);
      } else {
        throw new Error(`Unsupported input item type: "${itemType as string}"`);
      }
    }

    flushAssistant();
  }

  const config: ChatConfig = {
    reportPerformance: true,
  };

  if (req.cache_salt != null) {
    validateCacheSalt(req.cache_salt);
    config.cacheSalt = req.cache_salt;
  }
  if (req.max_output_tokens != null) {
    config.maxNewTokens = req.max_output_tokens;
  }
  if (req.temperature != null) {
    config.temperature = req.temperature;
  }
  if (req.top_p != null) {
    config.topP = req.top_p;
  }
  if (req.reasoning?.effort != null) {
    if (typeof req.reasoning.effort !== 'string' || !VALID_REASONING_EFFORTS.has(req.reasoning.effort)) {
      throw new Error('Invalid reasoning.effort');
    }
    config.reasoningEffort = req.reasoning.effort as ChatConfig['reasoningEffort'];
  }
  validateToolChoice(req);
  if (req.tools && req.tools.length > 0) {
    if (typeof req.tool_choice === 'object' && req.tool_choice?.type === 'function') {
      const targetName = req.tool_choice.name;
      config.tools = req.tools.filter((t) => t.name === targetName).map(mapTool);
      disableThinkingForForcedToolCall(config);
    } else {
      config.tools = req.tools.map(mapTool);
      if (req.tool_choice === 'required') {
        disableThinkingForForcedToolCall(config);
      }
    }
  }
  if (priorMessages && priorMessages.length > 0) {
    config.reuseCache = true;
  }

  applyExtraBodyMtpOverrides(config, req.extra_body);

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
  return stringifyStrictJson(messages, (_key, value: unknown) => {
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
