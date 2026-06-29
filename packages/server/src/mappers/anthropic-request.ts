/** Anthropic Messages API request → internal `ChatMessage[]` + `ChatConfig`. */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type {
  AnthropicContentBlock,
  AnthropicCountTokensRequest,
  AnthropicImageContentBlock,
  AnthropicMessagesRequest,
  AnthropicTextContentBlock,
  AnthropicToolDefinition,
  SystemBlock,
} from '../types-anthropic.js';
import { anthropicToolUseIdToInternal } from './anthropic-response.js';
import { applyExtraBodyMtpOverrides } from './request.js';

export interface MappedAnthropicRequest {
  messages: ChatMessage[];
  config: ChatConfig;
  /**
   * Client-supplied stop strings (Anthropic `stop_sequences`), normalized to
   * drop absent/empty entries. Carried alongside `config` rather than on it
   * because `ChatConfig` has no native stop field; a downstream consumer is
   * responsible for honouring these.
   */
  stopSequences: string[];
}

/**
 * Anthropic billing/attribution header prefix. Claude Code injects a leading
 * system block of the shape `"x-anthropic-billing-header: cc_version=...; cch=<token>;"`
 * where the `cch=` token rotates per request. Leaving this in the prompt
 * defeats prefix caching at BOTH the warm-slot gate (`getOrCreateWarmAny`,
 * which compares `requestedSystem` byte-equally) AND the native
 * token-prefix verifier inside `chatSessionStart`. We mirror vLLM's strategy
 * (`vllm/entrypoints/anthropic/serving.py`, commit 262b76a0, 2026-03-11):
 * stateless, per-block, prefix-only — drop entirely BEFORE tokenization, so
 * the model never sees the rotating token AND the byte-prefix is stable.
 *
 * Hardcoded (not configurable) and intentionally limited to this single
 * prefix to mirror vLLM's exact behaviour. The string-system branch is left
 * UNFILTERED to match upstream.
 */
const ANTHROPIC_BILLING_HEADER_PREFIX = 'x-anthropic-billing-header';

/**
 * Canonicalize the Anthropic `system` field into the same string the mapper
 * bakes into the leading `system` ChatMessage. Used by both
 * `mapAnthropicRequest` (so the model never sees the billing header) and the
 * `/v1/messages` warm-slot gate cache-key derivation (so the gate matches
 * across rotating billing tokens). The two views MUST stay in sync — a
 * single source of truth prevents drift.
 *
 * Asymmetry vs. `mapAnthropicRequest`: the mapper THROWS on non-text blocks
 * (it's a request-validation gate), but this helper silently skips them.
 * Safe because `mapAnthropicRequest` runs first as a pre-flight gate, so by
 * the time the cache-key is computed, the request shape has already been
 * validated.
 */
export function canonicalizeSystemForCacheKey(system: AnthropicMessagesRequest['system']): string | null {
  if (system == null) return null;
  if (typeof system === 'string') return system;
  const parts: string[] = [];
  for (const b of system) {
    if (b.type === 'text' && !b.text.startsWith(ANTHROPIC_BILLING_HEADER_PREFIX)) {
      parts.push(b.text);
    }
  }
  // An array whose every block is stripped (e.g. a request whose only system
  // block is the rotating `x-anthropic-billing-header` line) is semantically
  // equivalent to "no system" — collapse to `null` so it (a) does NOT push an
  // empty `system` ChatMessage that the chat template wraps with two extra
  // `<|im_start|>system\n<|im_end|>\n` tokens (perturbing the prefix vs. an
  // absent-system request), and (b) compares byte-equal to `undefined` /
  // missing on the warm-slot gate (`SessionRegistry.getOrCreateWarmAny`
  // checks `entry.instructions !== requestedInstructions`, where `null !==
  // ''` would otherwise miss the slot).
  if (parts.length === 0) return null;
  return parts.join('');
}

/**
 * Resolve the text content of a `tool_result` block. The internal `ChatMessage`
 * shape (NAPI-generated) has no `images` field on `role: 'tool'`, so nested
 * images are rejected outright — any hoist-to-trailing-user workaround loses
 * both declared order and per-tool association once downstream canonicalization
 * reorders the tool rows. Callers must send images as a top-level image block
 * in a separate user turn.
 */
function resolveToolResultContent(content?: string | (AnthropicTextContentBlock | AnthropicImageContentBlock)[]): {
  text: string;
} {
  if (content == null) return { text: '' };
  if (typeof content === 'string') return { text: content };
  const parts: string[] = [];
  for (const b of content) {
    if (b.type === 'text') {
      parts.push(b.text);
    } else if (b.type === 'image') {
      throw new Error(
        'Unsupported: nested image content in tool_result blocks is not representable in the internal ' +
          'message model. Send the image as a top-level image block in a separate user turn, and reference ' +
          'it from the tool_result via text.',
      );
    } else {
      throw new Error(`Unsupported tool_result content type: "${(b as { type: string }).type}"`);
    }
  }
  return { text: parts.join('') };
}

/** NAPI `ToolDefinition` requires `parameters.properties` to be a JSON string. */
function mapTool(tool: AnthropicToolDefinition): ToolDefinition {
  const schema = tool.input_schema;
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: {
        type: typeof schema['type'] === 'string' ? schema['type'] : 'object',
        properties: JSON.stringify(schema['properties'] ?? {}),
        required: Array.isArray(schema['required']) ? (schema['required'] as string[]) : undefined,
      },
    },
  };
}

export function mapAnthropicRequest(
  req: AnthropicMessagesRequest | AnthropicCountTokensRequest,
): MappedAnthropicRequest {
  const messages: ChatMessage[] = [];

  // The leading system message is assembled and `unshift`ed AFTER the message
  // loop so that any `system`-role message folded out of `req.messages` (see
  // the `role === 'system'` branch below) is concatenated with the top-level
  // `system` field into a single leading system prompt. We compute the
  // top-level contribution here but defer the push.
  let topLevelSystem: string | null = null;
  if (req.system != null) {
    if (typeof req.system === 'string') {
      topLevelSystem = req.system;
    } else {
      // Validate first — throw early on unsupported block types so the
      // request fails fast (this is the validation gate). Stripping of
      // the rotating billing-header prefix happens inside
      // `canonicalizeSystemForCacheKey`, the single source of truth
      // shared with `endpoints/messages.ts`'s `requestedSystem` cache
      // key. Routing both call sites through the same helper means
      // any future change to the strip semantics (prefix list,
      // normalization, etc.) lands in exactly one place — the mapped
      // messages and the cache-key view cannot drift.
      for (const b of req.system as SystemBlock[]) {
        if (b.type !== 'text') {
          throw new Error(`Unsupported system block type: "${(b as { type: string }).type}"`);
        }
      }
      // Helper returns `null` when every block was stripped (e.g. a
      // request whose only system block is the rotating
      // `x-anthropic-billing-header` line). An all-stripped array is
      // semantically equivalent to "no system", so it contributes
      // nothing — emitting `{ role: 'system', content: '' }` would
      // otherwise have the chat template wrap it with two extra
      // `<|im_start|>system\n<|im_end|>\n` tokens, perturbing the
      // prefix vs. an absent-system request and breaking prefix
      // caching across the two semantically-equivalent shapes.
      topLevelSystem = canonicalizeSystemForCacheKey(req.system);
    }
  }

  // Text folded out of any `system`-role message(s) in `req.messages`, in
  // encounter order. Anthropic has no system role in `messages`, but Claude
  // Code hooks inject one; its content is positionless "additional context"
  // so we fold it into the leading system prompt regardless of position.
  const foldedSystemParts: string[] = [];

  for (const msg of req.messages) {
    const { role, content } = msg;

    if (role === 'user') {
      if (typeof content === 'string') {
        messages.push({ role: 'user', content });
      } else {
        // An Anthropic user turn may carry either pure text/image blocks,
        // or a contiguous prefix of `tool_result` blocks optionally followed
        // by trailing text/image blocks. Interleaving text/image BEFORE a
        // tool_result is rejected — we cannot preserve author intent and
        // fan-out ordering without silently reordering the caller's blocks.
        // Caller-relative order within a tool_result prefix is preserved;
        // `validateAndCanonicalizeHistoryToolOrder` reorders later if needed.
        const toolResults: {
          toolCallId: string;
          content: string;
          isError: boolean;
        }[] = [];
        const trailingText: string[] = [];
        const trailingImages: Uint8Array[] = [];
        let seenNonToolResult = false;
        let seenToolResult = false;
        // The flat `ChatMessage` shape cannot represent a text block that
        // appears AFTER an image block in the same turn — the downstream
        // Jinja serializer always places text before images. Reject that
        // interleaving up front rather than silently reordering it and
        // changing the caller's intent. This parallels the identical
        // guard in `request.ts:resolveMessageContent` for the
        // `/v1/responses` mapper.
        let seenImage = false;

        for (const block of content as AnthropicContentBlock[]) {
          if (block.type === 'tool_result') {
            if (seenNonToolResult) {
              throw new Error(
                'Unsupported: tool_result blocks must appear as a contiguous prefix of the user ' +
                  'turn, before any text or image blocks. Interleaving a text/image block before a ' +
                  'tool_result would require reordering the caller-supplied blocks and silently ' +
                  'changing authorship.',
              );
            }
            seenToolResult = true;
            const resolved = resolveToolResultContent(block.content);
            // Anthropic clients echo back the same `toolu_<uuid>` we
            // emitted on the prior assistant turn. Translate it back to
            // the internal `call_<uuid>` shape so the native session
            // store's tool_call_id lookup (which sees the original
            // `call_*` id) still matches. Ids that lack the `toolu_`
            // prefix (legacy callers that already speak the internal
            // shape) pass through unchanged.
            toolResults.push({
              toolCallId: anthropicToolUseIdToInternal(block.tool_use_id),
              content: resolved.text,
              isError: block.is_error === true,
            });
          } else if (block.type === 'text') {
            if (seenImage) {
              throw new Error(
                'Unsupported: text block after an image block in the same user turn is not representable ' +
                  'in the internal message model. The flat ChatMessage shape and the Jinja serializer both ' +
                  'place all text before all images, so any mapping would silently reorder your content. ' +
                  'Place all text blocks before any image blocks, or split across separate user turns.',
              );
            }
            seenNonToolResult = true;
            trailingText.push(block.text);
          } else if (block.type === 'image' && block.source.type === 'base64') {
            seenNonToolResult = true;
            seenImage = true;
            trailingImages.push(Buffer.from(block.source.data, 'base64'));
          } else {
            throw new Error(`Unsupported content block type: "${block.type}"`);
          }
        }

        if (seenToolResult) {
          // The structured `isError` field on the internal `ChatMessage`
          // is the authoritative signal of tool-call failure (mirroring
          // the existing `toolCallId` pattern). Pass `tr.content`
          // through verbatim — no JSON envelope, no in-band marker — and
          // surface the error condition via the dedicated structured
          // field. The Rust-side wire-format renderers (Jinja serializer
          // for the cold-start path, ChatML formatter for the fallback
          // template) inject a short model-facing `[tool error]` cue
          // into the prompt when `isError === true`, but the
          // `ChatMessage.content` itself stays byte-for-byte equal to
          // the original payload so a successful tool result whose
          // content happens to start with the same marker text cannot
          // be confused with an errored one on read-back. Neither
          // mlx-lm nor mlx-vlm have a precedent for an in-band marker
          // here; the structured-field approach matches how
          // `toolCallId` is plumbed and survives round-tripping cleanly.
          for (const tr of toolResults) {
            const msg: ChatMessage = {
              role: 'tool',
              content: tr.content,
              toolCallId: tr.toolCallId,
            };
            if (tr.isError) {
              msg.isError = true;
            }
            messages.push(msg);
          }
          // Trailing suffix after a tool_result prefix: accept either
          // (a) text-only (concatenated) or (b) exactly one image block.
          // Mixing text+image or multiple images would silently reorder
          // content in the flat NAPI `ChatMessage` shape.
          const hasTrailingText = trailingText.length > 0;
          const hasTrailingImages = trailingImages.length > 0;
          if (hasTrailingText && hasTrailingImages) {
            throw new Error(
              'Unsupported: mixing trailing text and image blocks after a tool_result prefix is not ' +
                'representable in the internal message model. The flat ChatMessage shape cannot preserve ' +
                'the caller-declared relative order of interleaved text and images, so any mapping would ' +
                'silently reorder your content. Send any commentary as part of the tool_result text, and ' +
                'deliver additional images in a separate follow-up user turn.',
            );
          }
          if (hasTrailingImages && trailingImages.length > 1) {
            throw new Error(
              'Unsupported: multiple trailing image blocks after a tool_result prefix are not ' +
                'representable in the internal message model without silently reordering the images ' +
                'relative to any surrounding text. Send at most one trailing image block, and deliver ' +
                'additional images in a separate follow-up user turn.',
            );
          }
          if (hasTrailingText || hasTrailingImages) {
            const trailingMsg: ChatMessage = { role: 'user', content: trailingText.join('') };
            if (hasTrailingImages) {
              trailingMsg.images = trailingImages;
            }
            messages.push(trailingMsg);
          }
        } else {
          // Pure text/image user turn — always emit exactly one `user` message, even if empty.
          const userMsg: ChatMessage = { role: 'user', content: trailingText.join('') };
          if (trailingImages.length > 0) {
            userMsg.images = trailingImages;
          }
          messages.push(userMsg);
        }
      }
    } else if (role === 'assistant') {
      if (typeof content === 'string') {
        messages.push({ role: 'assistant', content });
      } else {
        // Collapse into a single assistant message. The internal shape does not
        // support text-after-tool_use ordering, so interleaved shapes are rejected.
        let text = '';
        let reasoningContent: string | undefined;
        const toolCalls: { id: string; name: string; arguments: string }[] = [];
        let seenToolUse = false;

        for (const block of content as AnthropicContentBlock[]) {
          if (block.type === 'text') {
            if (seenToolUse) {
              throw new Error('Text blocks after tool_use blocks are not supported in assistant messages');
            }
            text += block.text;
          } else if (block.type === 'thinking') {
            reasoningContent = (reasoningContent ?? '') + block.thinking;
          } else if (block.type === 'tool_use') {
            seenToolUse = true;
            // Anthropic clients echo back the `toolu_<uuid>` we emitted
            // on the prior assistant turn. Translate to the internal
            // `call_<uuid>` shape so the native session store and the
            // Qwen chat template paths see a consistent id family.
            toolCalls.push({
              id: anthropicToolUseIdToInternal(block.id),
              name: block.name,
              arguments: JSON.stringify(block.input),
            });
          } else {
            throw new Error(`Unsupported assistant content block type: "${block.type}"`);
          }
        }

        const assistantMsg: ChatMessage = { role: 'assistant', content: text };
        if (reasoningContent != null) {
          assistantMsg.reasoningContent = reasoningContent;
        }
        if (toolCalls.length > 0) {
          assistantMsg.toolCalls = toolCalls;
        }
        messages.push(assistantMsg);
      }
    } else if (role === 'system') {
      // `system` is not a role in the Anthropic Messages spec, but Claude
      // Code's SessionStart hooks (e.g. superpowers) inject a
      // `{ role: 'system' }` message carrying "additional context" into the
      // `messages` array. Rather than rejecting the request (HTTP 400), fold
      // its text into the leading system prompt (assembled after the loop).
      //
      // CONTRACT — position-agnostic hoist (deliberate): a `system`-role
      // message is folded to the SINGLE leading system prompt regardless of
      // where it appears in `messages`. This is intentional, not incidental:
      //   1. The Anthropic wire format has no positional `system` role, so
      //      any `{ role: 'system' }` here is non-spec tooling injection with
      //      no defined positional semantics to preserve.
      //   2. The only known producer (Claude Code SessionStart hooks) emits
      //      positionless "additional context" — conceptually system-level,
      //      not a turn-point instruction.
      //   3. The internal `ChatMessage`/`primeHistory` pipeline represents
      //      only a SINGLE leading system message; a mid-history system turn
      //      is not representable, so hoisting is the sole non-rejecting
      //      option. Multiple system-role messages accumulate in encounter
      //      order (handled at assembly below).
      if (typeof content === 'string') {
        foldedSystemParts.push(content);
      } else {
        let text = '';
        for (const block of content as AnthropicContentBlock[]) {
          if (block.type !== 'text') {
            throw new Error(`Unsupported content block type "${block.type}" in system-role message`);
          }
          text += block.text;
        }
        foldedSystemParts.push(text);
      }
    } else {
      throw new Error(`Unsupported message role: "${role as string}"`);
    }
  }

  // Assemble the single leading system prompt from the top-level `system`
  // field plus any folded `system`-role message text. Joined with `'\n\n'`
  // between distinct contributions (a single string when there is only one),
  // so a request with ONLY a top-level system — the overwhelmingly common
  // case — stays byte-identical to the pre-folding behaviour.
  //
  // Empty folded contributions are dropped so an empty hook context message
  // ({ role: 'system', content: '' }) neither corrupts a real top-level
  // system prompt with a trailing `'\n\n'` separator nor synthesises a bare
  // empty system message. The top-level `system` field itself is preserved
  // verbatim (an explicit empty string still emits, matching prior behaviour).
  const systemParts: string[] = [];
  if (topLevelSystem !== null) {
    systemParts.push(topLevelSystem);
  }
  for (const part of foldedSystemParts) {
    if (part.length > 0) {
      systemParts.push(part);
    }
  }
  if (systemParts.length > 0) {
    messages.unshift({ role: 'system', content: systemParts.join('\n\n') });
  }

  const config: ChatConfig = {
    reportPerformance: true,
  };

  if (req.max_tokens != null) {
    config.maxNewTokens = req.max_tokens;
  }
  if (req.temperature != null) {
    config.temperature = req.temperature;
  }
  if (req.top_p != null) {
    config.topP = req.top_p;
  }
  if (req.top_k != null) {
    config.topK = req.top_k;
  }

  // `stop_sequences` has no `ChatConfig` field to map onto, so it rides out on
  // the widened return instead. Normalize to drop absent/null entries, empty
  // strings (which would match at every position and stop generation
  // immediately), and whitespace-only entries (which would truncate normal
  // output at the first space/newline; the real Anthropic API rejects these
  // with a 400, so making them a no-op is the lowest-risk resolution). A
  // downstream consumer honours the result.
  const stopSequences = (req.stop_sequences ?? []).filter((s) => typeof s === 'string' && s.trim().length > 0);

  if (req.tools && req.tools.length > 0) {
    const toolChoice = req.tool_choice;
    if (toolChoice?.type === 'tool') {
      // `{type:'tool', name:'X'}` is a HARD constraint: the model MUST call X
      // and only X. If the caller omitted the name, or named a tool that is
      // not in `req.tools`, falling through to the all-tools path would
      // silently violate that contract. Reject up front so the failure mode
      // is loud and the client gets a clear 400.
      if (!toolChoice.name) {
        throw new Error('tool_choice.type is "tool" but no name was provided');
      }
      const matched = req.tools.filter((t) => t.name === toolChoice.name);
      if (matched.length === 0) {
        throw new Error(
          `tool_choice references tool "${toolChoice.name}" which is not present in the request's tools list`,
        );
      }
      config.tools = matched.map(mapTool);
    } else {
      // `tool_choice` is undefined, `{type:'auto'}`, or `{type:'any'}` — all
      // three semantically mean "let the model pick from any tool", so we
      // forward the full tools array.
      config.tools = req.tools.map(mapTool);
    }
  }

  applyExtraBodyMtpOverrides(config, req.extra_body);

  return { messages, config, stopSequences };
}
