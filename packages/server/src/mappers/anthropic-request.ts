/**
 * Maps Anthropic Messages API request to internal ChatMessage[] + ChatConfig.
 */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type {
  AnthropicContentBlock,
  AnthropicImageContentBlock,
  AnthropicMessagesRequest,
  AnthropicTextContentBlock,
  AnthropicToolDefinition,
  SystemBlock,
} from '../types-anthropic.js';

export interface MappedAnthropicRequest {
  messages: ChatMessage[];
  config: ChatConfig;
}

/**
 * Resolve the text (and any nested image) content of a tool_result
 * block. Anthropic's Messages API allows tool_result blocks to carry a
 * plain string, an array of text blocks, or an array that mixes text
 * and image blocks — the latter is common for tools that return
 * screenshots or rendered PDFs and was silently rejected by the
 * iter-25 mapper. The internal `ChatMessage` shape has no per-tool
 * image field, so the mapper returns the extracted images separately
 * and the caller appends them to a trailing `user` ChatMessage that
 * follows the tool block.
 */
function resolveToolResultContent(content?: string | (AnthropicTextContentBlock | AnthropicImageContentBlock)[]): {
  text: string;
  images: Uint8Array[];
} {
  if (content == null) return { text: '', images: [] };
  if (typeof content === 'string') return { text: content, images: [] };
  const parts: string[] = [];
  const images: Uint8Array[] = [];
  for (const b of content) {
    if (b.type === 'text') {
      parts.push(b.text);
    } else if (b.type === 'image') {
      if (b.source.type !== 'base64') {
        throw new Error(`Unsupported tool_result image source: "${b.source.type as string}"`);
      }
      images.push(Buffer.from(b.source.data, 'base64'));
    } else {
      throw new Error(`Unsupported tool_result content type: "${(b as { type: string }).type}"`);
    }
  }
  return { text: parts.join(''), images };
}

/**
 * Map an Anthropic tool definition to the internal ToolDefinition format.
 *
 * The NAPI layer requires `parameters.properties` to be a JSON string,
 * so we stringify the properties object here.
 */
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

/**
 * Convert the Anthropic Messages API request into internal ChatMessage[] + ChatConfig.
 */
export function mapAnthropicRequest(req: AnthropicMessagesRequest): MappedAnthropicRequest {
  const messages: ChatMessage[] = [];

  // System prompt goes first
  if (req.system != null) {
    if (typeof req.system === 'string') {
      messages.push({ role: 'system', content: req.system });
    } else {
      // Array of SystemBlock — concatenate all text blocks
      const systemParts: string[] = [];
      for (const b of req.system as SystemBlock[]) {
        if (b.type === 'text') {
          systemParts.push(b.text);
        } else {
          throw new Error(`Unsupported system block type: "${(b as { type: string }).type}"`);
        }
      }
      const systemText = systemParts.join('');
      messages.push({ role: 'system', content: systemText });
    }
  }

  // Map each message in turn
  for (const msg of req.messages) {
    const { role, content } = msg;

    if (role === 'user') {
      if (typeof content === 'string') {
        messages.push({ role: 'user', content });
      } else {
        // A single Anthropic user turn may carry either
        //
        //   (a) ONLY text/image blocks — mapped to a single `user`
        //       ChatMessage, OR
        //   (b) a contiguous PREFIX of `tool_result` blocks,
        //       optionally followed by trailing text/image blocks —
        //       mapped to a contiguous `tool` ChatMessage run that
        //       sits immediately after the preceding assistant
        //       fan-out, optionally followed by a trailing `user`
        //       ChatMessage carrying the text/images.
        //
        // Shapes that interleave text/image BEFORE a tool_result
        // (e.g. `[text("ignore"), tool_result(...)]`) remain
        // rejected: we cannot preserve both author intent and
        // fan-out ordering without silently reordering the caller's
        // blocks. A text-or-image suffix after the tool_result
        // block is unambiguous (it belongs to the SAME user turn
        // that just answered the tool call) and is the shape the
        // iter-25 mapper was over-eagerly rejecting.
        //
        // Nested image blocks inside `tool_result.content` — common
        // for tools that return screenshots or rendered PDFs — are
        // extracted by `resolveToolResultContent` and also appended
        // to the trailing user message. The internal `ChatMessage`
        // shape has no per-tool image field, so these images
        // logically follow the tool result rather than accompanying
        // it.
        //
        // Within a tool_result prefix we preserve the caller's
        // relative order. Downstream
        // `validateAndCanonicalizeHistoryToolOrder` will reorder
        // against the assistant's declared sibling order if needed.
        const toolResults: {
          toolCallId: string;
          content: string;
          isError: boolean;
        }[] = [];
        const trailingText: string[] = [];
        const trailingImages: Uint8Array[] = [];
        // Images hoisted out of nested `tool_result.content` arrays.
        // They flow onto the trailing user message alongside any
        // top-level text/image content the caller authored.
        const hoistedToolImages: Uint8Array[] = [];
        let seenNonToolResult = false;
        let seenToolResult = false;

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
            toolResults.push({
              toolCallId: block.tool_use_id,
              content: resolved.text,
              isError: block.is_error === true,
            });
            if (resolved.images.length > 0) {
              hoistedToolImages.push(...resolved.images);
            }
          } else if (block.type === 'text') {
            seenNonToolResult = true;
            trailingText.push(block.text);
          } else if (block.type === 'image' && block.source.type === 'base64') {
            seenNonToolResult = true;
            trailingImages.push(Buffer.from(block.source.data, 'base64'));
          } else {
            throw new Error(`Unsupported content block type: "${block.type}"`);
          }
        }

        if (seenToolResult) {
          // Emit the tool block. `ChatMessage` has no `isError`
          // field — it is a NAPI-generated struct owned by Rust
          // and cannot carry an extra boolean without a schema
          // change. Encoding the Anthropic `tool_result.is_error`
          // flag into `content` is therefore unavoidable, but the
          // iter-23 `[tool error] ` prefix (iter-24 finding 2)
          // was ambiguous and lossy:
          //
          //   * A JSON tool payload is mutated into an invalid
          //     JSON string by the prefix (`[tool error] {"err":1}`
          //     parses as literal text, not an object).
          //   * A successful payload that naturally starts with
          //     `[tool error] ` is indistinguishable from an
          //     errored one.
          //   * An errored payload that already carries the
          //     prefix gets double-prefixed on round-trip.
          //
          // Replace the prefix with a JSON envelope when
          // `is_error === true`: `{ "is_error": true, "content":
          // <original> }`. JSON escaping makes the encoding
          // unambiguous and preserves the raw payload verbatim,
          // and a successful tool_result whose content is already
          // a JSON-shaped string is passed through untouched so
          // callers that stream structured data keep exact
          // fidelity. The encoding convention is: `is_error` is
          // represented on tool messages ONLY when true, ONLY via
          // this envelope shape; every other shape on the wire is
          // a successful tool result.
          for (const tr of toolResults) {
            const encoded = tr.isError ? JSON.stringify({ is_error: true, content: tr.content }) : tr.content;
            messages.push({
              role: 'tool',
              content: encoded,
              toolCallId: tr.toolCallId,
            });
          }
          // Append a trailing user message when the caller supplied
          // additional text/image content after the tool_result
          // prefix OR when any tool_result carried nested image
          // blocks that must flow onto a user turn. Both text and
          // image may be empty individually — we still emit the
          // message as long as AT LEAST ONE of the three sources
          // (trailing text, trailing images, hoisted tool images)
          // has content.
          const hasTrailingContent =
            trailingText.length > 0 || trailingImages.length > 0 || hoistedToolImages.length > 0;
          if (hasTrailingContent) {
            const trailingMsg: ChatMessage = { role: 'user', content: trailingText.join('') };
            const combinedImages: Uint8Array[] = [];
            if (trailingImages.length > 0) combinedImages.push(...trailingImages);
            if (hoistedToolImages.length > 0) combinedImages.push(...hoistedToolImages);
            if (combinedImages.length > 0) {
              trailingMsg.images = combinedImages;
            }
            messages.push(trailingMsg);
          }
        } else {
          // Pure text/image user turn. Always emit exactly one
          // `user` message, even when both arrays are empty
          // (matches the pre-iter-23 behavior for an empty
          // content array).
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
        // Combine all blocks into a single assistant message.
        // The internal ChatMessage format does not support mixed text/tool_use
        // ordering, so reject interleaved blocks rather than silently reordering.
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
            toolCalls.push({
              id: block.id,
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
    } else {
      throw new Error(`Unsupported message role: "${role as string}"`);
    }
  }

  // Build ChatConfig
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

  // Tool definition and choice mapping
  if (req.tools && req.tools.length > 0) {
    const toolChoice = req.tool_choice;
    if (toolChoice?.type === 'tool' && toolChoice.name) {
      // Only the named tool
      const matched = req.tools.filter((t) => t.name === toolChoice.name);
      if (matched.length > 0) {
        config.tools = matched.map(mapTool);
      }
    } else {
      // auto, any, or unspecified → pass all tools
      config.tools = req.tools.map(mapTool);
    }
  }

  return { messages, config };
}
