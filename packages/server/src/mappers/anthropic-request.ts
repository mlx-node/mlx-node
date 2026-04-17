/** Anthropic Messages API request → internal `ChatMessage[]` + `ChatConfig`. */

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

export function mapAnthropicRequest(req: AnthropicMessagesRequest): MappedAnthropicRequest {
  const messages: ChatMessage[] = [];

  if (req.system != null) {
    if (typeof req.system === 'string') {
      messages.push({ role: 'system', content: req.system });
    } else {
      const systemParts: string[] = [];
      for (const b of req.system as SystemBlock[]) {
        if (b.type === 'text') {
          systemParts.push(b.text);
        } else {
          throw new Error(`Unsupported system block type: "${(b as { type: string }).type}"`);
        }
      }
      messages.push({ role: 'system', content: systemParts.join('') });
    }
  }

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
          // `ChatMessage` (NAPI-generated) has no `isError` field, so
          // Anthropic's `tool_result.is_error=true` is encoded as a JSON
          // envelope `{ "is_error": true, "content": <original> }`. The
          // envelope preserves the raw payload verbatim (unlike a text
          // prefix, which would corrupt JSON payloads and collide with
          // strings that legitimately start with the prefix). Every other
          // wire shape is a successful tool result.
          for (const tr of toolResults) {
            const encoded = tr.isError ? JSON.stringify({ is_error: true, content: tr.content }) : tr.content;
            messages.push({
              role: 'tool',
              content: encoded,
              toolCallId: tr.toolCallId,
            });
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

  if (req.tools && req.tools.length > 0) {
    const toolChoice = req.tool_choice;
    if (toolChoice?.type === 'tool' && toolChoice.name) {
      const matched = req.tools.filter((t) => t.name === toolChoice.name);
      if (matched.length > 0) {
        config.tools = matched.map(mapTool);
      }
    } else {
      config.tools = req.tools.map(mapTool);
    }
  }

  return { messages, config };
}
