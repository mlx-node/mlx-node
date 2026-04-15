/**
 * Maps Anthropic Messages API request to internal ChatMessage[] + ChatConfig.
 */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type {
  AnthropicContentBlock,
  AnthropicMessagesRequest,
  AnthropicToolDefinition,
  SystemBlock,
} from '../types-anthropic.js';

export interface MappedAnthropicRequest {
  messages: ChatMessage[];
  config: ChatConfig;
}

/**
 * Resolve the text content of a tool_result block, which can be a string,
 * an array of text blocks, or absent (empty string).
 */
function resolveToolResultContent(content?: string | { type: 'text'; text: string }[]): string {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  const parts: string[] = [];
  for (const b of content) {
    if (b.type === 'text') {
      parts.push(b.text);
    } else {
      throw new Error(`Unsupported tool_result content type: "${(b as { type: string }).type}"`);
    }
  }
  return parts.join('');
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
        // A single Anthropic user turn must carry either
        //
        //   (a) ONLY text/image blocks — mapped to a single `user`
        //       ChatMessage, OR
        //   (b) ONLY `tool_result` blocks — mapped to a contiguous
        //       `tool` block that sits immediately after the
        //       preceding assistant fan-out.
        //
        // Mixing the two in one turn is rejected (iter-23 finding
        // 3). The iter-22 mapper hoisted tool results to the front
        // and emitted the text/image content as a synthetic
        // trailing `user` message; that silently reordered
        // client-supplied blocks, so a turn like
        // `[text("ignore this result"), tool_result(...)]` would
        // reach the model as `tool(...)` followed by a new user
        // turn the caller never authored. Instead of canonicalizing
        // a lossy reorder we reject the mixed shape so the client
        // must split the turn into one message containing only
        // tool_result blocks and a separate message for the
        // follow-up text/images.
        //
        // Within a pure-tool-result turn we preserve the caller's
        // relative order. Downstream
        // `validateAndCanonicalizeHistoryToolOrder` will reorder
        // against the assistant's declared sibling order if needed.
        const toolResults: {
          toolCallId: string;
          content: string;
          isError: boolean;
        }[] = [];
        const pendingText: string[] = [];
        const pendingImages: Uint8Array[] = [];

        for (const block of content as AnthropicContentBlock[]) {
          if (block.type === 'text') {
            pendingText.push(block.text);
          } else if (block.type === 'image' && block.source.type === 'base64') {
            pendingImages.push(Buffer.from(block.source.data, 'base64'));
          } else if (block.type === 'tool_result') {
            toolResults.push({
              toolCallId: block.tool_use_id,
              content: resolveToolResultContent(block.content),
              isError: block.is_error === true,
            });
          } else {
            throw new Error(`Unsupported content block type: "${block.type}"`);
          }
        }

        const hasToolResults = toolResults.length > 0;
        const hasTextOrImage = pendingText.length > 0 || pendingImages.length > 0;
        if (hasToolResults && hasTextOrImage) {
          throw new Error(
            'Unsupported: a single user turn cannot mix tool_result blocks with text or image ' +
              'blocks. Split the turn into one message containing only tool_result blocks and a ' +
              'separate message for the follow-up text/images.',
          );
        }

        if (hasToolResults) {
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
        } else {
          // Pure text/image user turn. Always emit exactly one
          // `user` message, even when both arrays are empty
          // (matches the pre-iter-23 behavior for an empty
          // content array).
          const userMsg: ChatMessage = { role: 'user', content: pendingText.join('') };
          if (pendingImages.length > 0) {
            userMsg.images = pendingImages;
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
