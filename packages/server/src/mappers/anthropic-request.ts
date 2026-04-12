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
        // Emit messages in original block order, grouping consecutive text/image blocks
        let pendingText: string[] = [];
        let pendingImages: Uint8Array[] = [];

        const flushUserBlock = (): void => {
          if (pendingText.length > 0 || pendingImages.length > 0) {
            const userMsg: ChatMessage = { role: 'user', content: pendingText.join('') };
            if (pendingImages.length > 0) {
              userMsg.images = pendingImages;
            }
            messages.push(userMsg);
            pendingText = [];
            pendingImages = [];
          }
        };

        for (const block of content as AnthropicContentBlock[]) {
          if (block.type === 'text') {
            pendingText.push(block.text);
          } else if (block.type === 'image' && block.source.type === 'base64') {
            pendingImages.push(Buffer.from(block.source.data, 'base64'));
          } else if (block.type === 'tool_result') {
            // Flush any pending text/images before the tool result
            flushUserBlock();
            messages.push({
              role: 'tool',
              content: resolveToolResultContent(block.content),
              toolCallId: block.tool_use_id,
            });
          } else {
            throw new Error(`Unsupported content block type: "${block.type}"`);
          }
        }

        // Flush any remaining text/images
        flushUserBlock();
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
