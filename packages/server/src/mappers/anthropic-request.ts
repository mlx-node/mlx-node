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
  return content
    .filter((b) => b.type === 'text')
    .map((b) => b.text)
    .join('');
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
      const systemText = (req.system as SystemBlock[])
        .filter((b) => b.type === 'text')
        .map((b) => b.text)
        .join('');
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
        // Collect text blocks and tool_result blocks separately
        const textParts: string[] = [];
        const toolResults: { toolCallId: string; content: string }[] = [];

        const images: Uint8Array[] = [];

        for (const block of content as AnthropicContentBlock[]) {
          if (block.type === 'text') {
            textParts.push(block.text);
          } else if (block.type === 'tool_result') {
            toolResults.push({
              toolCallId: block.tool_use_id,
              content: resolveToolResultContent(block.content),
            });
          } else if (block.type === 'image' && block.source.type === 'base64') {
            images.push(Buffer.from(block.source.data, 'base64'));
          }
          // Ignore other block types
        }

        // Emit user message if there is text or images
        if (textParts.length > 0 || images.length > 0) {
          const userMsg: ChatMessage = { role: 'user', content: textParts.join('') };
          if (images.length > 0) {
            userMsg.images = images;
          }
          messages.push(userMsg);
        }

        // Emit each tool_result as a separate tool message
        for (const tr of toolResults) {
          messages.push({ role: 'tool', content: tr.content, toolCallId: tr.toolCallId });
        }
      }
    } else if (role === 'assistant') {
      if (typeof content === 'string') {
        messages.push({ role: 'assistant', content });
      } else {
        // Combine all blocks into a single assistant message
        let text = '';
        let reasoningContent: string | undefined;
        const toolCalls: { id: string; name: string; arguments: string }[] = [];

        for (const block of content as AnthropicContentBlock[]) {
          if (block.type === 'text') {
            text += block.text;
          } else if (block.type === 'thinking') {
            reasoningContent = (reasoningContent ?? '') + block.thinking;
          } else if (block.type === 'tool_use') {
            toolCalls.push({
              id: block.id,
              name: block.name,
              arguments: JSON.stringify(block.input),
            });
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
