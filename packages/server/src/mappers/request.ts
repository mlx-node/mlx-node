/**
 * Maps OpenAI Responses API request to internal ChatMessage[] + ChatConfig.
 */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type { ContentPart, ResponsesAPIRequest, ResponsesToolDefinition } from '../types.js';

/**
 * Resolve the text content of a message, which can be either a plain string
 * or an array of content parts.
 */
function resolveContent(content: string | ContentPart[]): string {
  if (typeof content === 'string') return content;
  const parts: string[] = [];
  for (const p of content) {
    if (p.type === 'input_text') {
      parts.push(p.text);
    } else {
      throw new Error(`Unsupported content part type: "${p.type as string}"`);
    }
  }
  return parts.join('');
}

/**
 * Map a Responses API tool definition to the internal ToolDefinition format.
 *
 * The NAPI layer requires `parameters.properties` to be a JSON string,
 * so we stringify the properties object here.
 */
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
 * Convert the Responses API request into internal ChatMessage[] + ChatConfig.
 */
export function mapRequest(req: ResponsesAPIRequest, priorMessages?: ChatMessage[]): MappedRequest {
  const messages: ChatMessage[] = [];

  // System instructions go first (before any history)
  if (req.instructions) {
    messages.push({ role: 'system', content: req.instructions });
  }

  // Prepend any prior conversation messages (from previous_response_id chain)
  if (priorMessages) {
    messages.push(...priorMessages);
  }

  // Map input
  if (typeof req.input === 'string') {
    messages.push({ role: 'user', content: req.input });
  } else {
    for (const item of req.input) {
      if (item == null || typeof item !== 'object') {
        throw new Error('Each input item must be a non-null object');
      }
      const itemType = item.type ?? 'message';

      if (itemType === 'message') {
        const msg = item as { role: string; content: string | ContentPart[] };
        // Map "developer" role to "system" (OpenAI convention)
        const role = msg.role === 'developer' ? 'system' : msg.role;
        if (role !== 'user' && role !== 'assistant' && role !== 'system') {
          throw new Error(`Unsupported message role: "${msg.role}"`);
        }
        messages.push({
          role,
          content: resolveContent(msg.content),
        });
      } else if (itemType === 'function_call') {
        // Reconstruct an assistant message with a tool call
        const fc = item as { name: string; arguments: string; call_id: string };
        messages.push({
          role: 'assistant',
          content: '',
          toolCalls: [{ name: fc.name, arguments: fc.arguments, id: fc.call_id }],
        });
      } else if (itemType === 'function_call_output') {
        // Tool result message
        const fco = item as { call_id: string; output: string };
        messages.push({
          role: 'tool',
          content: fco.output,
          toolCallId: fco.call_id,
        });
      } else {
        throw new Error(`Unsupported input item type: "${itemType as string}"`);
      }
    }
  }

  // Build ChatConfig
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
      // Don't pass any tools — user explicitly disabled tool use
    } else if (typeof req.tool_choice === 'object' && req.tool_choice?.type === 'function') {
      // Only pass the specifically named tool
      const targetName = req.tool_choice.name;
      const matched = req.tools.filter((t) => t.name === targetName);
      if (matched.length > 0) {
        config.tools = matched.map(mapTool);
      }
    } else {
      // 'auto', 'required', or unspecified — pass all tools
      config.tools = req.tools.map(mapTool);
    }
  }
  if (priorMessages && priorMessages.length > 0) {
    config.reuseCache = true;
  }

  return { messages, config };
}

/**
 * Reconstruct ChatMessage[] from a stored response chain.
 *
 * Each StoredResponseRecord contains `inputJson` (the messages sent)
 * and `outputJson` (the output items produced). We reconstruct the
 * conversation by interleaving input and output messages.
 */
export function reconstructMessagesFromChain(chain: { inputJson: string; outputJson: string }[]): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const record of chain) {
    // Add the original input messages
    const inputMessages = JSON.parse(record.inputJson) as ChatMessage[];
    messages.push(...inputMessages);

    // Reconstruct assistant message from output items
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
    const toolCalls: { name: string; arguments: string; id?: string }[] = [];

    for (const item of outputItems) {
      if (item.type === 'message' && item.content) {
        assistantText += item.content.map((c) => c.text).join('');
      } else if (item.type === 'reasoning' && item.summary) {
        thinkingText += item.summary.map((s) => s.text).join('');
      } else if (item.type === 'function_call') {
        toolCalls.push({
          name: item.name!,
          arguments: item.arguments!,
          id: item.call_id,
        });
      }
    }

    if (assistantText || toolCalls.length > 0) {
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
