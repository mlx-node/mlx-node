/** OpenAI Responses API request → internal `ChatMessage[]` + `ChatConfig`. */

import type { ChatConfig, ChatMessage, ToolDefinition } from '@mlx-node/core';

import type { ContentPart, ResponsesAPIRequest, ResponsesToolDefinition } from '../types.js';

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

  // Coalesce a `message + function_call+` run (or a pure `function_call+` run)
  // into ONE assistant `ChatMessage` carrying both `content` and `toolCalls`.
  // `ChatSession.sendStream()` appends exactly one assistant message per turn,
  // and `validateAndCanonicalizeHistoryToolOrder` requires each fan-out's
  // `toolCalls` to pair 1:1 with the trailing tool block — splitting would
  // reshape the conversation and make the walker reject the turn as orphaned.
  // A `message` item immediately after a `function_call` starts a new turn.
  if (typeof req.input === 'string') {
    messages.push({ role: 'user', content: req.input });
  } else {
    let prevItemType: string | null = null;
    for (const item of req.input) {
      if (item == null || typeof item !== 'object') {
        throw new Error('Each input item must be a non-null object');
      }
      const itemType = item.type ?? 'message';

      if (itemType === 'message') {
        const msg = item as { role: string; content: string | ContentPart[] };
        // OpenAI "developer" maps to our "system".
        const role = msg.role === 'developer' ? 'system' : msg.role;
        if (role !== 'user' && role !== 'assistant' && role !== 'system') {
          throw new Error(`Unsupported message role: "${msg.role}"`);
        }
        messages.push({
          role,
          content: resolveContent(msg.content),
        });
      } else if (itemType === 'function_call') {
        // Coalesce onto the preceding assistant turn — see the loop header.
        const fc = item as { name: string; arguments: string; call_id: string };
        const last = messages[messages.length - 1];
        const canCoalesce =
          (prevItemType === 'function_call' || prevItemType === 'message') &&
          last !== undefined &&
          last.role === 'assistant';
        if (canCoalesce) {
          if (last!.toolCalls === undefined) {
            last!.toolCalls = [];
          }
          last!.toolCalls!.push({ name: fc.name, arguments: fc.arguments, id: fc.call_id });
        } else {
          messages.push({
            role: 'assistant',
            content: '',
            toolCalls: [{ name: fc.name, arguments: fc.arguments, id: fc.call_id }],
          });
        }
      } else if (itemType === 'function_call_output') {
        const fco = item as { call_id: string; output: string };
        messages.push({
          role: 'tool',
          content: fco.output,
          toolCallId: fco.call_id,
        });
      } else {
        throw new Error(`Unsupported input item type: "${itemType as string}"`);
      }

      prevItemType = itemType;
    }
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
 * Reconstruct `ChatMessage[]` from a stored response chain. Each record
 * stores `inputJson` (messages sent) and `outputJson` (output items); we
 * interleave them.
 */
export function reconstructMessagesFromChain(chain: { inputJson: string; outputJson: string }[]): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const record of chain) {
    const inputMessages = JSON.parse(record.inputJson) as ChatMessage[];
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
